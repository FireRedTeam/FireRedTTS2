import argparse
import os
import pickle
import yaml
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import optuna
import json
import torch
from torch.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter

from accelerate import Accelerator

from posttrain_dataloader import create_dataloaders

from fireredtts2.llm.utils import load_model, WarmupDecayLR, summarize, get_grad_norm


def train(args: argparse.Namespace, config: dict, trial: optuna.Trial = None):
    """
    trial is only used when we are sweeping hyperparameters.
    """

    # accelerator
    accelerator = Accelerator()
    current_gpu = int(torch.cuda.current_device())
    device = accelerator.device  # "cuda"
    n_gpus = torch.cuda.device_count()
    print(
        f"---Number of GPUs: {n_gpus}",
        "---current_gpu:",
        current_gpu,
        "---device:",
        device,
    )

    # prepare log
    logs_folder = config["train"]["logs_folder"]
    if accelerator.is_main_process:
        writer = SummaryWriter(log_dir=logs_folder)

    print("---Load LLM Model...")
    model = load_model(config, args.checkpoint_path, device)

    trainloader, valloader = create_dataloaders(
        train_datasets=config["dataset"]["train_dataset_dir"],
        validation_datasets=config["dataset"]["valid_dataset_dir"],
        batch_size=config["train"]["batch_size"],
        device=device,
        infinite_train=False,
        num_workers=8,
    )

    eff_batch_size = config["train"]["batch_size"] * config["train"]["accumulate_num"]

    total_steps = (config["train"]["n_epochs"] * len(trainloader)) // config["train"][
        "accumulate_num"
    ]
    print("---total_steps:", total_steps)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["train"]["lr"],
        weight_decay=config["train"]["weight_decay"],
    )
    scheduler = WarmupDecayLR(
        optimizer,
        config["train"]["warmup_steps"],
        total_steps,
        config["train"]["lr_decay"],
    )

    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "effective_batch_size": eff_batch_size,
        "config": config,
        "args": args,
        "best_val_loss": float("inf"),
    }

    # accelerate prepare
    (model, trainloader, optimizer, scheduler) = accelerator.prepare(
        model, trainloader, optimizer, scheduler
    )

    # Training loop
    total_step = 1
    real_step = 1
    model.train()
    for epoch in range(config["train"]["n_epochs"]):
        step = 1
        total_loss = 0.0
        total_text_loss = 0.0
        total_c0_loss = 0.0
        total_c_loss = 0.0
        log_loss = []
        log_text_loss = []
        log_c0_loss = []
        log_c_loss = []
        for tokens, tokens_mask in trainloader:
            tokens, tokens_mask = tokens.to(device), tokens_mask.to(device)
            with accelerator.autocast():
                loss, text_loss, c0_loss, c_loss = model(tokens, tokens_mask)
                loss = loss / config["train"]["accumulate_num"]

                # only for logs
                text_loss = text_loss / config["train"]["accumulate_num"]
                c0_loss = c0_loss / config["train"]["accumulate_num"]
                c_loss = c_loss / config["train"]["accumulate_num"]

                total_loss += loss.item()
                total_text_loss += text_loss.item()
                total_c0_loss += c0_loss.item()
                total_c_loss += c_loss.item()

            # 梯度传导
            accelerator.backward(loss)

            # 到达累积步数时一次更新
            if (real_step) % config["train"]["accumulate_num"] == 0:
                grad_norm = get_grad_norm(model=model)
                accelerator.clip_grad_norm_(
                    model.parameters(), config["train"]["max_grad_norm"]
                )
                accelerator.wait_for_everyone()
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                accelerator.wait_for_everyone()

                if accelerator.is_main_process:
                    log_loss.append(total_loss)
                    log_text_loss.append(total_text_loss)
                    log_c0_loss.append(total_c0_loss)
                    log_c_loss.append(total_c_loss)

                    print(
                        "--device:",
                        device,
                        "---epoch:",
                        epoch,
                        "---real_step:",
                        real_step,
                        "---step:",
                        step,
                        "---total_step:",
                        total_step,
                        "---total_loss:",
                        round(total_loss, ndigits=2),
                        "---text_loss:",
                        round(total_text_loss, ndigits=2),
                        "--backbone_loss:",
                        round(total_c0_loss, ndigits=2),
                        "--decoder_loss:",
                        round(total_c_loss, ndigits=2),
                        # "---log_loss:",
                        # log_loss,
                        "---learning_rate:",
                        optimizer.param_groups[0]["lr"],
                        "---grad_norm:",
                        round(grad_norm, ndigits=2),
                    )

                    # write to tensorboard
                    if total_step % config["train"]["log_every"] == 0:
                        scalar_dict = {
                            "train/total_loss": sum(log_loss) / len(log_loss),
                            "train/text_loss": sum(log_text_loss) / len(log_text_loss),
                            "train/backbone_loss": sum(log_c0_loss) / len(log_c0_loss),
                            "train/decoder_loss": sum(log_c_loss) / len(log_c_loss),
                            "train/grad_norm": grad_norm,
                            "train/lr": optimizer.param_groups[0]["lr"],
                        }
                        summarize(
                            writer=writer,
                            global_step=total_step,
                            scalars=scalar_dict,
                        )
                        # 重置log loss
                        log_loss = []
                        log_text_loss = []
                        log_c0_loss = []
                        log_c_loss = []

                    # save models
                    if total_step % config["train"]["save_every"] == 0:
                        state["model"] = accelerator.get_state_dict(model=model)
                        torch.save(
                            state,
                            os.path.join(logs_folder, f"model_{total_step}.pt"),
                        )

                total_loss = 0.0
                total_text_loss = 0.0
                total_c0_loss = 0.0
                total_c_loss = 0.0

                step += 1
                total_step += 1

            real_step += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, default=None)

    args = parser.parse_args()

    # load config
    config = json.load(open(args.config_path))
    print("---config:\n", config)
    os.makedirs(config["train"]["logs_folder"], exist_ok=True)

    final_val_loss = train(args, config)
