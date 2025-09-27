from pathlib import Path
from setuptools import setup, find_packages


HERE = Path(__file__).parent
long_description = (HERE / "README.md").read_text()

setup(
	name="fireredtts2",
	version="0.1.0",
	description="FireRedTTS2 - speech generation utilities and model wrapper",
	long_description=long_description,
	long_description_content_type="text/markdown",
	author="FireRedTTS2",
	packages=find_packages(exclude=("tests", "examples")),
	include_package_data=True,
	install_requires=[
		"torchtune",
		"torchao",
		"transformers",
		"einops",
		"gradio",
	],
	python_requires=">=3.8",
)
