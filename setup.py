from setuptools import setup, find_packages

with open("README.md", "r") as fh:
	long_description = fh.read()

setup(
	name="morphotrack",
	version="1.0.1",
	author="Tatsuya Murakami",
	author_email="cunshang.m.tatsuya@gmail.com",
	description="This repository contains the code analyze morphogenic track in a cerebral cortex.",
	long_description=long_description,
	long_description_content_type="text/markdown",
	url="",
	packages=find_packages(),
	classifiers=[
		"Programming Language :: Python :: 3",
		"License :: OSI Approved :: MIT License",
		"Operating System :: OS Independent",
	]
)
