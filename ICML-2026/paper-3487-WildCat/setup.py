from setuptools import setup, find_packages

setup(
    name="wildcat",
    version="1.0.0",
    author="Tobias Schröder & Lester Mackey",
    author_email="",
    description="",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/microsoft/wildcat",  # Replace with your repo URL
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch",  # Pytorch package
    ],
    python_requires="<3.13",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # Update license type
        "Operating System :: OS Independent",
    ],
)
