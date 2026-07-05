from setuptools import setup, find_packages

# Read dependencies from requirements.txt
with open("requirements.txt", encoding="utf-8") as f:
    install_requires = [
        line.strip()
        for line in f
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="pyciphod",
    version="0.1",
    author="Charles Assaad, Federico Baldo, Simon Ferreira, Timothée Loranchet",
    author_email="charles.assaad@inserm.fr",
    description="A Python package for causal discovery, causal inference, and root cause analysis",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=install_requires,
)