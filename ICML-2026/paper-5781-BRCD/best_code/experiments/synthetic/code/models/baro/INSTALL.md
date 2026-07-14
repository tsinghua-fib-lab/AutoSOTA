# Installation

We assume the users already satisfy the [REQUIREMENTS.md](REQUIREMENTS.md) and ready to have Python 3.10 installed. Then, users can install BARO from PyPI or build BARO from source. In addition, users who familiar with Continuous Integration (CI) can take a look at our [build-and-test.yml](https://github.com/phamquiluan/baro/blob/main/.github/workflows/build-and-test.yml) configuration to see how we install and test our BARO on Linux and Windows machine from Python 3.7 to 3.12.

**Table of contents**
  * [Installation Instruction](#installation-instruction)
    + [Install Python 3.10](#install-python-310)
    + [Clone BARO from GitHub](#clone-baro-from-github)
    + [Create and activate a virtual environment](#create-and-activate-a-virtual-environment)
    + [Install BARO from PyPI or Build BARO from source](#install-baro-from-pypi-or-build-baro-from-source)
  * [Test the installation](#test-the-installation)
  * [Basic usage example](#basic-usage-example)
 
## Installation Instruction

### Install Python 3.10

```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update -y
sudo apt-get install -y python3.10 python3.10-dev python3.10-venv
```

### Clone BARO from GitHub


```bash
git clone https://github.com/phamquiluan/baro.git && cd baro
```


### Create and activate a virtual environment

```
# create a virtual environment
python3.10 -m venv env

# activate the environment
. env/bin/activate
```

### Install BARO from PyPI or Build BARO from source

```bash
# install BARO from PyPI
pip install fse-baro

# build BARO from source
pip install -e .
```

## Test the installation

Users can perform testing using the following commands:

```bash
pytest tests/test.py
```

<details>
<summary>The expected output would look like this</summary>

```bash

(ins)(env) luan@machine:~/ws/baro$ pytest tests/test.py 
============================================ test session starts =============================================
platform linux -- Python 3.10.13, pytest-7.4.0, pluggy-1.3.0
rootdir: /home/luan/ws/baro
collected 4 items                                                                                            

tests/test.py ....                                                                                     [100%]

======================================= 4 passed in 501.44s (0:08:21) ========================================
(ins)(env) luan@machine:~/ws/baro$ 

```
</details>

## Basic usage example

Users can check a basic usage example of BARO in the [README.md#basic-usage-example](https://github.com/phamquiluan/baro/blob/main/README.md#how-to-use) section.
