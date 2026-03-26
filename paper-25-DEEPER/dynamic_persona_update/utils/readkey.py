import os
import sys
cur_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, cur_path+"/..")

import configparser

def readkey():
    config_path = "utils/apikey.ini"
    config = configparser.ConfigParser()
    config.read(config_path)
    return config

def set_env():
    config = readkey()
    os.environ["OPENAI_API_KEY"] = config["OpenAI"]["OPENAI_API_KEY"] 