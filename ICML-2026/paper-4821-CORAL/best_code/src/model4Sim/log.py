import os
import sys
import time
from datetime import datetime

def setup_logger(dataset_name, opt):
    """
    Redirect stdout to a log file and also log the argparse parameters.
    """
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"SASRec_{dataset_name}_{timestamp}.log"
    log_path = os.path.join(log_dir, log_filename)

    log_file = open(log_path, 'w')
    sys.stdout = log_file
    sys.stderr = log_file

    print("==================== Training Configuration ====================")
    for arg, value in vars(opt).items():
        print(f"{arg}: {value}")
    print("===============================================================")

class TqdmToFile:
    def __init__(self, file):
        self.file = file

    def write(self, message):
        if not message.isspace():
            self.file.write(message)
            self.file.flush()

    def flush(self):
        self.file.flush()

def redirect_tqdm_to_console():
    sys.stderr = sys.__stderr__