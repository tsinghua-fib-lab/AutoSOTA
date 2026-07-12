import sys
import os

class Logger:
    def __init__(self, training=0, log_dir="logs"):
        # set training mode
        self.training = training
        
        # create log directory if it doesn't exist
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # choose log file based on training variable
        log_file = "train.log" if self.training == 1 else "test.log"
        self.log_file_path = os.path.join(log_dir, log_file)
        
        # open file for writing logs
        self.log_file = open(self.log_file_path, "w+")
        
        # save original sys.stdout for restoration later
        self.original_stdout = sys.stdout
        
        # redirect print to log file
        sys.stdout = self

    def write(self, message):
        """Write message to log file"""
        if message != '\n':  # Avoid writing empty lines
            while message.endswith('\n'): message.pop()  # Remove trailing newline for consistency
            self.log_file.write(message + '\n')
            self.log_file.flush()  # Ensure timely writing to file

    def flush(self):
        """Flush the buffer"""
        self.log_file.flush()

    def close(self):
        """Close the file and restore sys.stdout"""
        sys.stdout = self.original_stdout
        self.log_file.close()


# Usage example
if __name__ == "__main__":
    training = 1  # set training mode, 0 for testing, 1 for training
    logger = Logger(training=training)  # create Logger instance

    print("This is a test message.")  # this message will be output to different log files based on training mode

    logger.close()  # close logger at the end