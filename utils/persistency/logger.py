import traceback
from datetime import datetime


class Logger:
    def __init__(self, log_file):
        self.log_file = log_file
        with open(self.log_file, 'a') as f:
            f.write(f"Log file: {str(self.log_file)}\n"
                    f"Created on: {str(datetime.now()).split('.')[0]}\n"
                    f"-------------------------------------------------------------------------------------------\n\n")

    def log(self, message):
        print(message)
        with open(self.log_file, 'a') as f:
            f.write(message + '\n')

    def end_log(self):
        with open(self.log_file, 'a') as f:
            f.write(f"\n\n-----------------------------------------------------------------------------------------\n\n"
                    f"End of Log File.\n"
                    f"Terminated on: {str(datetime.now()).split('.')[0]}\n")

    def err(self, exception):
        print(f"\033[91mERROR: {exception}\033[0m")
        traceback.print_exc()
        with open(self.log_file, 'a') as f:
            f.write(f"\n\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
                    f"-------------------------------------------------------------------------------------------\n"
                    f"ERROR: {exception}\n\n")
            traceback.print_exception(type(exception), exception, exception.__traceback__, file=f)
            f.write(f"\n-------------------------------------------------------------------------------------------\n"
                    f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\n")
