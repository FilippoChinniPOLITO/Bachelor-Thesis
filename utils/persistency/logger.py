import os
import traceback
from datetime import datetime


YELLOW = '\033[93m'
RED = '\033[91m'
GREEN = '\033[92m'
END_COLOR = '\033[0m'
DASH_LINE = f"{'-':-<90}"
EQUAL_LINE = f"{'=':=<90}"
WARN_LINE = f"{'!':!<90}"
TEST_LINE = f"{'?':?<90}"
TILD_LINE = f"{'~':~<90}"


class Logger:
    def __init__(self, log_file):
        self.log_file = log_file
        self.start_time = datetime.now()
        self.init_message()
        self.log_count = 0

    def init_message(self):
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'a') as f:
                f.write(f"Log file: {str(self.log_file)}\n"
                        f"Created on: {str(self.start_time).split('.')[0]}\n"
                        f"{DASH_LINE}\n\n")

    def log(self, message):
        print(message)
        with open(self.log_file, 'a') as f:
            f.write(message + '\n')

        self.periodic_log(message)

    def end_log(self):
        with open(self.log_file, 'a') as f:
            f.write(f"\n\n\n{DASH_LINE}\n"
                    f"End of Log File.\n"
                    f"Duration: {str(datetime.now() - self.start_time).split('.')[0]}\n"
                    f"Terminated on: {str(datetime.now()).split('.')[0]}\n")

    def err(self, exception):
        print(f"{RED}ERROR: {exception}{END_COLOR}")
        traceback.print_exc()
        with open(self.log_file, 'a') as f:
            f.write(f"\n\n{WARN_LINE}\n"
                    f"{DASH_LINE}\n"
                    f"ERROR: {exception}\n\n")
            traceback.print_exception(type(exception), exception, exception.__traceback__, file=f)
            f.write(f"\n\n{DASH_LINE}\n"
                    f"{WARN_LINE}\n\n")

    def test(self, test_message):
        print(f"{YELLOW}TEST: {test_message}{END_COLOR}")
        with open(self.log_file, 'a') as f:
            f.write(f"\n\n{TEST_LINE}\n"
                    f"TEST: {test_message}\n"
                    f"{TEST_LINE}\n\n")

    def periodic_log(self, former_mess):
        condition = ('Complete' in former_mess) or ('Pruned' in former_mess)
        if condition:
            self.log_count += 1
        if (self.log_count % 10 == 0) and (self.log_count > 0) and condition:
            print(f"{GREEN}PERIODIC LOG: {str(datetime.now()).split('.')[0]}{END_COLOR}")
            with open(self.log_file, 'a') as f:
                f.write(f"\n\n{TILD_LINE}\n"
                        f"PERIODIC LOG\n\n"
                        f"State:                   Still Running\n"
                        f"Completed/Pruned Trials: {self.log_count}\n"
                        f"Date:                    {str(datetime.now()).split('.')[0]}\n"
                        f"Up-Time:                 {str(datetime.now() - self.start_time).split('.')[0]}\n"
                        f"{TILD_LINE}\n\n")
