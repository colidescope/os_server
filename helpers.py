from datetime import datetime

verbose_log = 2
verbose_print = 0

def log(message, level=2):
    now = datetime.now()
    current_time = now.strftime("%m/%d/%Y %H:%M:%S")

    if verbose_log >= level:
        with open("log.txt", 'a') as f:
            f.write("{} | {}".format(current_time, message) + "\n")
    if verbose_print >= level:
        print("LOG:", current_time, message)
