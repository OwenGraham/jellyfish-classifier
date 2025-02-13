import sys

class Colours:
    OKGREEN = '\033[92m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'

def progress_bar(iteration, total, prefix='', suffix='', length=30, fill='█'):
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    completed_bar = fill * filled_length
    pending_bar = '|' +'-' * (length - filled_length - 1) if iteration < total - 1 else ''
    sys.stdout.write(f'\r{prefix} |{Colours.OKGREEN}{completed_bar}{Colours.ENDC}{Colours.FAIL}{pending_bar}{Colours.ENDC}| {percent}% {suffix}')
    sys.stdout.flush() if iteration < total else print()

# Function to log progress
def log(epoch, epochs, step, total_loss, log_step, data_size):
    avg_loss = total_loss / log_step
    sys.stderr.write(f"\rEpoch: {epoch+1}/{epochs}, Step: {step}/{data_size}, Loss: {avg_loss}")
    sys.stderr.flush()