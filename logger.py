from logger_config import setup_logger

# Global variables
logger = None
run_id = None

def get_logger():
    global logger, run_id
    if logger is None:
        logger, run_id = setup_logger()
    return logger, run_id

# Initialize logger when the module is imported
test_logger, _ = get_logger()