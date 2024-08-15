import logging
import os
import uuid
from logging.handlers import RotatingFileHandler

class ErrorLogFilter(logging.Filter):
    def filter(self, record):
        return record.levelno >= logging.WARNING

def setup_logger(run_id=None):
    from config import LOG_DIR, get_log_file_path

    if run_id is None:
        run_id = str(uuid.uuid4())

    logger = logging.getLogger(f'BTC_AI_{run_id}')
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Console Handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # Info File Handler
    info_log_file = get_log_file_path(run_id)
    os.makedirs(os.path.dirname(info_log_file), exist_ok=True)
    info_fh = RotatingFileHandler(info_log_file, maxBytes=10*1024*1024, backupCount=5)
    info_fh.setLevel(logging.INFO)
    info_fh.setFormatter(formatter)
    logger.addHandler(info_fh)
    
    # Error File Handler
    error_log_file = os.path.join(LOG_DIR, f'error.log')
    error_fh = RotatingFileHandler(error_log_file, maxBytes=10*1024*1024, backupCount=5)
    error_fh.setLevel(logging.WARNING)
    error_fh.setFormatter(formatter)
    error_fh.addFilter(ErrorLogFilter())
    logger.addHandler(error_fh)
    
    return logger, run_id