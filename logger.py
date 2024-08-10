import logging
import os
from datetime import datetime

def setup_logger(log_folder='logs'):
    # Create logs folder if it doesn't exist
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    # Create a unique log file name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_folder, f'btc_ai_{timestamp}.log')

    # Set up logging configuration
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # This will also print to console
        ]
    )

    return logging.getLogger()