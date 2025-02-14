import logging
import os

def setup_logger(exp_dir):
    """Sets up the logger for training logs."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    log_file = os.path.join(exp_dir, "train.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    
    return logger
