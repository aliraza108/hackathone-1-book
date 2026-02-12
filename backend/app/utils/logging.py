"""
Logging utilities for the application
"""
import logging
from datetime import datetime
import os

def setup_logger(name, log_file=None, level=logging.INFO):
    """Function to setup as many loggers as you want"""
    
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s %(message)s')

    if log_file:
        handler = logging.FileHandler(log_file)        
    else:
        handler = logging.StreamHandler()
        
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


def get_app_logger():
    """Get the main application logger"""
    log_level = os.getenv("LOG_LEVEL", "INFO")
    return setup_logger(__name__, level=getattr(logging, log_level))


def log_api_call(endpoint: str, method: str, user_id: str = None, duration: float = None):
    """Log API calls for monitoring"""
    logger = get_app_logger()
    logger.info(f"API CALL: {method} {endpoint} | User: {user_id} | Duration: {duration}s")


def log_error(error: Exception, context: str = ""):
    """Log errors with context"""
    logger = get_app_logger()
    logger.error(f"ERROR in {context}: {str(error)}", exc_info=True)