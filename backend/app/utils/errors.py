"""
Utility functions for logging and error handling
"""
import logging
from datetime import datetime
import os

# Set up logging
def setup_logging():
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('app.log'),
            logging.StreamHandler()
        ]
    )

def get_logger(name):
    return logging.getLogger(name)


# Error handling
class AppError(Exception):
    """Base application error"""
    def __init__(self, message, status_code=500):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class ValidationError(AppError):
    """Validation error"""
    def __init__(self, message):
        super().__init__(message, 400)


class NotFoundError(AppError):
    """Not found error"""
    def __init__(self, message):
        super().__init__(message, 404)


class InternalError(AppError):
    """Internal server error"""
    def __init__(self, message):
        super().__init__(message, 500)