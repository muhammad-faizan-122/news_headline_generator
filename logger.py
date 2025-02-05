from loguru import logger


def get_logger(filename="logs/main.log"):
    """logger function"""
    # Remove the default terminal handler
    logger.remove()
    logger.add(
        filename,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {file}:{function}:{line} | {message}",
        rotation="1 day",  # Rotate logs daily
        retention="7 days",  # Keep logs for 7 days
        compression="zip",  # Compress old logs
    )
    return logger
