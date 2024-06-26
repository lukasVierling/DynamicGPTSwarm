#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
from pathlib import Path
from loguru import logger
from swarm.utils.const import GPTSWARM_ROOT

def configure_logging(print_level: str = "INFO", logfile_level: str = "DEBUG") -> None:
    """
    Configure the logging settings for the application.

    Args:
        print_level (str): The logging level for console output.
        logfile_level (str): The logging level for file output.
    """
    logger.remove()
    logger.add(sys.stderr, level=print_level)
    logger.add(GPTSWARM_ROOT / 'logs/log.txt', level=logfile_level)

def initialize_log_file(mode: str, time_stamp: str) -> Path:
    """
    Initialize the log file with a start message and return its path.

    Args:
        mode (str): The mode of operation, used in the file path.
        time_stamp (str): The current timestamp, used in the file path.

    Returns:
        Path: The path to the initialized log file.
    """
    try:
        log_file_path = GPTSWARM_ROOT / f'result/{mode}/logs/log_{time_stamp}.txt'
        os.makedirs(log_file_path.parent, exist_ok=True)
        with open(log_file_path, 'w') as file:
            file.write("============ Start ============\n")
    except OSError as error:
        logger.error(f"Error initializing log file: {error}")
        raise
    return log_file_path

def swarmlog(sender: str, text: str, cost: float, result_file: Path = None, solution: list = []) -> None:
    """
    Custom log function for swarm operations.

    Args:
        sender (str): The name of the sender.
        text (str): The text message to log.
        cost (float): The cost associated with the operation.
        result_file (Path, optional): Path to the result file. Default is None.
        solution (list, optional): Solution data to be logged. Default is an empty list.
    """
    formatted_message = f"{sender} | 💵Total Cost: {cost:.5f}\n{text}"
    logger.info(formatted_message)

# It's generally a good practice to have a main function to control the flow of your script
def main():
    configure_logging()

if __name__ == "__main__":
    main()
