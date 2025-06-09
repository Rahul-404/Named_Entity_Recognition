import os
import sys

import logging
from datetime import datetime

LOG_DIR = "logs"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

def get_log_filename():
    file_name = f"log_{TIMESTAMP}.log"
    return file_name

LOG_FILENAME = get_log_filename()

os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILENAME)

logging.basicConfig(
    level=logging.INFO,
    filemode="w",
    format="[%(asctime)s] \t%(levelname)s \t%(lineno)d  \t%(filename)s \t%(funcName)s() \t%(message)s",
    filename=LOG_FILE_PATH
)

logger = logging.getLogger("ner")
