"""
Common utility functions
"""

import numpy as np
import os
from datetime import datetime
import logging

def get_next_run(output_path):
    idx = 0
    path = os.path.join(output_path, "run_{:03d}".format(idx))
    while os.path.exists(path):
        idx += 1
        path = os.path.join(output_path, "run_{:03d}".format(idx))
    return path

def get_logger(outdir):
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.FileHandler(os.path.join(outdir, datetime.now().strftime('log_%d_%m_%Y'))))
    logger.addHandler(logging.StreamHandler())
    return logger



