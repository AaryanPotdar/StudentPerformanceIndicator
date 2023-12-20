import os
import sys

import numpy as np
import pandas as pd
import dill # library we need to create pickle file

from src.exception import CustomExcpetion
from src.logger import logging

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj: # write in binary mode
            dill.dump(obj, file_obj)    # we are saving the pickle name in hard disk

    except Exception as e:
        raise CustomExcpetion(e, sys)