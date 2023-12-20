# main aim is to read the dataset from a specific data source
# read data and split into train/ test set

import os
import sys
from src.exception import CustomExcpetion
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass
#used to automatically make class variable

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifact', 'train.csv')
    test_data_path: str = os.path.join('artifact', 'test.csv')
    raw_data_path: str = os.path.join('artifact', 'data.csv')

class DataIngestion:
    #if other fucntions inside clss, use constructor method
    def __init__(self):
        self.ingestion_config=DataIngestionConfig() # the three paths will automatically get saved in this variable (as sub-objects)

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion component")
        try:
            df = pd.read_csv('notebook/data/stud.csv')
            logging.info("Read the dataset as pandas data frame")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True) # need to combine path directory
            
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split initiated")

            train_set,test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomExcpetion(e, sys)
        
if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()
    # artifact folder created and logs updated