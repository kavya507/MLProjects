import os
import sys
cwd = os.getcwd()
print(cwd)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:

    train_data_path: str = os.path.join('../../artifacts', "train.csv")
    test_data_path: str = os.path.join('../../artifacts', "test.csv")
    raw_data_path: str = os.path.join('../../artifacts', "raw.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        logging.info("Entered dataingetsion model")

    def initiate_data_ingestion(self):
        logging.info("Entered dataingetsion model")
        try:
            df = pd.read_csv('/Users/kavyabaltha/Desktop/MachineLearning/notebook/data/StudentsPerformance.csv')
            logging.info("Read the dataset as dataframe")
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=50)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info(" Dataingetsion is compelete")
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

# test