from ner.exception import CustomeException
from ner.logger import logger
from ner.config.configuration import Configuration
from ner.entity.config_entity import DataIngestionConfig
import sys
from datasets import load_dataset

class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        logger.info("Data ingestion component started...")
        self.data_ingestion_config = data_ingestion_config

    def get_data(self):
        try:
            logger.info("Loading data from HuggingFace...")
            pan_en_data = load_dataset(self.data_ingestion_config.dataset_name,
                                       name = self.data_ingestion_config.subset_name)
            logger.info(f"Data Info : {pan_en_data}")

            return pan_en_data
        except Exception as e:
            raise CustomeException(e, sys)