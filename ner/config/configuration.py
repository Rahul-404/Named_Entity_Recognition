import os, sys
from ner.utils import read_config
from ner.exception import CustomeException
from ner.entity.config_entity import DataIngestionConfig, DataValidtaionConfig, DataPreprocessingConfig
from ner.constants import *
from transformers import AutoConfig, AutoTokenizer
from ner.logger import logger

class Configuration:

    def __init__(self):
        try:
            logger.info("Reading the configuration file...")
            self.config = read_config(file_name=CONFIG_FILE_NAME)
        except Exception as e:
            raise CustomeException(e, sys)
        
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        try:
            dataset_name = self.config[DATA_INGESTION_KEY][DATASET_NAME]
            subset_name = self.config[DATA_INGESTION_KEY][SUBSET_NAME]
            data_store = os.path.join(os.getcwd(), self.config[PATH_KEY][ARTIFACTS_KEY],
                                      self.config[PATH_KEY][DATA_STORE_KEY])
            
            data_ingestion_config = DataIngestionConfig(
                dataset_name=dataset_name,
                subset_name=subset_name,
                data_store=data_store
            )

            return data_ingestion_config
        except Exception as e:
            raise CustomeException(e, sys)
        
    def get_data_validation_config(self) -> DataValidtaionConfig:
        try:
            split = self.config[DATA_VALIDATION_KEY][DATA_SPLIT]
            columns = self.config[DATA_VALIDATION_KEY][COLUMNS_CHECK]

            null_value_check = self.config[DATA_VALIDATION_KEY][NULL_CHECK]
            type_check = self.config[DATA_VALIDATION_KEY][TYPE_CHECK]

            data_validation_config = DataValidtaionConfig(
                dataset=None,
                data_split=split,
                columns_check=columns,
                type_check=type_check,
                null_check=null_value_check
            )

            return data_validation_config
        except Exception as e:
            raise CustomeException(e, sys)
        
    def get_data_preprocessing_config(self):
        try:
            model_name = self.config[BASE_MODEL_CONFIG][BASE_MODEL_NAME]
            tags = self.config[DATA_PREPROCESSING_KEY][NER_TAGS_KEY]

            index2tag = {index: tag for index, tag in enumerate(tags)}
            tag2index = {tag: index for index, tag in enumerate(tags)}

            tokenizer = AutoTokenizer.from_pretrained(self.config[BASE_MODEL_CONFIG][BASE_MODEL_NAME])

            data_preprocessing_config = DataPreprocessingConfig(
                model_name=model_name,
                tags=tags,
                index2tag=index2tag,
                tag2index=tag2index,
                tokenizer=tokenizer
            )

            return data_preprocessing_config
        except Exception as e:
            raise CustomeException(e, sys) 