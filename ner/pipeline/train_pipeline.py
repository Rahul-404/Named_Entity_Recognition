from ner.components.data_ingestion import DataIngestion
from ner.components.data_validation import DataValidation
from ner.components.data_preparation import DataPreprocessing
from ner.components.model_trainer import TrainTokenClassifier
from ner.config.configuration import Configuration

from ner.logger import logger
from ner.exception import CustomeException
import sys
from typing import Dict, List

class TrainPipeline:

    def __init__(self, config):
        self.config = config
        
    def run_data_ingestion(self) -> Dict:
        try:
            logger.info("Running DataIngestion in TrainingPipeline")
            data_ingestion_config = self.config.get_data_ingestion_config()
            ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
            data = ingestion.get_data()
            return data
        except Exception as e:
            raise CustomeException(e, sys)
        
    def run_data_validation(self, data: Dict) -> List[bool]:
        try:
            logger.info("Running DataValidation in TrainingPipeline")
            data_validation_config = self.config.get_data_validation_config()
            validation = DataValidation(data_validation_config=data_validation_config, data=data)
            checks = validation.run_all_checks()
            return checks
        except Exception as e:
            raise CustomeException(e, sys)
        
    def run_data_preparation(self, data) -> Dict:
        try:
            logger.info("Running DataPreprocessing in TrainingPipeline")
            data_preprocessing_config = self.config.get_data_preprocessing_config()
            preprocessor = DataPreprocessing(data_preprocessing_config=data_preprocessing_config, data=data)
            processed_data = preprocessor.prepare_data_for_fine_tuning()
            return processed_data
        except Exception as e:
            raise CustomeException(e, sys)
        
    def run_model_training(self, processed_data):
        try:
            logger.info("Running Model Training in TrainingPipeline")
            model_trainer_config = self.config.get_model_trainer_config()
            classifier = TrainTokenClassifier(model_trainer_config=model_trainer_config, processed_data=processed_data)
            classifier.train()
            logger.info("Model training completed successfully.")
        except Exception as e:
            raise CustomeException(e, sys)
        
    def validate_check(check):
        for i in check:
            if i == False:
                return False
            else:
                return True

    def run_pipeline(self):
        data = self.run_data_ingestion()
        checks = self.run_data_validation(data)
        if self.validate_check(checks):
            processed_data = self.run_data_preparation(data)
            logger.info(f"Preprocessed data : {processed_data}")
            self.run_model_training(processed_data=processed_data)
        else:
            logger.error(f"Checks Failed")