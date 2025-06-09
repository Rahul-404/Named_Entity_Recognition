import pandas as pd
import sys
from ner.exception import CustomeException
from ner.logger import logger
from ner.entity.config_entity import DataValidtaionConfig
from typing import Dict, List

class DataValidation:

    def __init__(self, data_validation_config: DataValidtaionConfig, data: Dict):
        logger.info("Data Validation component started...")
        self.data_validation_config = data_validation_config
        self.data = data

    def check_columns_names(self) -> bool:
        try:
            logger.info("Checking Columns of all the splits...")
            column_check_result = list()

            for split_name in self.data_validation_config.data_split:
                column_check_result.append(
                    sum(pd.DataFrame(self.data[split_name]).columns == self.data_validation_config.columns_check)
                )
            logger.info(f"Check Results {column_check_result}")

            if sum(column_check_result) == len(self.data_validation_config.data_split) * \
                len(self.data_validation_config.columns_check):
                return True
            else:
                return False
        except Exception as e:
            raise CustomeException(e, sys)
        
    def type_check(self) -> bool:
        try:
            logger.info("Checking datatypes of all the columns")
            type_check_result = list()

            for split_name in self.data_validation_config.data_split:
                df = pd.DataFrame(self.data[split_name])

                type_check_result.append(
                    sum(df.dtypes == self.data_validation_config.type_check)
                )
            logger.info(f"Check Results {type_check_result}")

            if sum(type_check_result) == len(self.data_validation_config.data_split) * \
                len(self.data_validation_config.type_check):
                return True
            else:
                return False

        except Exception as e:
            raise CustomeException(e, sys)
        
    def null_check(self) -> bool:
        try:
            logger.info("Checking nulls of all columns")
            null_check_result = list()

            for split_name in self.data_validation_config.data_split:
                df = pd.DataFrame(self.data[split_name])
                null_check_result.append(
                    sum(df.isnull().sum() == 0)
                )
            logger.info(f"Check Results {null_check_result}")

            if sum(null_check_result) == len(self.data_validation_config.data_split) * \
                len(self.data_validation_config.columns_check):
                return True
            else:
                return False
        except Exception as e:
            raise CustomeException(e, sys)
        
    def run_all_checks(self) -> List[bool]:
        try:
            logger.info("Running all data validation checks...")
            results = [
                self.check_columns_names(),
                self.type_check(),
                self.null_check()
            ]
            logger.info(f"Data Validation Results: {results}")
            return results
        except Exception as e:
            raise CustomeException(e, sys)
        
    