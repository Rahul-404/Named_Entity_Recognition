from collections import namedtuple

DataIngestionConfig = namedtuple('DataIngestionConfig', ["dataset_name", "subset_name", "data_store"])

DataValidtaionConfig = namedtuple('DataValidationConfig', ["dataset", "data_split", "columns_check", "type_check", "null_check"])

DataPreprocessingConfig = namedtuple('DataPreprocessingConfig', ["model_name", "tags", "index2tag",
                                                                 "tag2index", "tokenizer"])