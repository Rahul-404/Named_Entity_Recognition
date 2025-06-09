from ner.exception import CustomeException
from ner.logger import logger
from ner.config.configuration import Configuration
from ner.entity.config_entity import DataPreprocessingConfig
import sys
from typing import Dict, Any

class DataPreprocessing:

    def __init__(self, data_preprocessing_config: DataPreprocessingConfig, data: Any):
        logger.info("Data preprocessing component started...")
        self.data_preprocessing_config = data_preprocessing_config
        self.data = data

    def create_tag_name(self, batch):
        return {"ner_tags_str": [self.data_preprocessing_config.index2tag[idx] for idx in batch["ner_tags"]]}
    
    def tokenize_and_align_labels(examples):
        try:
            logger.info("Tokenizing and aligning labels...")
            tokenized_inputs = xlmr_tokenizer(examples["tokens"], truncation=True, 
                                            is_split_into_words=True)
            labels = []

            for idx, label in enumerate(examples["ner_tags"]):
                word_ids = tokenized_inputs.word_ids(batch_index=idx)
                previous_word_idx = None
                label_ids = []
                for word_idx in word_ids:
                    if word_idx is None or word_idx == previous_word_idx:
                        label_ids.append(-100)
                    else:
                        label_ids.append(label[word_idx])
                    previous_word_idx = word_idx
                labels.append(label_ids)

            tokenized_inputs["labels"] = labels

            return tokenized_inputs
        except Exception as e:
            raise CustomeException(e, sys)
        
    def encode_en_dataset(self, corpus):
        try:
            return corpus.map(self.tokenize_and_align_labels, batched=True, 
                            remove_columns=['langs', 'ner_tags', 'tokens'])
        except Exception as e:   
            raise CustomeException(e, sys)    
        
    def preprare_data_for_finetuning(self) -> Dict:
        try:
            self.data = self.data.mao(self.create_tag_name)
            panx_en_encoded = self.encode_en_dataset(self.data)
            return panx_en_encoded
        except Exception as e:
            raise CustomeException(e, sys)  