from ner.exception import CustomeException
from ner.logger import logger
from ner.config.configuration import Configuration
from ner.entity.config_entity import ModelTrainerConfig
from ner.components.model_architecture import XLMRobertaForTokenClassification
from transformers import DataCollatorForTokenClassification
import sys, os
from transformers import Trainer, TrainingArguments
from typing import Dict
import numpy as np
from seqeval.metrics import f1_score

class TrainTokenClassifier:

    def __init__(self, model_trainer_config: ModelTrainerConfig, processed_data: Dict):
        self.model_trainer_config = model_trainer_config
        self.preprocessed_data = processed_data

    def create_training_args(self):
        try:
            logger.info("Create training arguments...")
            loggigng_steps = len(self.preprocessed_data['train'].select(range(60))) // self.model_trainer_config.batch_size

            training_args = TrainingArguments(
                output_dir=self.model_trainer_config.output_dir,
                log_level="error",
                num_train_epochs=self.model_trainer_config.epochs,
                per_device_train_batch_size=self.model_trainer_config.batch_size,
                per_device_eval_batch_size=self.model_trainer_config.batch_size,
                save_steps=self.model_trainer_config.save_steps,
                weight_decay=0.01,
                disable_tqdm=False,
                save_total_limit=1,
                logging_steps=loggigng_steps,
                push_to_hub=True,
                no_cuda=True,  # Set to True for CPU training, False for GPU
            )

            return training_args
        except Exception as e:
            raise CustomeException(e, sys)

    def model_init(self):
        try:
            logger.info("Initializing model...")
            model = XLMRobertaForTokenClassification.from_pretrained(
                self.model_trainer_config.model_name,
                config=self.model_trainer_config.xlmr_config
            )
            return model
        except Exception as e:
            raise CustomeException(e, sys)
        
    def data_collator(self):
        try:
            logger.info("Creating data collator...")
            data_collator = DataCollatorForTokenClassification(tokenizer=self.model_trainer_config.tokenizer)
            return data_collator
        except Exception as e:
            raise CustomeException(e, sys)
        
    def align_prediction(self, predictions, labels):
        try:
            preds = np.argmax(prediction, axis=2)
            batch_size, seq_len = preds.shape
            labels_list, preds_list = [], []

            for batch_idx in range(batch_size):
                example_labels, example_preds = [], []
                for seq_idx in range(seq_len):
                    # Ignore labels IDs = -100
                    if labels[batch_idx][seq_idx] != -100:
                        example_labels.append(self.model_trainer_config.index2tag[labels[batch_idx][seq_idx]])
                        example_preds.append(self.model_trainer_config.index2tag[preds[batch_idx][seq_idx]])
                
                labels_list.append(example_labels)
                preds_list.append(example_preds)
            
            return preds_list, labels_list 
        except Exception as e:
            raise CustomeException(e, sys)
        
    def compute_metrics(self, eval_pred):
        try:
            y_pred, y_true = self.align_prediction(eval_pred.predictions, eval_pred.labels)
            return {"f1": f1_score(y_true, y_pred)} 
        except Exception as e:
            raise CustomeException(e, sys)
        
    def train(self):
        try:
            logger.info("Model Training Started")
            trainer = Trainer(
                model_init = self.model_init,
                args = self.create_training_args(),
                data_collator = self.data_collator(),
                compute_metrics = self.compute_metrics,
                train_dataset = self.preprocessed_data['train'].select(range(60)),  # For testing purposes, use a subset
                eval_dataset = self.preprocessed_data['validation'].select(range(10)),  # For testing purposes, use a subset 
                processing_class = self.model_trainer_config.tokenizer
            )
            logger.info("Training is Running...")
            result = trainer.train()
            logger.info(f"Result of Training is {result}")
            trainer.save_model(self.model_trainer_config.output_dir)
            logger.info(f"Model is saved at {self.model_trainer_config.output_dir}")
        except Exception as e:
            raise CustomeException(e, sys)