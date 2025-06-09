from ner.components.model_architecture import XLMRobertaForTokenClassification
from ner.config.configuration import Configuration
from ner.exception import CustomeException
import torch
import os
import numpy as np
import sys

# class PredictionPipeline:
#     def __init__(self, config: Configuration):
#         self.prediction_pipeline_config = config.get_model_predict_pipeline_config()
#         self.tokenizer = self.prediction_pipeline_config.tokenizer
#         self.tags = self.prediction_pipeline_config.tags

#         if len(os.listdir(self.prediction_pipeline_config.output_dir)) == 0:
#             raise LookupError("Model directory is empty. Please train the model first.")
        
#         self.model = XLMRobertaForTokenClassification.from_pretrained(pretrained_model_name_or_path=self.prediction_pipeline_config.output_dir)


    # def predict(self, text):
    #     try:
    #         tokens = self.tokenizer(text).tokens()
    #         input_ids = self.tokenizer(text, return_tensors='pt').input_ids.to('cpu')
    #         output = self.model(input_ids)[0]
    #         predictions = torch.argmax(output, dim=2)
    #         print(f"predictions: {predictions}")
    #         preds = [self.tags[p] for p in predictions[0].cpu().numpy()]
    #         # filtered_tags = []
    #         filtered_preds = []
    #         for token, pred in zip(tokens, preds):
    #             if "_" in token[0]:
    #                 # filtered_tags.append(token)
    #                 filtered_preds.append(pred)
    #         return filtered_preds
    #     except Exception as e:
    #         raise CustomeException(e, sys)
        
    # def run_pipeline(self, data):
    #     prediction = self.predict(data)
    #     response = {
    #         "input_data": data.split(),
    #         "tags": prediction
    #     }
    #     return response

class PredictionPipeline:
    def __init__(self, config: Configuration):
        self.prediction_pipeline_config = config.get_model_predict_pipeline_config()
        self.tokenizer = self.prediction_pipeline_config.tokenizer
        self.tags = self.prediction_pipeline_config.tags

        if len(os.listdir(self.prediction_pipeline_config.output_dir)) == 0:
            raise LookupError("Model directory is empty. Please train the model first.")
        
        self.model = XLMRobertaForTokenClassification.from_pretrained(
            pretrained_model_name_or_path=self.prediction_pipeline_config.output_dir
        )
        self.model.eval()  # put model in eval mode

    def predict(self, text):
        try:
            encoding = self.tokenizer(text, return_tensors='pt', truncation=True)
            input_ids = encoding.input_ids
            attention_mask = encoding.attention_mask

            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)

            tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
            preds = [self.tags[pred.item()] for pred in predictions[0]]

            filtered_preds = []
            filtered_tokens = []

            for token, pred in zip(tokens, preds):
                if not token.startswith("▁"):  # XLM-R uses SentencePiece → '▁' denotes token start
                    continue
                filtered_preds.append(pred)
                filtered_tokens.append(token)

            return list(zip(filtered_tokens, filtered_preds))

        except Exception as e:
            raise CustomeException(e, sys)

    def run_pipeline(self, data):
        predictions = self.predict(data)
        response = {
            "input_data": [token.lstrip("▁") for token, _ in predictions],  # remove leading '▁'
            "tags": [tag for _, tag in predictions]
        }
        return response