from ner.components.data_ingestion import DataIngestion
from ner.components.data_validation import DataValidation
from ner.components.data_preparation import DataPreprocessing
from ner.components.model_trainer import TrainTokenClassifier
from ner.config.configuration import Configuration
from ner.pipeline.predict_pipeline import PredictionPipeline
from datetime import datetime

project_config = Configuration()
# ingestion = DataIngestion(project_config.get_data_ingestion_config())
# en_data = ingestion.get_data()

# validate = DataValidation(data_validation_config=project_config.get_data_validation_config(), 
#                             data=en_data)

# check = validate.run_all_checks()

# print("Data Validation Check Results:", check)

# def validate_check():
#     for i in check:
#         if i == False:
#             return False
#         else:
#             return True

# if validate_check():
#     preprocessed_data = DataPreprocessing(data_preprocessing_config=project_config.get_data_preprocessing_config(), 
#                                       data=en_data)
#     p_data = preprocessed_data.prepare_data_for_fine_tuning()

#     classifier = TrainTokenClassifier(model_trainer_config=project_config.get_model_trainer_config(), 
#                                       processed_data=p_data)
#     classifier.train()

start = datetime.now()
text = "Hello, my name is John Doe and I live in New York City."
pipeline = PredictionPipeline(config=project_config)
print(f"\nInput: {text}\n")
print(f"Output: {pipeline.run_pipeline(text)}\n")
end = datetime.now() - start
print(f"Time taken for prediction: {end.total_seconds()} seconds\n")