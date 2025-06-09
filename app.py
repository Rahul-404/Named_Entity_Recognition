from fastapi.responses import JSONResponse
from fastapi import FastAPI, Request
import uvicorn

from ner.constants import HOST, PORT
from ner.pipeline.train_pipeline import TrainPipeline
from ner.pipeline.predict_pipeline import PredictionPipeline
from ner.config.configuration import Configuration

app = FastAPI()

@app.post('/train')
def train():
    try:
        pipeline = TrainPipeline(Configuration())
        pipeline.run_pipeline()
        return JSONResponse(status_code=200, content={"message": "Training completed successfully."})
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": "Error during training."})
    
@app.post('/predict')
def predict(data):
    try:
        pipeline = PredictionPipeline(config=Configuration())
        response = pipeline.run_pipeline(data)
        return JSONResponse(status_code=200, content=response)
    except Exception as e:
        print(e)
        return JSONResponse(status_code=500, content={"message": "Error during prediction."})
    
if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)