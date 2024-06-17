from fastapi import FastAPI, HTTPException
from pydantic_models import PredictionInput, BatchPredictionInput
from model_predict import load_model, predict, predict_batch

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Welcome to the Cayena data prediction API!"}

model = load_model("models/catboost_model_v0")

@app.post("/predict")
async def predict_route(input_data: PredictionInput):
    data_dict = input_data.dict()
    try:
        prediction, probability = predict(model, data_dict)
        return {"prediction": prediction, "probability": probability}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_batch")
async def predict_batch_route(input_data: BatchPredictionInput):
    data_list = [data.dict() for data in input_data.inputs]
    try:
        results = predict_batch(model, data_list)
        return {"predictions": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))