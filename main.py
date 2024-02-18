import os

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from ml.data import apply_label, process_data
from ml.model import inference, load_model
import pickle

# DO NOT MODIFY
class Data(BaseModel):
    age: int = Field(..., example=37)
    workclass: str = Field(..., example="Private")
    fnlgt: int = Field(..., example=178356)
    education: str = Field(..., example="HS-grad")
    education_num: int = Field(..., example=10, alias="education-num")
    marital_status: str = Field(
        ..., example="Married-civ-spouse", alias="marital-status"
    )
    occupation: str = Field(..., example="Prof-specialty")
    relationship: str = Field(..., example="Husband")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., example=0, alias="capital-gain")
    capital_loss: int = Field(..., example=0, alias="capital-loss")
    hours_per_week: int = Field(..., example=40, alias="hours-per-week")
    native_country: str = Field(..., example="United-States", alias="native-country")

# TODO: enter the path for the saved encoder
project_path = ""
path = os.path.join(project_path, "model", "encoder.pkl")
encoder = load_model(path)

# TODO: enter the path for the saved model
model_path = os.path.join(project_path, "model", "model.pkl")
model = load_model(model_path)

print(path, model_path)

# TODO: create a RESTful API using FastAPI
app = FastAPI()

@app.on_event("startup")
async def startup_event():
    global model, encoder, binarizer
    model = pickle.load(open("./model/model.pkl", "rb"))
    encoder = pickle.load(open("./model/encoder.pkl", "rb"))
    #binarizer = pickle.load(open("./model/lb.pkl", "rb"))


# TODO: create a GET on the root giving a welcome message
@app.get("/")
async def get_root():
    """ Say hello!"""
    return {"message":"Hello and welcome!"}

# TODO: create a POST on a different path that does model inference
@app.post("/data/")
async def post_inference(data: Data):
    global encoder
    # DO NOT MODIFY: turn the Pydantic model into a dict.
    data_dict = data.dict()
    # DO NOT MODIFY: clean up the dict to turn it into a Pandas DataFrame.
    # The data has names with hyphens and Python does not allow those as variable names.
    # Here it uses the functionality of FastAPI/Pydantic/etc to deal with this.
    data = {k.replace("_", "-"): [v] for k, v in data_dict.items()}
    data = pd.DataFrame.from_dict(data)
    print(data)

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country"
    ]


    data_processed, y, encoder, lb =  process_data(
        data,
        categorical_features=cat_features,
        training=False,
        encoder = encoder
        )

    _inference = inference(model, data_processed)
    return {"result": apply_label(_inference)}
