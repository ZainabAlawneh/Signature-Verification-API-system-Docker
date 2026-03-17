from fastapi import FastAPI, UploadFile, File
import torch
from PIL import Image
import io
from app.inference import load_model , predict
app = FastAPI()
model = None


@app.on_event("startup")
def first_load():
    global model
    model = load_model()





@app.post("/prediction")
async def prediction(image1: UploadFile = File(...),
                  image2: UploadFile=File(...)):


    content1 = await image1.read()
    content2 = await image2.read()
    result = predict(content1, content2, model)

    return {"Prediction ":result}



    


