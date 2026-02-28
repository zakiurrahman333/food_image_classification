from fastapi import FastAPI, UploadFile, File
import uvicorn
import torch
import torchvision
from model import create_effnetb2
from utils import predict_image
import shutil
import os

app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class names
class_names = ["pizza", "steak", "sushi"]

# Load model
model = create_effnetb2(len(class_names))
model.load_state_dict(torch.load("effnetb2_pizza_model.pth", map_location=device))
model.eval()

# EfficientNet transforms
weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
transform = weights.transforms()

@app.get("/")
def home():
    return {"message": "Pizza-Steak-Sushi Classifier API is running 🚀"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    file_path = f"temp_{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    prediction = predict_image(model, file_path, transform, class_names)

    os.remove(file_path)

    return {"prediction": prediction}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)    