from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

app = FastAPI()

# Global variables
DATA_PATH = "synthetic_machine_data.csv"
MODEL_PATH = "model.pkl"

# Upload Endpoint
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)
    if not all(col in df.columns for col in ["Machine_ID", "Temperature", "Run_Time", "Downtime_Flag"]):
        return {"error": "Invalid dataset format. Required columns: Machine_ID, Temperature, Run_Time, Downtime_Flag"}
    df.to_csv(DATA_PATH, index=False)
    return {"message": "File uploaded successfully"}

# Train Endpoint
@app.post("/train")
async def train_model():
    try:
        df = pd.read_csv(DATA_PATH)
        X = df[["Temperature", "Run_Time"]]
        y = df["Downtime_Flag"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LogisticRegression()
        model.fit(X_train, y_train)

        # Save the model
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)

        # Evaluate the model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        return {"accuracy": accuracy, "f1_score": f1}
    except Exception as e:
        return {"error": str(e)}

# Predict Endpoint
class PredictionInput(BaseModel):
    Temperature: float
    Run_Time: float

@app.post("/predict")
async def predict(data: PredictionInput):
    try:
        # Load the saved model and scaler
        with open(MODEL_PATH, "rb") as f:
            saved_objects = pickle.load(f)
            model = saved_objects["model"]
            scaler = saved_objects["scaler"]

        # Prepare the input data
        input_data = [[data.Temperature, data.Run_Time]]
        input_data_scaled = scaler.transform(input_data)

        # Make predictions
        prediction = model.predict(input_data_scaled)
        confidence = max(model.predict_proba(input_data_scaled)[0])

        # Convert prediction to a readable format
        result = "Yes" if prediction[0] == 1 else "No"

        return {"Downtime": result, "Confidence": round(confidence, 2)}
    except Exception as e:
        return {"error": str(e)}

