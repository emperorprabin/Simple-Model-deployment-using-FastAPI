import io
import joblib
import pandas as pd
from fastapi import FastAPI,HTTPException, UploadFile,File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel,Field

app=FastAPI()

model=joblib.load("house_price_model.joblib")
features=joblib.load("house_features.joblib")

#input schema
class House_features(BaseModel):
    MedInc:float=Field(gt=0,description="median Income of Neighbourhod")
    House_age:float=Field(gt=0,description="Age of the hpouse")
    AveRooms:float=Field(gt=0,description="average number of rooms")
    AveBedrms:float=Field(gt=0,description="Average number of bedrooms")
    Pop:float=Field(gt=0,description="Total population in the area")
    AveOcc:float=Field(gt=0,description="average number of people living in a house")
    Latitude:float=Field(description="Latitude coordinates of the house")
    Longitude:float=Field(description="Longitude of the house")

#home
@app.get("/")
def home():
    return{
        "message":"California house prediction API",
        "status":"Model Running",
        "endpoint":"Send POST  request to /predict"
    }

@app.get("/health")
def health():
    return{"status":"Health check Running",
           "model":"RandomForestregressor",
           "features":features,
           "averageError":"$39,000"}

@app.post("/predict")
def predict(house:House_features):
    try:
        input=pd.DataFrame([{
            "MedInc":house.MedInc,
            "HouseAge":house.House_age,
            "AveRooms":house.AveRooms,
            "AveBedrms":house.AveBedrms,
            "Population":house.Pop,
            "AveOccup":house.AveOcc,
            "Latitude":house.Latitude,
            "Longitude":house.Longitude
        }])
        predicted = model.predict(input)[0]
        price_USD= predicted*100000

        return{
            "predicted Price":f"${price_USD:.0f}",
            "predicted Price Short":f"{predicted:,.2f} hunderd thousand dollars",
            "Confidence_range":f"${price_USD-39000:,.0f} to ${price_USD+39000:.0f}"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed {e}"
        )

@app.post("/predict-file")

async def predict_file(file: UploadFile= File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a csv file only"
        )
    contents=await file.read()

    df= pd.read_csv(io.BytesIO(contents))

    required_columns=[
        "MedInc","HouseAge","AveRooms","AveBedrms","Population","AveOccup","Latitude","Longitude"
    ]
    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise HTTPException(
            status_code=400,
            detail=f"These columns are missing from your file {missing_columns}"
        )

    if len(df)==0:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file has no data"
        )
    try:
        predictions=model.predict(df[required_columns])
        df["Predicted columns"]=predictions*100000

        df["Predicted columns"]=df["Predicted columns"].apply(lambda x:f"${x:.0f}")

        output=df.to_csv(index=False)

        return StreamingResponse(
            io.StringIO(output),
            media_type="text/csv",
            headers={
                "Content-disposition":"Attachment; filename=Prediction.csv"
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed due to {str(e)}"
        )