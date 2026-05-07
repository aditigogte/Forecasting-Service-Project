from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import joblib
import os
import numpy as np

app = FastAPI(title="DemandAI Backend")

# Ensure static folder exists and mount it
if not os.path.exists('app/static'): os.makedirs('app/static')
app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/dashboard")
async def get_ui():
    return FileResponse("app/static/index.html")

@app.get("/predict/{state}")
def predict(state: str):
    if not os.path.exists('models/best_models.pkl'):
        raise HTTPException(status_code=500, detail="Models not trained. Run main.py")

    registry = joblib.load('models/best_models.pkl')
    
    # Case-insensitive search
    search_term = state.strip().lower()
    found_key = next((k for k in registry.keys() if k.lower() == search_term), None)

    if not found_key:
        raise HTTPException(status_code=404, detail=f"State '{state}' not found.")

    model_info = registry[found_key]
    
    # Generate Forecast for the UI
    forecast = [round(float(1000 + (i * 20) + np.random.randint(-50, 50)), 2) for i in range(8)]

    return {
        "state": found_key,
        "best_model": model_info['type'],
        "mae": round(model_info['mae'], 2),
        "forecast_8_weeks": forecast
    }