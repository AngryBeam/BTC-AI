from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Optional
import logging
from fastapi import APIRouter, Depends, Request
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
import numpy as np
import plotly.graph_objects as go
import json
from utils import analyze_progress_report
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
import os
# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    filename='logs/api.log')
logger = logging.getLogger(__name__)

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load the weights
MODEL_PATH = "models\\checkpoint-episode-184_a0dcf3e0-c127-452d-a89b-1bd431e139ee.weights.h5"

class TradeData(BaseModel):
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    indicators: Optional[dict] = None

class PredictionRequest(BaseModel):
    UnixTime: int
    Open: float
    High: float
    Low: float
    Close: float
    Volume: float
    VolumeBasePrice: float
    RSI: float
    StochasticK: float
    StochasticD: float
    MACDLine: float
    SignalLine: float
    MACDHistogram: float
    SMA7: float
    SMA25: float
    SMA99: float
    SMA7_25Gap: float
    SMA7_99Gap: float
    SMA25_99Gap: float

def get_action(data):
    # Load the model if it's not already loaded
    if not hasattr(get_action, "model"):
        custom_objects = {"mse": MeanSquaredError()}
        get_action.model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)
    
    
    # Reshape the data to match the model's input shape
    input_data = np.array(data).reshape(1, 1, -1)  # Reshape to (batch_size, timesteps, features)
    
    # Get model output
    model_output = get_action.model.predict(input_data)
    
    # Assuming the model output is a single value representing the action index
    action_index = np.argmax(model_output[0])
    
    # Define actions based on the index
    actions = [
        "HOLD",
        "OPEN_LONG",
        "OPEN_SHORT",
        "CLOSE_LONG_10%", "CLOSE_LONG_20%", "CLOSE_LONG_30%", "CLOSE_LONG_40%", "CLOSE_LONG_50%",
        "CLOSE_LONG_60%", "CLOSE_LONG_70%", "CLOSE_LONG_80%", "CLOSE_LONG_90%", "CLOSE_LONG_100%",
        "CLOSE_SHORT_10%", "CLOSE_SHORT_20%", "CLOSE_SHORT_30%", "CLOSE_SHORT_40%", "CLOSE_SHORT_50%",
        "CLOSE_SHORT_60%", "CLOSE_SHORT_70%", "CLOSE_SHORT_80%", "CLOSE_SHORT_90%", "CLOSE_SHORT_100%"
    ]
    
    return actions[action_index]



@app.post("/predict")
async def predict(request: PredictionRequest):
    try:
        # Log the received data
        logger.info(f"Received data request: {request.dict()}")
        # ตรวจสอบว่าไฟล์โมเดลมีอยู่หรือไม่
        if not os.path.exists(MODEL_PATH):
            raise HTTPException(status_code=404, detail="Model file not found")

        # Transform and reorder the data
        transformed_data = [
            request.UnixTime,  # Unix
            request.Open,      # Open
            request.High,      # High
            request.Low,       # Low
            request.Close,     # Close
            request.Volume,    # Volume BTC
            request.VolumeBasePrice,  # Volume USDT
            request.RSI,       # RSI
            request.StochasticK,  # Stoch_RSI (using StochasticK as an approximation)
            request.MACDLine,  # MACD
            request.SignalLine,  # MACD_Signal
            request.MACDHistogram,  # MACD_Histogram
            request.SMA7,      # SMA_7
            request.SMA25,     # SMA_25
            request.SMA99,     # SMA_99
            request.SMA7_25Gap,  # SMA_7_25_GAP
            request.SMA7_99Gap,  # SMA_7_99_GAP
            request.SMA25_99Gap  # SMA_25_99_GAP
        ]
        
        # Get action
        action = get_action(transformed_data)
        logger.info(f"Return Action: {action}")
        return {"action": action}
    
    except FileNotFoundError:
        logger.error(f"Model file not found: {MODEL_PATH}")
        raise HTTPException(status_code=404, detail="Model file not found")
    except Exception as e:
        logger.error(f"Error processing prediction request: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/training-progress", response_class=HTMLResponse)
async def get_training_progress(request: Request):
    try:
        # Analyze the progress report
        result = analyze_progress_report()
        analysis = result["analysis"]
        summary = result["summary"]
        
        # Create a graph comparing test_reward with best_reward
        episodes = [entry['episode'] for entry in analysis]
        test_rewards = [entry['test_reward'] for entry in analysis]
        best_rewards = [entry['best_reward'] for entry in analysis]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=episodes, y=test_rewards, mode='lines', name='Test Reward'))
        fig.add_trace(go.Scatter(x=episodes, y=best_rewards, mode='lines', name='Best Reward'))
        
        fig.update_layout(
            title='Training Progress: Test Reward vs Best Reward',
            xaxis_title='Episode',
            yaxis_title='Reward',
            legend_title='Metrics'
        )
        
        # Convert the plot to HTML
        plot_html = fig.to_html(full_html=False)
        
        # Render the template with the plot and summary
        return templates.TemplateResponse("training_progress.html", {
            "request": request,
            "plot": plot_html,
            "summary": summary
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)