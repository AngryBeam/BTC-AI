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

import json
from utils import analyze_progress_report
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
import os
from collections import Counter
from plotly.subplots import make_subplots
import plotly.graph_objects as go

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
        result = analyze_progress_report()
        analysis = result["analysis"]
        summary = result["summary"]
        
        episodes = []
        test_rewards = []
        current_rewards = []
        balances = []
        total_pnls = []
        sharpe_ratios = []
        gammas = []
        epsilons = []
        epsilon_mins = []
        epsilon_decays = []
        action_histories = []
        
        current_retrain = 0
        i=0
        for entry in analysis:
            if entry['episode'] == 0:
                current_retrain += 1
            i+=1
            episodes.append(f"{i}")
            #episodes.append(f"{current_retrain}-{entry['episode']}")
            test_rewards.append(entry['test_reward'])
            current_rewards.append(entry['current_reward'])
            balances.append(entry['test_metrics'].get('balance', 0))
            total_pnls.append(entry['test_metrics'].get('total_pnl', 0))
            sharpe_ratios.append(entry['test_metrics'].get('sharpe_ratio', 0))
            gammas.append(entry['agent'].get('gamma', 0))
            epsilons.append(entry['agent'].get('epsilon', 0))
            epsilon_mins.append(entry['agent'].get('epsilon_min', 0))
            epsilon_decays.append(entry['agent'].get('epsilon_decay', 0))
            action_histories.extend(entry['agent'].get('action_history', []))

        # Create plots
        fig = make_subplots(rows=4, cols=2, subplot_titles=(
            'Test Reward vs Current Reward', 'Total PNL',
            'Balance', 'Sharpe Ratio',
            'Agent Parameters', 'Action Distribution',
            '', ''
        ))

        # Test Reward vs Current Reward
        fig.add_trace(go.Scatter(x=episodes, y=test_rewards, mode='lines', name='Test Reward'), row=1, col=1)
        fig.add_trace(go.Scatter(x=episodes, y=current_rewards, mode='lines', name='Current Reward'), row=1, col=1)

        # Total PNL
        fig.add_trace(go.Scatter(x=episodes, y=total_pnls, mode='lines', name='Total PNL'), row=1, col=2)

        # Balance (separate graph)
        fig.add_trace(go.Scatter(x=episodes, y=balances, mode='lines', name='Balance'), row=2, col=1)

        # Sharpe Ratio
        fig.add_trace(go.Scatter(x=episodes, y=sharpe_ratios, mode='lines', name='Sharpe Ratio'), row=2, col=2)

        # Agent Parameters
        fig.add_trace(go.Scatter(x=episodes, y=gammas, mode='lines', name='Gamma'), row=3, col=1)
        fig.add_trace(go.Scatter(x=episodes, y=epsilons, mode='lines', name='Epsilon'), row=3, col=1)
        fig.add_trace(go.Scatter(x=episodes, y=epsilon_mins, mode='lines', name='Epsilon Min'), row=3, col=1)
        fig.add_trace(go.Scatter(x=episodes, y=epsilon_decays, mode='lines', name='Epsilon Decay'), row=3, col=1)

        # Action Distribution
        action_counts = Counter(action_histories)
        action_map = analysis[-1]['agent'].get('action_map', {})
        actions = [action_map.get(str(i), f"Action {i}") for i in range(len(action_map))]
        action_values = [action_counts[i] for i in range(len(action_map))]
        fig.add_trace(go.Bar(x=actions, y=action_values, name='Action Distribution'), row=3, col=2)

        fig.update_layout(height=2000, width=1200, title_text="Training Progress Analysis")
        
        # Update y-axis labels
        fig.update_yaxes(title_text="Reward", row=1, col=1)
        fig.update_yaxes(title_text="Total PNL", row=1, col=2)
        fig.update_yaxes(title_text="Balance", row=2, col=1)
        fig.update_yaxes(title_text="Sharpe Ratio", row=2, col=2)
        fig.update_yaxes(title_text="Parameter Value", row=3, col=1)
        fig.update_yaxes(title_text="Action Count", row=3, col=2)

        plot_html = fig.to_html(full_html=False)

        # Additional analysis
        avg_test_reward = sum(test_rewards) / len(test_rewards)
        avg_current_reward = sum(current_rewards) / len(current_rewards)
        max_balance = max(balances)
        min_balance = min(balances)
        avg_sharpe_ratio = sum(sharpe_ratios) / len(sharpe_ratios)
        
        most_common_action = action_map.get(str(action_counts.most_common(1)[0][0]), "Unknown")

        additional_analysis = f"""
        Average Test Reward: {avg_test_reward:.2f}
        Average Current Reward: {avg_current_reward:.2f}
        Max Balance: {max_balance:.2f}
        Min Balance: {min_balance:.2f}
        Average Sharpe Ratio: {avg_sharpe_ratio:.2f}
        Most Common Action: {most_common_action}
        """

        return templates.TemplateResponse("training_progress.html", {
            "request": request,
            "plot": plot_html,
            "summary": summary,
            "additional_analysis": additional_analysis
        })
    except Exception as e:
        logger.error(f"Error in get_training_progress: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)