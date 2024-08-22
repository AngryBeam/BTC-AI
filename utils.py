import json
import glob
from config import CHECKPOINT_DIR, PROGRESS_REPORT_PATH
import os
import tensorflow as tf
from tensorflow.keras import mixed_precision
from logger import get_logger
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

logger, _ = get_logger()
def setup_tensorflow():
    logger.info('Staring setup_tensorflow')
    logger.info('Setting up additional config')
    
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)

def scaler_info(scaler, data):
    # Log sample of original data
    logger.info("Sample of original data:")
    logger.info(data.head())
    
    # Log sample of scaled data
    scaled_data = scaler.transform(data.values)
    logger.info("Sample of scaled data:")
    logger.info(scaled_data[:5])  # Log first 5 rows
    
    # Log scaler parameters
    logger.info("Scaler parameters:")
    logger.info(f"Data min: {scaler.data_min_}")
    logger.info(f"Data max: {scaler.data_max_}")
    logger.info(f"Data range: {scaler.data_range_}")
    
    # Log statistics of scaled data
    logger.info("Scaled data statistics:")
    logger.info(f"Min values: {np.min(scaled_data, axis=0)}")
    logger.info(f"Max values: {np.max(scaled_data, axis=0)}")
    logger.info(f"Mean values: {np.mean(scaled_data, axis=0)}")
    logger.info(f"Std values: {np.std(scaled_data, axis=0)}")

def tensor_to_serializable(obj):
    if isinstance(obj, tf.Tensor):
        return obj.numpy().tolist()
    elif isinstance(obj, dict):
        return {k: tensor_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [tensor_to_serializable(v) for v in obj]
    else:
        return obj
    
def report_progress(agent, episode, timeframe, test_reward, current_reward, best_reward, retrain_attempts, avg_metrics, test_metrics):
    # ลบ balance_history จาก avg_metrics และ test_metrics ก่อน
    if isinstance(avg_metrics, dict) and ('balance_history' in avg_metrics):
        avg_metrics.pop('balance_history', None)
    
    if isinstance(test_metrics, dict) and ('balance_history' in test_metrics):
        test_metrics.pop('balance_history', None)

    
    progress = {
        'timestamp': datetime.datetime.now().strftime("%Y %b %d %H:%M:%S"),
        'episode': episode,
        'timeframe': timeframe,
        'test_reward': tensor_to_serializable(test_reward),
        'current_reward': tensor_to_serializable(current_reward),
        'best_reward': tensor_to_serializable(best_reward),
        'retrain_attempts': retrain_attempts,
        'avg_metrics': tensor_to_serializable(avg_metrics),
        'test_metrics': tensor_to_serializable(test_metrics)
    }
    
    # Convert agent attributes to serializable format
    #agent_dict = {k: tensor_to_serializable(v) for k, v in agent.__dict__.items() if not k.startswith('_')}
    #progress['agent'] = agent_dict

    with open(PROGRESS_REPORT_PATH, 'a') as f:
        json.dump(progress, f, default=lambda o: f"<<non-serializable: {type(o).__qualname__}>>")
        f.write('\n') 

def find_latest_checkpoint():
    checkpoints = glob.glob(os.path.join(CHECKPOINT_DIR, "*.keras"))
    if checkpoints:
        latest = max(checkpoints, key=os.path.getctime)
        return latest
    return None

def get_last_trained_timeframe():
    try:
        with open('progress.json', 'r') as f:
            lines = f.readlines()
        
        if not lines:
            return None
        
        for line in reversed(lines):
            try:
                last_report = json.loads(line.strip())
                return last_report.get('timeframe')
            except json.JSONDecodeError:
                continue
        
        return None
    except FileNotFoundError:
        return None

def analyze_progress_report(file_path='logs/ai_progress_report.json'):
    with open(file_path, 'r') as f:
        progress_data = [json.loads(line) for line in f]
    
    analysis = []
    episodes = []
    test_rewards = []
    current_rewards = []
    best_rewards = []
    timestamps = []

    for entry in progress_data:
        episodes.append(entry['episode'])
        test_rewards.append(entry['test_reward'])
        current_rewards.append(entry['current_reward'])
        best_rewards.append(entry['best_reward'])
        timestamps.append(datetime.datetime.fromisoformat(entry['timestamp']))
        
        analysis_entry = {
            'episode': entry['episode'],
            'test_reward': entry['test_reward'],
            'current_reward': entry['current_reward'],
            'best_reward': entry['best_reward'],
            'timestamp': entry['timestamp'],
            'test_metrics': entry.get('test_metrics', {}),
            'avg_metrics': entry.get('avg_metrics', {}),
            'agent': entry.get('agent', {})
        }
        analysis.append(analysis_entry)

    # Calculate time differences
    time_diffs = [(timestamps[i+1] - timestamps[i]).total_seconds() / 60 for i in range(len(timestamps)-1)]
    avg_time_per_episode = sum(time_diffs) / len(time_diffs) if time_diffs else 0

    # Analyze metrics
    last_entry = progress_data[-1]
    avg_metrics = last_entry['avg_metrics']
    test_metrics = last_entry['test_metrics']

    summary = {
        "progress_count": len(episodes),
        "avg_time_per_episode": round(avg_time_per_episode, 2),
        "final_test_reward": round(test_rewards[-1], 2),
        "final_current_reward": round(current_rewards[-1], 2),
        "final_best_reward": round(best_rewards[-1], 2),
        "avg_metrics": avg_metrics,
        "test_metrics": test_metrics
    }

    return {
        "summary": summary,
        "analysis": analysis
    }

def calculate_market_volatility(data, window=20, trading_periods=252):
        """
        Calculate the market volatility using a rolling standard deviation of returns.
        
        :param data: pandas DataFrame with a 'Close' column
        :param window: the rolling window for calculating volatility (default: 20 days)
        :param trading_periods: number of trading periods in a year (default: 252 for daily data)
        :return: annualized volatility
        """
        # Calculate daily returns
        returns = data['Close'].pct_change()
        
        # Calculate rolling standard deviation of returns
        rolling_std = returns.rolling(window=window).std()
        
        # Annualize the volatility
        annualized_vol = rolling_std * np.sqrt(trading_periods)
        
        return annualized_vol.iloc[-1]  # Return the most recent volatility value