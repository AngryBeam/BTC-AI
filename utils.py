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


logger, _ = get_logger()
def setup_tensorflow():
    logger.info('Staring setup_tensorflow')
    # Set TensorFlow to use all available cores
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
    # Check for GPU availability
    logger.info('Checking for GPU available')
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        logger.info("GPU is available. Using GPU for training.")
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    logger.info('Setting up additional config')
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)

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
    progress = {
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
    agent_dict = {k: tensor_to_serializable(v) for k, v in agent.__dict__.items() if not k.startswith('_')}
    progress['agent'] = agent_dict

    with open(PROGRESS_REPORT_PATH, 'a') as f:
        json.dump(progress, f, default=lambda o: f"<<non-serializable: {type(o).__qualname__}>>")
        f.write('\n') 

def find_latest_checkpoint():
    checkpoints = glob.glob(os.path.join(CHECKPOINT_DIR, "*.h5"))
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

def analyze_progress_report():
    df = pd.read_json(PROGRESS_REPORT_PATH, lines=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Instead of plotting, return the data
    analysis = df[['episode', 'test_reward', 'best_reward']].to_dict('records')
    
    summary = {
        "total_episodes": df['episode'].max(),
        "total_retrain_attempts": df['retrain_attempts'].max(),
        "best_reward_achieved": df['best_reward'].max(),
        "latest_test_reward": df['test_reward'].iloc[-1]
    }
    
    return {
        "analysis": analysis,
        "summary": summary
    }