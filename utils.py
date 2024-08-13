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

def report_progress(agent, episode, timeframe, test_reward, current_reward, best_reward, retrain_attempts, metrics):
    progress = {
        "timestamp": datetime.datetime.now().isoformat(),
        "episode": episode,
        "timeframe": timeframe,
        "test_reward": test_reward,
        "current_reward": current_reward,
        "best_reward": best_reward,
        "retrain_attempts": retrain_attempts,
        "metrics": metrics
    }
    with open(PROGRESS_REPORT_PATH, "a") as f:
        json.dump(progress, f)
        f.write("\n")
    logger.info(f"Progress Report: Episode {episode}, Timeframe {timeframe}, Test Reward {test_reward}")

def find_latest_checkpoint():
    checkpoints = glob.glob(os.path.join(CHECKPOINT_DIR, "*.h5"))
    if checkpoints:
        latest = max(checkpoints, key=os.path.getctime)
        return latest
    return None

def get_last_trained_timeframe():
    try:
        with open(PROGRESS_REPORT_PATH, "r") as f:
            lines = f.readlines()
            if lines:
                last_report = json.loads(lines[-1])
                return last_report["timeframe"]
    except FileNotFoundError:
        pass
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