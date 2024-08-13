import pandas as pd
from logger import get_logger

import os
import sys
from agent import DQNAgent
from data_processing import prepare_data, create_timeframes
from training import train_on_timeframes
from utils import setup_tensorflow
from config import TRAINING_DATA_PATH
from custom_ta import calculate_technical_indicators
import traceback

os.environ["OMP_NUM_THREADS"] = "4"  # Adjust this number based on your CPU cores
os.environ["TF_NUM_INTRAOP_THREADS"] = "4"
os.environ["TF_NUM_INTEROP_THREADS"] = "4"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

if __name__ == "__main__":
    try:
        print("Getting logger")
        logger, run_id = get_logger()
        print(f"Got logger with run_id: {run_id}")
        logger.info(f"Starting new run with ID: {run_id}")
        print("Logged start message")

        logger.info("Setting up TensorFlow...")
        setup_tensorflow()
        logger.info("TensorFlow setup completed")
        
        logger.info("Preparing data...")
        df_sorted = prepare_data(TRAINING_DATA_PATH)
        logger.info("Data preparation completed")
        
        logger.info("Creating timeframes...")
        timeframes = create_timeframes(df_sorted)
        logger.info("Timeframes created")

        logger.info("Adding technical indicators to timeframes...")
        for i, (tf_name, df) in enumerate(timeframes):
            logger.info(f"Processing timeframe: {tf_name}")
            logger.info(f"DataFrame shape: {df.shape}")
            logger.info(f"DataFrame columns: {df.columns}")
            
            df_with_indicators = calculate_technical_indicators(df)
            timeframes[i] = (tf_name, df_with_indicators)
        logger.info("Technical indicators added to all timeframes")
        #sys.exit()

        # ตรวจสอบ state_size จาก timeframe แรก (เช่น 'daily')
        first_tf_name, first_df = timeframes[0]
        state_size = len(first_df.columns)

        logger.info(f"First timeframe: {first_tf_name}")
        logger.info(f"Columns in first timeframe: {first_df.columns}")
        logger.info(f"State size determined from first timeframe: {state_size}")

        
        action_size = 5  # No action, Buy, Sell, Close Long, Close Short
        logger.info(f"State size: {state_size}, Action size: {action_size}")
        #sys.exit()
        performance_threshold = 0.8
        logger.info(f"Starting training with performance threshold: {performance_threshold}")
        train_on_timeframes(timeframes, state_size, action_size, performance_threshold)
        
        logger.info("Training completed on all timeframes")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        logger.error(traceback.format_exc())