import pandas as pd
from logger import get_logger
import uuid
import os
import sys
from agent import DQNAgent
from data_processing import prepare_data, create_timeframes
from training import train_on_timeframes
from utils import setup_tensorflow
from config import TRAINING_DATA_PATH
from custom_ta import calculate_technical_indicators
import traceback
import multiprocessing
import platform
import tensorflow as tf

#os.environ["OMP_NUM_THREADS"] = "4"  # Adjust this number based on your CPU cores
#os.environ["TF_NUM_INTRAOP_THREADS"] = "4"
#os.environ["TF_NUM_INTEROP_THREADS"] = "4"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
gpus = tf.config.list_physical_devices('GPU')
tf.config.threading.set_intra_op_parallelism_threads(0)
tf.config.threading.set_inter_op_parallelism_threads(0)
#tf.config.optimizer.set_jit(True)  # เปิดใช้ XLA JIT compilation

if __name__ == "__main__":
    train_id = run_id = str(uuid.uuid4())

    if os.name == 'posix':  # 'posix' represents Unix-like systems, including Linux
        #pass
        multiprocessing.set_start_method('spawn')
    if gpus:
        try:
            # ใช้ GPU ทั้งหมดที่มี
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)
    else:
        print("No GPU available, using CPU instead")
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
        
        '''
        Only work with Standard and multiprocessing
        '''
        methods = ['standard', 'multiprocessing']
        select_method = methods[0]

        logger.info(f"Processing Training with method: {select_method}")
        train_on_timeframes(timeframes, state_size, action_size, performance_threshold,use_process_num=8, training_method=select_method, mcts=True)
        
        logger.info("Training completed on all timeframes")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        logger.error(traceback.format_exc())