import os

# กำหนด base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# กำหนด paths
TRAINING_DATA_PATH = os.path.join(BASE_DIR, 'training_data', 'Binance_BTCUSDT_1h.csv')
LOG_DIR = os.path.join(BASE_DIR, 'logs')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
CHECKPOINT_DIR = os.path.join(MODEL_DIR, 'checkpoints')
PROGRESS_REPORT_PATH = os.path.join(LOG_DIR, 'ai_progress_report.json')
PROCESS_NUM = os.cpu_count() or 4  # Fallback to 1 if os.cpu_count() returns None

# สร้าง directories ถ้ายังไม่มี
for dir_path in [LOG_DIR, MODEL_DIR, CHECKPOINT_DIR]:
    os.makedirs(dir_path, exist_ok=True)

def get_log_file_path(run_id):
    return os.path.join(LOG_DIR, f'BTC_AI_{run_id}.log')
    #return os.path.join(LOG_DIR, f'training.log')

def get_model_file_path(filename):
    return os.path.join(MODEL_DIR, 'episode', f'{filename}.h5')

def get_timeframe_model_file_path(filename):
    return os.path.join(MODEL_DIR, 'timeframe', f'{filename}.h5')

def get_final_model_file_path(filename):
    return os.path.join(MODEL_DIR, f'{filename}.h5')

def get_checkpoint_path(filename):
    return os.path.join(CHECKPOINT_DIR, f'{filename}.h5')