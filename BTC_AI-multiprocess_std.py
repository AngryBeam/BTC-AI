import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Input
from tensorflow.keras.models import Model
from keras.optimizers import Adam
import random
from collections import deque
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, SMAIndicator
from custom_ta import calculate_technical_indicators  # Import from custom_ta library
from TradingEmulator import TradingEmulator  # Import from TradingEmulator library
from logger import setup_logger
import multiprocessing
import os
import uuid
from tensorflow.keras import mixed_precision

os.environ["OMP_NUM_THREADS"] = "4"  # Adjust this number based on your CPU cores
os.environ["TF_NUM_INTRAOP_THREADS"] = "4"
os.environ["TF_NUM_INTEROP_THREADS"] = "4"


logger = setup_logger()
# Check for GPU availability
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
if len(tf.config.list_physical_devices('GPU')) > 0:
    print("GPU is available. Using GPU for training.")
else:
    print("No GPU available. Using CPU for training.")
print("CPU cores available: ", multiprocessing.cpu_count())


def split_data(data, start_index, batch_size=100, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    end_index = start_index + batch_size
    if end_index > len(data):
        end_index = len(data)
    
    batch = data.iloc[start_index:end_index]
    train_size = int(len(batch) * train_ratio)
    val_size = int(len(batch) * val_ratio)
    
    train = batch.iloc[:train_size]
    val = batch.iloc[train_size:train_size+val_size]
    test = batch.iloc[train_size+val_size:]
    
    return train, val, test, end_index

    
class DQNAgent:
    def __init__(self, state_size, action_size, max_positions=10):
        self.state_size = state_size
        self.action_size = action_size
        self.max_positions = max_positions
        self.total_actions = 1 + 2 + 2 * max_positions  # Hold, Open Long, Open Short, Close Long (partial), Close Short (partial)
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()  # เพิ่ม Target Network
        self.update_target_model()  # อัพเดท Target Network
        self.device = '/GPU:0' if len(tf.config.list_physical_devices('GPU')) > 0 else '/CPU:0'
    
    @tf.function
    def predict(self, state):
        return self.model(state)

    @tf.function
    def train_step(self, states, target_f):
        with tf.GradientTape() as tape:
            predictions = self.model(states)
            loss = self.model.loss(target_f, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss
    
    def _build_model(self):
        input_layer = Input(shape=(None, self.state_size))
        lstm1 = LSTM(256, return_sequences=True)(input_layer)
        lstm2 = LSTM(128)(lstm1)
        dense1 = Dense(64, activation='relu')(lstm2)
        output_layer = Dense(self.total_actions, activation='linear')(dense1)
        
        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model
    
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())  # อัพเดท Target Network

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.total_actions)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])
    
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])

        # Reshape next_states to match the expected input shape
        next_states = next_states.reshape(batch_size, -1, 16)

        targets = rewards + self.gamma * np.amax(self.target_model.predict(next_states), axis=1) * (1 - dones)
        target_f = self.model.predict(states.reshape(batch_size, -1, 16))
        for i, action in enumerate(actions):
            target_f[i][action] = targets[i]

        self.model.fit(states.reshape(batch_size, -1, 16), target_f, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def replay_old(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            
            with tf.device(self.device):
                self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def load(self, name):
        self.model.load_weights(name)
    
    def save(self, name):
        self.model.save_weights(name)
        self.model.save(f"FULL-{name}")

def run_episode(data, agent, episode, initial_balance, batch_size, scaler, return_dict):
    try:
        emulator = TradingEmulator(initial_balance=initial_balance)
        scaled_data = scaler.transform(data)
        
        state = scaled_data[0].reshape(1, 1, -1)  # Reshape to (batch_size, timesteps, features)
        total_reward = 0
        
        for t in range(1, len(scaled_data)):
            try:
                action = agent.act(state)
                
                if action == 0:  # Hold
                    pass
                elif action == 1:  # Open Long
                    emulator.open_position('long', 1, data.iloc[t]['Close'])
                elif action == 2:  # Open Short
                    emulator.open_position('short', 1, data.iloc[t]['Close'])
                else:  # Partial Close
                    close_action = (action - 3) // agent.max_positions
                    close_amount = ((action - 3) % agent.max_positions + 1) / agent.max_positions
                    if close_action == 0:  # Close Long
                        emulator.close_position('long', emulator.position['long'] * close_amount, data.iloc[t]['Close'])
                    else:  # Close Short
                        emulator.close_position('short', emulator.position['short'] * close_amount, data.iloc[t]['Close'])
                
                next_state = scaled_data[t].reshape(1, 1, -1)  # Reshape to (batch_size, timesteps, features)
                reward = emulator.balance - initial_balance
                done = t == len(scaled_data) - 1
                
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                
                emulator.update(data.iloc[t]['Close'])
                emulator.trailing_stop_loss(data.iloc[t]['Close'])
                
                if len(agent.memory) > batch_size:
                    agent.replay(batch_size)
                logger.info(f"run_episode:{episode+1}, t:{t}, range:{len(scaled_data)}")
            except Exception as e:
                logger.error(f"Error at step {t} in episode {episode+1}: {e}")
                break

        logger.info(f"Completed for loop in episode {episode+1}")
        
        metrics = emulator.get_performance_metrics()
        logger.info(f"Episode: {episode+1}")
        logger.info(f"Total Reward: {total_reward}")
        logger.info(f"Balance: {metrics['balance']:.2f}")
        logger.info(f"Unrealized P/L: {metrics['unrealized_pnl']:.2f}")
        logger.info(f"Total P/L: {metrics['total_pnl']:.2f}")
        logger.info(f"Max Profit: {metrics['max_profit']:.2f}")
        logger.info(f"Max Drawdown: {metrics['max_drawdown']:.2f}")
        logger.info(f"Capital Loaded (Long): {metrics['capital_loaded_long']:.2f}")
        logger.info(f"Capital Loaded (Short): {metrics['capital_loaded_short']:.2f}")
        logger.info(f"Total Capital Loaded: {metrics['total_capital_loaded']:.2f}")
        logger.info(f"Max Capital Loaded: {metrics['max_capital_loaded']:.2f}")
        logger.info(f"open_trade_long: {metrics['open_trade_long']:.2f}")
        logger.info(f"open_trade_short: {metrics['open_trade_short']:.2f}")
        logger.info(f"close_trade_long: {metrics['close_trade_long']:.2f}")
        logger.info(f"close_trade_short: {metrics['close_trade_short']:.2f}")
        logger.info(f"max_open_position_long: {metrics['max_open_position_long']:.2f}")
        logger.info(f"max_open_position_short: {metrics['max_open_position_short']:.2f}")
        logger.info(f"trading_count_long: {metrics['trading_count_long']:.2f}")
        logger.info(f"trading_count_short: {metrics['trading_count_short']:.2f}")
        logger.info("--------------------")
        
        return_dict[episode] = total_reward
        logger.info(f"Episode {episode+1} completed successfully")
    except Exception as e:
        logger.error(f"Error in episode {episode+1}: {e}")
        return_dict[episode] = None

def train_agent(data, agent, episodes=10, batch_size=64, initial_balance=100000):  # ปรับ Batch Size
    scaler = MinMaxScaler()
    scaler.fit(data)
    
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    
    best_reward = float('-inf')
    no_improvement = 0
    
    processes = []
    for episode in range(episodes):
        process = multiprocessing.Process(target=run_episode, args=(data, agent, episode, initial_balance, batch_size, scaler, return_dict))
        processes.append(process)
        process.start()
    
    for process in processes:
        process.join()
    
    # ตรวจสอบว่า return_dict ไม่ว่างเปล่า
    if not return_dict:
        raise ValueError("No episodes were completed successfully.")
    
    # ลบค่า None ออกจาก return_dict
    valid_rewards = {k: v for k, v in return_dict.items() if v is not None}
    if not valid_rewards:
        raise ValueError("No valid episodes were completed successfully.")
    
    best_episode = max(valid_rewards, key=valid_rewards.get)
    best_reward = valid_rewards[best_episode]
    
    logger.info(f"Best Episode: {best_episode+1} with Reward: {best_reward}")
    best_model_path = f"D:\\python\\BTC_AI\\models\\episode\\{str(uuid.uuid4())}.weights.h5"
    agent.save(best_model_path)
    
    # อัพเดท Target Network ทุกๆ M Episodes
    if (episode + 1) % 5 == 0:  # อัพเดททุกๆ 5 Episodes
        agent.update_target_model()
    
    return agent, best_model_path

def fine_tune_agent(data, agent, best_model_path, episodes=10, batch_size=64, initial_balance=100000):
    # โหลดโมเดลที่ดีที่สุด
    agent.load(best_model_path)
    
    # ฝึกโมเดลเพิ่มเติม
    agent, best_model_path = train_agent(data, agent, episodes=episodes, batch_size=batch_size, initial_balance=initial_balance)
    
    return agent, best_model_path
def setup_tensorflow():
    # Set TensorFlow to use all available cores
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
    
    # Optionally, you can also set memory growth if you're using a GPU
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)

    os.environ["OMP_NUM_THREADS"] = "8"  # Adjust this number based on your CPU cores
    os.environ["TF_NUM_INTRAOP_THREADS"] = "8"
    os.environ["TF_NUM_INTEROP_THREADS"] = "8"

def check_timeframe_data(timeframe, df):
    logger.info(f"Checking data for {timeframe} timeframe:")
    logger.info(f"Shape: {df.shape}")
    logger.info(f"Columns: {df.columns.tolist()}")
    logger.info(f"NaN values:\n{df.isna().sum()}")
    logger.info(f"Data types:\n{df.dtypes}")
    
    # Check for infinite values
    inf_count = np.isinf(df.select_dtypes(include=np.number)).sum().sum()
    logger.info(f"Infinite values: {inf_count}")
    
    # Check for consistent feature count
    expected_features = 16  # Adjust this based on your expected number of features
    if df.shape[1] != expected_features:
        logger.warning(f"Unexpected number of features: {df.shape[1]}. Expected: {expected_features}")
    
    # Check for very large or very small values
    numeric_cols = df.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        min_val = df[col].min()
        max_val = df[col].max()
        if abs(min_val) > 1e10 or abs(max_val) > 1e10:
            logger.warning(f"Column {col} has extreme values: min={min_val}, max={max_val}")
    
    logger.info("Data check completed.")
    logger.info("----------------------------------------")


def resample_data(df, timeframe):
    agg_dict = {
        'Unix': 'first',
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume BTC': 'sum',
        'Volume USDT': 'sum'
    }
    
    if timeframe == 'W-MON':
        resampled = df.resample(timeframe).agg(agg_dict)
    elif timeframe == 'D':
        resampled = df.resample(timeframe).agg(agg_dict)
    elif timeframe == '4H':
        resampled = df.resample(timeframe).agg(agg_dict)
    else:
        resampled = df  # For 1h, no resampling needed
    
    return resampled


if __name__ == "__main__":
    setup_tensorflow()
    # Read and prepare data
    df = pd.read_csv('training_data/Binance_BTCUSDT_1h.csv', skiprows=1).drop(['Symbol', 'tradecount'], axis=1)
    df_sorted = df.sort_values('Unix', ascending=True).reset_index(drop=True)
    df_sorted['Date'] = pd.to_datetime(df_sorted['Unix'], unit='ms')
    df_sorted.set_index('Date', inplace=True)

    for col in ['Unix', 'Open', 'High', 'Low', 'Close', 'Volume BTC', 'Volume USDT']:
        df_sorted[col] = pd.to_numeric(df_sorted[col], errors='coerce')

    timeframes = [
        ('weekly', resample_data(df_sorted, 'W-MON')),
        ('daily', resample_data(df_sorted, 'D')),
        ('4h', resample_data(df_sorted, '4H')),
        ('1h', df_sorted),
    ]

    best_model_path = None
    for timeframe, df in timeframes:
        logger.info(f"Training on {timeframe} timeframe, with data: {len(df)}")
        df = calculate_technical_indicators(df)
        df = df.dropna()
        logger.info(df.head())
        check_timeframe_data(timeframe, df)
        exit
        
        state_size = len(df.columns)
        action_size = 5  # No action, Buy, Sell, Close Long, Close Short
        agent = DQNAgent(state_size, action_size)
        
         # Use a larger batch size for better GPU utilization
        batch_size = 1024
        episodes = 50
        
        # Train on the entire dataset
        #agent = train_agent(df, agent, episodes=episodes, batch_size=batch_size)

  
        if best_model_path is None:
            # Train on the entire dataset
            agent, best_model_path = train_agent(df, agent, episodes=episodes, batch_size=batch_size)
        else:
            agent, best_model_path = fine_tune_agent(df, agent, best_model_path, episodes=episodes, batch_size=batch_size)
        # Fine-tune the best model in a loop
        
        fine_tune_iterations = 5
        for i in range(fine_tune_iterations):
            logger.info(f"Fine-tuning iteration {i+1}")
            agent, best_model_path = fine_tune_agent(df, agent, best_model_path, episodes=10, batch_size=batch_size)

        agent.save(f"models/{timeframe}-{str(uuid.uuid4())}.weights.h5")
        
        logger.info(f"Completed training on {timeframe} timeframe")
        logger.info("----------------------------------------")
    
    logger.info("Training completed on all timeframes")