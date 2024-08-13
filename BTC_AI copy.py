import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
import random
from collections import deque
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, SMAIndicator
from custom_ta import calculate_technical_indicators  # Import from custom_ta library
from TradingEmulator import TradingEmulator  # Import from TradingEmulator library
from logger import setup_logger

logger = setup_logger()
# Check for GPU availability
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
if len(tf.config.list_physical_devices('GPU')) > 0:
    print("GPU is available. Using GPU for training.")
else:
    print("No GPU available. Using CPU for training.")


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
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(LSTM(64, input_shape=(self.state_size, 1), return_sequences=True))
        model.add(LSTM(32, return_sequences=False))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(self.total_actions, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.total_actions)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])
    
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def load(self, name):
        self.model.load_weights(name)
    
    def save(self, name):
        self.model.save_weights(name)

def train_agent(data, agent, episodes=10, batch_size=32, initial_balance=100000):
    emulator = TradingEmulator(initial_balance=initial_balance)
    
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    
    for episode in range(episodes):
        state = scaled_data[0].reshape(1, -1, 1)
        total_reward = 0
        
        for t in range(1, len(scaled_data)):
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
            
            next_state = scaled_data[t].reshape(1, -1, 1)
            reward = emulator.balance - initial_balance
            done = t == len(scaled_data) - 1
            
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            emulator.update(data.iloc[t]['Close'])
            emulator.trailing_stop_loss(data.iloc[t]['Close'])
            
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        
        metrics = emulator.get_performance_metrics()
        logger.info(f"Episode: {episode+1}/{episodes}")
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
        logger.info("--------------------")
    
    return agent

if __name__ == "__main__":
    # Read and prepare data
    df = pd.read_csv('Binance_BTCUSDT_1h.csv', skiprows=1).drop('Symbol', axis=1)
    df_sorted = df.sort_values('Unix', ascending=True).reset_index(drop=True)
    df_sorted['Date'] = pd.to_datetime(df_sorted['Unix'], unit='ms')
    df_sorted.set_index('Date', inplace=True)

    for col in ['Open', 'High', 'Low', 'Close', 'Volume BTC']:
        df_sorted[col] = pd.to_numeric(df_sorted[col], errors='coerce')

        #('4h', df_sorted.resample('4H').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume BTC': 'sum'})),
        #('daily', df_sorted.resample('D').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume BTC': 'sum'})),
        #('weekly', df_sorted.resample('W-MON').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume BTC': 'sum'}))
    

    timeframes = [
        ('1h', df_sorted),
        ]

    for timeframe, df in timeframes:
        logger.info(f"Training on {timeframe} timeframe")
        df = calculate_technical_indicators(df)
        
        state_size = len(df.columns)
        action_size = 5  # No action, Buy, Sell, Close Long, Close Short
        agent = DQNAgent(state_size, action_size)
        
        start_index = 0
        batch_size=720*3
        train_ratio=0.8
        val_ratio=0.10
        test_ratio=0.10
        while start_index < len(df):
            
            train, val, test, end_index = split_data(df, start_index, batch_size=batch_size, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio)
            
            # Combine previous validation and test data with new training data
            if start_index > 0:
                prev_val_test = df.iloc[start_index-(int(batch_size*(val_ratio+test_ratio))):start_index]
                train = pd.concat([prev_val_test, train])
            
            logger.info(f"Training from start_index:{start_index}, to end_index:{end_index}")
            agent = train_agent(train, agent)
            
            # Validate and test the agent here
            # You can create separate functions for validation and testing
            
            start_index = end_index
        
        agent.save(f"model_weights_{timeframe}.h5")
        logger.info(f"Completed training on {timeframe} timeframe")
        logger.info("----------------------------------------")

    logger.info("Training completed on all timeframes")