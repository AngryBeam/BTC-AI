import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Input, Lambda, Add, BatchNormalization, Dropout, Attention
from tensorflow.keras.models import Model
from keras.optimizers import Adam
from collections import deque
from tensorflow.keras.losses import Huber
import tensorflow.keras.backend as K

import random
from TradingEmulator import TradingEmulator  # Import from TradingEmulator library
from logger import get_logger
#from prioritized_replay_buffer import PrioritizedReplayBuffer
from prioritized_replay_buffer import PrioritizedReplayBuffer

import os
import json
from tensorflow.keras.models import save_model, load_model
from tensorflow.keras.regularizers import l2
from MonteCarloTreeSearch import MCTS

logger, _ = get_logger()
class DQNAgent:
    def __init__(self, state_size, action_size, max_positions=10, use_mcts=True, mcts_simulations=1000, mcts_exploration=1.41, mcts_max_depth=50):
        logger = get_logger()[0]
        logger.info(f"Initializing DQNAgent with state_size: {state_size}, action_size: {action_size}")
        
        self.state_size = state_size
        self.action_size = action_size
        self.max_positions = max_positions
        #self.total_actions = 1 + 2 + 2 * max_positions
        #self.memory = deque(maxlen=10000)
        #self.memory = PrioritizedReplayBuffer(capacity=10000)
        self.memory = PrioritizedReplayBuffer(capacity=10000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.epsilon_increase_rate = 1.005
        self.epsilon_floor = 0.05  # ค่า epsilon ขั้นต่ำที่จะไม่เพิ่มขึ้น
        self.epsilon_decay_rate = 0.995
        self.epsilon_max = 1.0
        # Create the optimizer once
        optimizer = Adam(learning_rate=self.learning_rate, clipnorm=1.0)
        self.optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
        self.loss_history = []
        self.action_history = []

        if len(tf.config.list_physical_devices('GPU')) > 0:
            self.device = '/GPU:0'
            logger.info(f"DQNAgent Using GPU")
        else:
            self.device = '/CPU:0'
            logger.info(f"DQNAgent Using CPU")

        #To change action map
        #1: "Open Long",
        #2: "Open Short",
        #It using in MCTS method
        self.action_map = {
            0: "Hold",
            1: "Open Long",
            2: "Open Short",
            3: "Close All",
            4: "Close All Long",
            5: "Close All Short",
            6: "Partial Close Long",
            7: "Partial Close Short",
            8: "Hold Long",
            9: "Hold Short",
            10: "Hold Long, Close Short",
            11: "Hold Short, Close Long",
            12: "Hold Long, Partial Close Short",
            13: "Hold Short, Partial Close Long",
            14: "Open More Long (Dip)",
            15: "Open More Short (Spike)",
            16: "Open More Long (Spike)",
            17: "Open More Short (Dip)"
        }

        self.total_actions = len(self.action_map)
        self.action_size = self.total_actions

        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

        self.profitable_pnl = []
        self.pnl_history = deque(maxlen=100)
        self.growth_rate_history = deque(maxlen=20)
        
        self.high_water_mark = float('-inf')
        self.stable_pnl_count = 0
        self.stable_pnl_threshold = 5  # จำนวนครั้งที่ PNL คงที่ก่อนที่จะเริ่มลด epsilon
        

        self.use_mcts = use_mcts
        # สำหรับ MCTS
        self.position_index = 0  # ปรับตามโครงสร้างข้อมูลของคุณ
        self.step_index = 0  # ปรับตามโครงสร้างข้อมูลของคุณ
        self.balance_index = 2  # ปรับตามโครงสร้างข้อมูลของคุณ
        self.previous_balance_index = 3  # ปรับตามโครงสร้างข้อมูลของคุณ
        self.max_steps = 1000  # ปรับตามความเหมาะสม
        self.profitable_trade_bonus = 10  # ปรับตามความเหมาะสม
        self.losing_trade_penalty = 5  # ปรับตามความเหมาะสม
        if self.use_mcts:
            self.mcts = MCTS(self, num_simulations=mcts_simulations, 
                             exploration_constant=mcts_exploration,
                             max_depth=mcts_max_depth)
            self.mcts_used = 0
        
    @tf.function
    def predict(self, state):
        return self.model(state)

    '''
    custom training loop , replacing .fit()
    '''
    @tf.function
    def train_step(self, states, target_f):
        with tf.GradientTape() as tape:
            predictions = self.model(states)
            loss = self.model.loss(target_f, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        # Check for gradient explosion
        if any(np.isinf(g).any() or np.isnan(g).any() for g in gradients):
            logger.error("train_step - Gradient explosion detected. Skipping this batch.")
       
        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss
    
    def _build_model(self):
        logger = get_logger()[0]
        logger.info(f"Building model with input shape: (None, None, {self.state_size})")
        
        input_layer = Input(shape=(None, self.state_size))
        lstm1 = LSTM(256, return_sequences=True)(input_layer)
        #lstm1 = BatchNormalization()(lstm1)
        lstm2 = LSTM(128)(lstm1)
        #lstm2 = BatchNormalization()(lstm2)
        dense1 = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(lstm2)
        dense1 = Dropout(0.2)(dense1)
        dense2 = Dense(32, activation='relu', kernel_regularizer=l2(0.01))(dense1)
        dense2 = Dropout(0.2)(dense2)

        output_layer = Dense(self.total_actions, activation='linear')(dense2)
        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(loss='mse', optimizer=self.optimizer)
        #clipnorm=1.0
        '''
        input_layer = Input(shape=(None, self.state_size))

        # LSTM layers with residual connections
        lstm1 = LSTM(128, return_sequences=True)(input_layer)
        lstm1 = BatchNormalization()(lstm1)
        lstm2 = LSTM(128, return_sequences=True)(lstm1)
        lstm2 = BatchNormalization()(lstm2)
        residual = Add()([lstm1, lstm2])

        # Attention mechanism
        attention_output = Attention()([residual, residual])
        

        # Flatten the output for dense layers
        #flattened = tf.keras.layers.Flatten()(attention_output)
        # Use GlobalAveragePooling1D instead of Flatten
        pooled = tf.keras.layers.GlobalAveragePooling1D()(attention_output)
        reshaped = tf.keras.layers.Reshape((128,))(pooled)  # ปรับ 128 ตามจำนวน features ที่คุณต้องการ
        # Dense layers
        dense1 = Dense(64, activation='relu')(reshaped)
        dense1 = Dropout(0.2)(dense1)
        dense2 = Dense(32, activation='relu')(dense1)
        dense2 = Dropout(0.2)(dense2)

        # Value stream
        value_stream = Dense(32, activation='relu')(dense2)
        value = Dense(1)(value_stream)
        
        # Advantage stream
        advantage_stream = Dense(32, activation='relu')(dense2)
        advantage = Dense(self.total_actions)(advantage_stream)

        # Combine value and advantage
        output_layer = Add()([
            value,
            Lambda(lambda a: a - K.mean(a, axis=1, keepdims=True))(advantage)
        ])
        
        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(loss=Huber(), optimizer=self.optimizer)

        # Print model summary for debugging
        model.summary()
        logger.info(f"Model output shape: {model.output_shape}")
        '''
        return model
    
    def update_epsilon(self, current_pnl):
        self.pnl_history.append(current_pnl)
        
        if len(self.pnl_history) < 2:
            return self.epsilon

        # อัปเดต High Water Mark
        if current_pnl > self.high_water_mark:
            self.high_water_mark = current_pnl
            self.stable_pnl_count = 0
        else:
            self.stable_pnl_count += 1

        # คำนวณอัตราการเติบโต
        previous_pnl = self.pnl_history[-2]
        if previous_pnl != 0:
            growth_rate = (current_pnl - previous_pnl) / abs(previous_pnl)
        else:
            growth_rate = 0

        self.growth_rate_history.append(growth_rate)

        if len(self.growth_rate_history) < 5:
            return self.epsilon

        # คำนวณ growth score
        avg_growth = np.mean(self.growth_rate_history)
        growth_std = np.std(self.growth_rate_history)
        growth_score = avg_growth / (growth_std + 1e-6)

        # ปรับ epsilon
        if growth_score > 0 and self.epsilon > self.epsilon_min:
            # ลด epsilon ถ้ามีการเติบโตที่ดี
            self.epsilon *= self.epsilon_decay_rate
        elif growth_score < -0.5 and self.epsilon < self.epsilon_floor:
            # เพิ่ม epsilon ถ้าผลการดำเนินงานแย่ลงและ epsilon ต่ำกว่า floor
            self.epsilon *= self.epsilon_increase_rate
        elif self.stable_pnl_count > self.stable_pnl_threshold:
            # ลด epsilon อย่างช้าๆ ถ้า PNL คงที่เป็นเวลานาน
            self.epsilon *= (1 - 0.1 * self.epsilon_decay_rate)

        # จำกัด epsilon
        self.epsilon = max(self.epsilon_min, min(self.epsilon, self.epsilon_max))

        return self.epsilon
    
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())  # อัพเดท Target Network

    def memory_size(self):
        return len(self.memory)
        
    def remember(self, state, action, reward, next_state, done):
        # Ensure state and next_state are 2D
        #Comment zone is stable version using with self.memory = deque(maxlen=10000)
        #state = np.reshape(state, (1, self.state_size))
        #next_state = np.reshape(next_state, (1, self.state_size))
        #self.memory.append((state, action, reward, next_state, done))
        # Ensure state and next_state are 2D
        # Ensure state and next_state are 2D
        if np.isnan(state).any() or np.isinf(state).any():
            logger.error(f"Agent.Remember - State contains invalid values: {state}")
        
        experience = (state, action, reward, next_state, done)
        state = np.reshape(state, (1, 1, self.state_size))
        next_state = np.reshape(next_state, (1, 1, self.state_size))
        prediction = self.model.predict(state)

        if np.isnan(prediction).any():
            logger.error(f"Agent.Remember - Prediction contains NaN values: {prediction}")
        if np.isinf(prediction).any():
            logger.error(f"Prediction contains infinity values: {prediction}")
            # Replace inf values with a large finite number
            #prediction = np.nan_to_num(prediction, posinf=1e30, neginf=-1e30)
        
        if action >= prediction.shape[1]:
            logger.error(f"Agent.Remember - Action {action} is out of bounds for prediction shape {prediction.shape}")
       
        
        if prediction.shape[1] != self.total_actions:
            logger.error(f"Agent.Remember - Prediction shape: {prediction.shape}, Action: {action}")
            logger.error(f"Agent.Remember - Mismatch between prediction shape {prediction.shape} and total_actions {self.total_actions}")
            
        target = reward + self.gamma * np.max(self.target_model.predict(next_state)[0]) * (1 - done)
        current_q = prediction[0][action]
        error = abs(target - current_q)

        if np.isinf(current_q):
            logger.error(f"current_q is infinity. Using a large finite number.")
            #current_q = -1e30 if current_q < 0 else 1e30
        if np.isinf(target):
            logger.error(f"target is infinity. Using a large finite number.")
            #target = 1e30 if target > 0 else -1e30
    

        
        prev_size = len(self.memory)
        if np.isnan(error) or np.isinf(error):
            logger.error(f"Agent.Remember - Warning: Invalid error. State: {state}, Action: {action}, Reward: {reward}, error var value: {error}, target:{target}, current_q:{current_q}, action:{action}, prediction:{prediction}")
            error = 1e-6  # ใช้ค่าเริ่มต้นถ้า error ไม่ถูกต้อง
            # ตรวจสอบ weights
            for layer in self.model.layers:
                weights = layer.get_weights()
                if any(np.isnan(w).any() for w in weights):
                    logger.error(f"Agent.Remember - NaN weights detected in layer: {layer.name}")
            
        self.memory.add(max(error, 1e-6), experience)
        new_size = len(self.memory)
        logger.debug(f"Agent.Remember - Memory size before: {prev_size}, after: {new_size}")
        self.action_history.append(action)

    @tf.function
    def _dqn_action(self, state):
        act_values = self.model(state)
        return tf.argmax(act_values[0], output_type=tf.int32)

    def act(self, state):
        if tf.random.uniform((), dtype=tf.float32) <= self.epsilon:
            return tf.random.uniform((), minval=0, maxval=self.total_actions, dtype=tf.int32)
        else:
            #state_reshaped = tf.reshape(state, (1, 1, -1))
            state_reshaped = tf.reshape(state, (1, 1, self.state_size))
            
            if self.use_mcts:
                logger.info(f'ACT using MCTS with:{state_reshaped}')
                mcts_action = self.mcts.search(state_reshaped)
                if mcts_action is not None and mcts_action < self.total_actions:
                    self.mcts_used += 1
                    return tf.constant(mcts_action, dtype=tf.int32)
            
            return self._dqn_action(state_reshaped)

    def replay(self, batch_size):
        logger.info(f"Replay memory size: {self.memory_size()}")
        if self.memory_size() < batch_size:
            logger.info(f"Not enough samples in memory. Current size: {self.memory_size()}")
            return
        
        batch_size = min(batch_size, len(self.memory))
        logger.info(f'replay batch_size: {batch_size}')
        
        samples, idxs, is_weights = self.memory.sample(batch_size)
        
        logger.info(f'replay samples: {samples[:5]}')
        print(f"Samples type: {type(samples)}")

        if all(s == 0 for s in samples):
            logger.warning("All samples are 0. Checking memory content...")
            for i in range(min(10, len(self.memory))):
                logger.info(f"Memory item {i}: {self.memory[i]}")
                
        states = np.array([s[0] for s in samples])
        actions = np.array([s[1] for s in samples])
        rewards = np.array([s[2] for s in samples])
        next_states = np.array([s[3] for s in samples])
        dones = np.array([s[4] for s in samples])

        states = np.reshape(states, (batch_size, 1, self.state_size))
        next_states = np.reshape(next_states, (batch_size, 1, self.state_size))

        targets = rewards + self.gamma * (np.amax(self.target_model.predict(next_states), axis=1)) * (1 - dones)
        targets_full = self.model.predict(states)

        for i, action in enumerate(actions):
            clipped_target = np.clip(targets[i], np.finfo(targets_full.dtype).min, np.finfo(targets_full.dtype).max)
            targets_full[i, action] = clipped_target

        loss = self.model.train_on_batch(states, targets_full, sample_weight=is_weights)
        self.loss_history.append(loss)
        
        # Update priorities
        new_priorities = np.abs(targets - targets_full[np.arange(batch_size), actions])
        new_priorities = np.clip(new_priorities, 1e-8, None)  # Ensure priorities are not too small
        
        for i in range(batch_size):
            idx = idxs[i]
            self.memory.update(idx, new_priorities[i])

        #As we using update_epsilon run on process_episode_results inside training
        #if self.epsilon > self.epsilon_min:
        #    self.epsilon *= self.epsilon_decay

        #Beta annealing: This gradually increases the importance sampling correction over time.
        self.memory.increase_beta()

        return loss

    def save(self, name):
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(name), exist_ok=True)
            
            # Save the model in .keras format
            save_model(self.model, f"{name}")
            
            logger.info(f"Successfully saved model to {name}.keras")
        except Exception as e:
            logger.error(f"Error saving model to {name}.keras: {str(e)}")
            raise

    def load(self, name):
        try:
            # Load the model in .keras format
            loaded_model = load_model(f"{name}", compile=False)
            
            # Copy weights to our models
            self.model.set_weights(loaded_model.get_weights())
            self.target_model.set_weights(loaded_model.get_weights())
            
            # Recompile the models
            #self.model.compile(loss='mse', optimizer=self.optimizer)
            #self.target_model.compile(loss='mse', optimizer=self.optimizer)
            
            logger.info(f"Successfully loaded model from {name}.keras")
        except Exception as e:
            logger.error(f"Error loading model from {name}.keras: {str(e)}")
            raise


    def update_learning_rate(self, factor):
        self.learning_rate *= factor
        self.model.optimizer.learning_rate.assign(self.learning_rate)

    def get_average_loss(self):
        return np.mean(self.loss_history) if self.loss_history else None

    def get_action_distribution(self):
        if not self.action_history:
            return None
        unique, counts = np.unique(self.action_history, return_counts=True)
        return {self.action_map[action]: count for action, count in zip(unique, counts)}
    def reset_history(self):
        self.loss_history = []
        self.action_history = []


    '''
    MTCS Method
    '''

    def _ensure_2d(self, state):
        if isinstance(state, tf.Tensor):
            if state.shape.ndims == 3:
                return tf.squeeze(state, axis=0)
        elif isinstance(state, np.ndarray):
            if state.ndim == 3:
                return np.squeeze(state, axis=0)
        return state

    def is_exploring(self, state):
        return tf.random.uniform((), dtype=tf.float32) <= self.epsilon

    # เพิ่มเมธอดสำหรับ MCTS
    def get_possible_actions(self, state):
        # Return list of possible actions based on current state
        return list(range(self.total_actions))

    def get_next_state(self, state, action):
        # สร้างสำเนาของสถานะปัจจุบัน
        next_state = np.copy(state)
        if self.position_index < len(next_state):
            # จำลองผลของการกระทำ
            if action == self.action_map.get('Open Long'):
                # ปรับปรุงสถานะเมื่อเปิด Long position
                next_state[self.position_index] = 1
            elif action == self.action_map.get('Open Short'):
                # ปรับปรุงสถานะเมื่อเปิด Short position
                next_state[self.position_index] = -1
            # ... (เพิ่มเงื่อนไขสำหรับการกระทำอื่นๆ)
        
        #return next_state
        else:
            logger.warning(f"position_index {self.position_index} is out of bounds for state with length {len(next_state)}")
        
        return next_state.reshape(1, 1, -1)

    def is_terminal(self, state):
        state = self._ensure_2d(state)
        if state.shape[1] <= max(self.step_index, self.balance_index):
            logger.warning(f"State shape {state.shape} is smaller than expected.")
            return True
        current_step = state[0, self.step_index]
        current_balance = state[0, self.balance_index]
        return current_step >= self.max_steps or current_balance <= 0

    def get_reward(self, state):
        state = self._ensure_2d(state)
        if state.shape[1] <= max(self.previous_balance_index, self.balance_index):
            logger.warning(f"State shape {state.shape} is smaller than expected.")
            return 0
        previous_balance = state[0, self.previous_balance_index]
        current_balance = state[0, self.balance_index]
        return current_balance - previous_balance