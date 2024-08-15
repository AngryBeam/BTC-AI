import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Input
from tensorflow.keras.models import Model
from keras.optimizers import Adam
from collections import deque
import random
from TradingEmulator import TradingEmulator  # Import from TradingEmulator library
from logger import get_logger
#from prioritized_replay_buffer import PrioritizedReplayBuffer
from prioritized_replay_buffer import PrioritizedReplayBuffer, RayCompatiblePrioritizedReplayBuffer
import ray
import os
import json
from tensorflow.keras.models import save_model, load_model

logger, _ = get_logger()
class DQNAgent:
    def __init__(self, state_size, action_size, max_positions=10, use_ray=False):
        logger = get_logger()[0]
        logger.info(f"Initializing DQNAgent with state_size: {state_size}, action_size: {action_size}")
        
        self.state_size = state_size
        self.action_size = action_size
        self.max_positions = max_positions
        self.total_actions = 1 + 2 + 2 * max_positions
        #self.memory = deque(maxlen=10000)
        #self.memory = PrioritizedReplayBuffer(capacity=10000)
        self.use_ray = use_ray
        if use_ray:
            self.memory = RayCompatiblePrioritizedReplayBuffer.remote(capacity=10000)
        else:
            self.memory = PrioritizedReplayBuffer(capacity=10000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        # Create the optimizer once
        optimizer = Adam(learning_rate=self.learning_rate)
        self.optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
        
        
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
       
        self.loss_history = []
        self.action_history = []

        if len(tf.config.list_physical_devices('GPU')) > 0:
            self.device = '/GPU:0'
            logger.info(f"DQNAgent Using GPU")
        else:
            self.device = '/CPU:0'
            logger.info(f"DQNAgent Using CPU")

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
        logger = get_logger()[0]
        logger.info(f"Building model with input shape: (None, None, {self.state_size})")
        
        input_layer = Input(shape=(None, self.state_size))
        lstm1 = LSTM(256, return_sequences=True)(input_layer)
        lstm2 = LSTM(128)(lstm1)
        dense1 = Dense(64, activation='relu')(lstm2)
        output_layer = Dense(self.total_actions, activation='linear')(dense1)
        
        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(loss='mse', optimizer=self.optimizer)
        return model
    
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())  # อัพเดท Target Network

    def memory_size(self):
        if self.use_ray:
            return ray.get(self.memory.__len__.remote())
        else:
            return len(self.memory)
        
    def remember(self, state, action, reward, next_state, done):
        # Ensure state and next_state are 2D
        #Comment zone is stable version using with self.memory = deque(maxlen=10000)
        #state = np.reshape(state, (1, self.state_size))
        #next_state = np.reshape(next_state, (1, self.state_size))
        #self.memory.append((state, action, reward, next_state, done))
        # Ensure state and next_state are 2D
        # Ensure state and next_state are 2D
        experience = (state, action, reward, next_state, done)
        state = np.reshape(state, (1, 1, self.state_size))
        next_state = np.reshape(next_state, (1, 1, self.state_size))
        
        target = reward + self.gamma * np.max(self.target_model.predict(next_state)[0]) * (1 - done)
        current_q = self.model.predict(state)[0][action]
        error = abs(target - current_q)
        
        if self.use_ray:
            ray.get(self.memory.add.remote(max(error, 1e-6), experience))
        else:
            self.memory.add(max(error, 1e-6), experience)
        self.action_history.append(action)

    @tf.function
    def act(self, state):
        if tf.random.uniform((), dtype=tf.float32) <= self.epsilon:
            return tf.random.uniform((), minval=0, maxval=self.action_size, dtype=tf.int32)
        else:
            # Reshape the state to match the expected input shape
            state_reshaped = tf.reshape(state, (1, 1, -1))
            act_values = self.model(state_reshaped)
            return tf.argmax(act_values[0], output_type=tf.int32)
    '''
    def act(self, state):
        logger = get_logger()[0]
        logger.debug(f"Act method: state shape before reshape: {state.shape}")
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.total_actions)
        state = np.reshape(state, (1, 1, self.state_size))  # Reshape to (1, 1, state_size)
        logger.debug(f"Act method: state shape after reshape: {state.shape}")
    
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])
    '''
    def replay(self, batch_size):
        if self.memory_size() < batch_size:
            return
        
        batch_size = min(batch_size, len(self.memory))
        if self.use_ray:
            samples, idxs, is_weights = ray.get(self.memory.sample.remote(batch_size))
        else:
            samples, idxs, is_weights = self.memory.sample(batch_size)
        
        

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

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

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
    '''
    def get_action_distribution(self):
        if not self.action_history:
            return None
        unique, counts = np.unique(self.action_history, return_counts=True)
        return dict(zip(unique, counts))
    '''
    def get_action_distribution(self):
        if not self.action_history:
            return None
        unique, counts = np.unique(self.action_history, return_counts=True)
        return {self.action_map[action]: count for action, count in zip(unique, counts)}
    def reset_history(self):
        self.loss_history = []
        self.action_history = []