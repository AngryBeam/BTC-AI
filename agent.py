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
from prioritized_replay_buffer import PrioritizedReplayBuffer
import os
import json
from tensorflow.keras.models import save_model, load_model

logger, _ = get_logger()
class DQNAgent:
    def __init__(self, state_size, action_size, max_positions=10):
        logger = get_logger()[0]
        logger.info(f"Initializing DQNAgent with state_size: {state_size}, action_size: {action_size}")
        
        self.state_size = state_size
        self.action_size = action_size
        self.max_positions = max_positions
        self.total_actions = 1 + 2 + 2 * max_positions
        #self.memory = deque(maxlen=10000)
        self.memory = PrioritizedReplayBuffer(capacity=10000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.optimizer = Adam(learning_rate=self.learning_rate)
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
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

    def remember(self, state, action, reward, next_state, done):
        # Ensure state and next_state are 2D
        #Comment zone is stable version using with self.memory = deque(maxlen=10000)
        #state = np.reshape(state, (1, self.state_size))
        #next_state = np.reshape(next_state, (1, self.state_size))
        #self.memory.append((state, action, reward, next_state, done))
        # Ensure state and next_state are 2D
        experience = (state, action, reward, next_state, done)
        state = np.reshape(state, (1, 1, self.state_size))
        next_state = np.reshape(next_state, (1, 1, self.state_size))
        target = reward + self.gamma * np.max(self.target_model.predict(next_state)[0]) * (1 - done)
        current_q = self.model.predict(state)[0][action]
        error = abs(target - current_q)
        self.memory.add(error, experience)


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
        if len(self.memory) < batch_size:
            return

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
            targets_full[i, action] = targets[i]

        loss = self.model.train_on_batch(states, targets_full, sample_weight=is_weights)

        # Update priorities
        errors = np.abs(targets - targets_full[np.arange(batch_size), actions])
        for i in range(batch_size):
            idx = idxs[i]
            self.memory.update(idx, errors[i])

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss

    
    '''
    def replay(self, batch_size):
        logger = get_logger()[0]
        minibatch = random.sample(self.memory, batch_size)
        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])

        logger.debug(f"Replay method: states shape before reshape: {states.shape}")
        
        states = np.reshape(states, (batch_size, 1, self.state_size))
        next_states = np.reshape(next_states, (batch_size, 1, self.state_size))
        
        logger.debug(f"Replay method: states shape after reshape: {states.shape}")
        logger.debug(f"Replay method: next_states shape after reshape: {next_states.shape}")

        targets = rewards + self.gamma * (np.amax(self.target_model.predict(next_states), axis=1)) * (1 - dones)
        logger.debug(f"Replay method: targets shape: {targets.shape}")
        targets_full = self.model.predict(states)
        
        logger.debug(f"Replay method: targets_full shape: {targets_full.shape}")
        
        for i, action in enumerate(actions):
            targets_full[i, action] = targets[i]

        history = self.model.fit(states, targets_full, epochs=1, verbose=0)
        loss = history.history['loss'][0]
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return loss

    
    
    

    def load(self, name):
        self.model.load_weights(name)
    
    def save(self, name):
        self.model.save_weights(name)
    
    def save(self, name):
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(name), exist_ok=True)
            
            # Save the model architecture and weights
            self.model.save(name)
            
            # Save the optimizer configuration
            optimizer_config = self.optimizer.get_config()
            with open(f"{name}_optimizer.json", 'w') as f:
                json.dump(optimizer_config, f)
            
            logger.info(f"Successfully saved model and optimizer to {name}")
        except Exception as e:
            logger.error(f"Error saving model to {name}: {str(e)}")
            raise

    def load(self, name):
        try:
            # Load the model architecture and weights
            self.model = tf.keras.models.load_model(name, compile=False)
            self.target_model = tf.keras.models.load_model(name, compile=False)
            
            # Try to load the optimizer configuration
            optimizer_config_path = f"{name}_optimizer.json"
            if os.path.exists(optimizer_config_path):
                with open(optimizer_config_path, 'r') as f:
                    optimizer_config = json.load(f)
                self.optimizer = Adam.from_config(optimizer_config)
                logger.info(f"Successfully loaded optimizer configuration from {optimizer_config_path}")
            else:
                logger.warning(f"Optimizer configuration file not found at {optimizer_config_path}. Using default optimizer.")
                self.optimizer = Adam(learning_rate=self.learning_rate)
            
            # Recompile the models with the loaded or new optimizer
            self.model.compile(loss='mse', optimizer=self.optimizer)
            self.target_model.compile(loss='mse', optimizer=self.optimizer)
            
            logger.info(f"Successfully loaded model from {name}")
        except Exception as e:
            logger.error(f"Error loading model from {name}: {str(e)}")
            raise
    '''
    def save(self, name):
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(name), exist_ok=True)
            
            # Save the model in .keras format
            save_model(self.model, f"{name}.keras", save_format="keras")
            
            logger.info(f"Successfully saved model to {name}.keras")
        except Exception as e:
            logger.error(f"Error saving model to {name}.keras: {str(e)}")
            raise

    def load(self, name):
        try:
            # Load the model in .keras format
            loaded_model = load_model(f"{name}.keras", compile=False)
            
            # Copy weights to our models
            self.model.set_weights(loaded_model.get_weights())
            self.target_model.set_weights(loaded_model.get_weights())
            
            # Recompile the models
            self.model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
            self.target_model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
            
            logger.info(f"Successfully loaded model from {name}.keras")
        except Exception as e:
            logger.error(f"Error loading model from {name}.keras: {str(e)}")
            raise


    def update_learning_rate(self, factor):
        self.learning_rate *= factor
        self.model.optimizer.learning_rate.assign(self.learning_rate)
