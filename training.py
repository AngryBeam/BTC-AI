from sklearn.preprocessing import MinMaxScaler
import multiprocessing
import uuid
from logger import setup_logger
from TradingEmulator import TradingEmulator
from custom_ta import calculate_technical_indicators  # Import from custom_ta library
from data_processing import split_data
from utils import report_progress, get_last_trained_timeframe, find_latest_checkpoint, scaler_info, calculate_market_volatility
from agent import DQNAgent
from logger import get_logger
from config import get_model_file_path, get_final_model_file_path,get_timeframe_model_file_path, PROCESS_NUM, get_checkpoint_path
import traceback
import numpy as np
import tensorflow as tf
from multiprocessing import Value
import time
import threading
from multiprocessing import Process, Queue
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

logger, _ = get_logger()


def run_episode(data, agent, episode, initial_balance, batch_size, scaler, return_dict):
    logger = get_logger()[0]
    try:
        logger.info(f"Starting episode {episode+1}")
        emulator = TradingEmulator(initial_balance=initial_balance)
        previous_balance = initial_balance

        scaled_data = scaler.transform(data.values)
        logger.info(f"Episode {episode+1}: scaled_data shape: {scaled_data.shape}")
        logger.info(f"Episode {episode+1}: agent.state_size: {agent.state_size}")
        
        if len(scaled_data) < 1:
            logger.error(f"Episode {episode+1}: Not enough data points. scaled_data length: {len(scaled_data)}")
            return_dict[episode] = ('error', 'Not enough data points')
            return
        
        state = scaled_data[0].reshape(1, agent.state_size)
        logger.info(f"Episode {episode+1}: Initial state shape: {state.shape}")
        
        total_reward = 0
        mcts_used = 0  # เพิ่มตัวแปรนี้เพื่อนับจำนวนครั้งที่ใช้ MCTS
        
        for t in range(1, len(scaled_data)):
            logger.info(f"Episode {episode+1}: Starting step {t}")
            action = agent.act(state)
            if agent.use_mcts and not agent.is_exploring(state):  # เพิ่มเงื่อนไขนี้
                mcts_used += 1
            logger.info(f"Episode {episode+1}, Step {t}: Action taken: {action}")

            current_volatility = calculate_market_volatility(data.iloc[:t+1])


            current_price = data.iloc[t]['Close']
            previous_price = data.iloc[t-1]['Close']
            price_change = (current_price - previous_price) / previous_price
            emulator.take_action(action, 1, current_price, price_change, current_volatility)

            next_state = scaled_data[t].reshape(1, agent.state_size)

            
            reward = emulator.calculate_reward(initial_balance, previous_balance, action)
            
            
            previous_balance = emulator.balance
            done = t == len(scaled_data) - 1
            
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            emulator.update(data.iloc[t]['Close'])
            emulator.trailing_stop_loss(data.iloc[t]['Close'])
            
            if agent.memory_size() > batch_size:
                loss = agent.replay(batch_size)
                logger.debug(f"Episode {episode+1}, Step {t}: Replay loss: {loss}")
            
            logger.debug(f"Episode {episode+1}, Step {t}: Current total reward: {total_reward}")

        logger.info(f"Episode {episode+1} completed successfully. Total reward: {total_reward}")
        
        metrics = emulator.get_performance_metrics() if hasattr(emulator, 'get_performance_metrics') else {}
        metrics['total_reward'] = total_reward
        metrics['average_reward'] = total_reward / len(scaled_data)
        metrics['loss'] = np.mean(agent.loss_history) if agent.loss_history else None
        metrics['mcts_used'] = mcts_used  # เพิ่มข้อมูล MCTS usage ในเมตริกส์
        
        action_counts = np.bincount(agent.action_history, minlength=agent.action_size)
        metrics['action_distribution'] = action_counts.tolist()
        
        return_dict[episode] = (total_reward, metrics)
        logger.info(f"Successfully added result to return_dict for episode {episode+1}")
    except Exception as e:
        logger.error(f"Error in episode {episode+1}: {str(e)}")
        logger.error(traceback.format_exc())
        return_dict[episode] = ('error', str(e))
    finally:
        logger.info(f"Episode {episode+1} finished (run_episode)")



def fine_tune_agent(data, agent, best_model_path, state_size, action_size, episodes=500, batch_size=1024, initial_balance=100000, num_processes=PROCESS_NUM, timeframe="1h", performance_threshold=0.8, training_method='standard'):
    # Load the best model
    agent.load(best_model_path)
    
    # Choose the appropriate training function based on the method
    train_func = {
        'standard': train_agent_standard,
        'multiprocessing': train_agent_multiprocessing
    }.get(training_method, train_agent_standard)
    
    # Fine-tune the model
    agent, best_model_path = train_func(
        data, agent, state_size, action_size, 
        episodes=episodes, 
        batch_size=batch_size, 
        initial_balance=initial_balance, 
        num_processes=num_processes, 
        timeframe=timeframe, 
        performance_threshold=performance_threshold
    )
    
    return agent, best_model_path

def train_on_timeframes(timeframes, state_size, action_size, performance_threshold=0.8, episodes=500, batch_size=1024, use_process_num=PROCESS_NUM, training_method='standard', mcts=False):
    last_trained_timeframe = get_last_trained_timeframe()
    start_training = False if last_trained_timeframe else True
    logger = get_logger()[0]
    logger.info(f"Training on timeframes with state_size: {state_size}, action_size: {action_size}")

    latest_checkpoint = find_latest_checkpoint()
    agent = DQNAgent(state_size, action_size, use_mcts=mcts)
    if latest_checkpoint:
        logger.info(f"Latest checkpoint found: {latest_checkpoint}")
        # ถามผู้ใช้ว่าต้องการเทรนต่อหรือไม่
        user_input = input("Do you want to continue training from this checkpoint? (Y/n): ").strip().lower()
        
        if user_input == '' or user_input == 'y':
            logger.info("Continuing training from the latest checkpoint.")
            agent.load(latest_checkpoint)

    logger.info(f'Total CPU:{PROCESS_NUM}, Using:{use_process_num}')
    first_run = True
    process_from_check_point_yet = False
    if training_method=='standard':
        use_process_num=1
    try:


        # Choose the appropriate training function based on the method
        train_func = {
            'standard': train_agent_standard,
            'multiprocessing': train_agent_multiprocessing
        }.get(training_method, train_agent_standard)

        for timeframe, df in timeframes:
      
            if start_training or timeframe == last_trained_timeframe:
                start_training = True
                logger.info(f"Training on {timeframe} timeframe")
     
                df = df.dropna()
                
                logger.info("Starting new training session.")
                agent, best_model_path = train_func(
                    df, agent, state_size, action_size, 
                    timeframe=timeframe, 
                    performance_threshold=performance_threshold, 
                    episodes=episodes, batch_size=batch_size,
                    num_processes=use_process_num
                )
                
                logger.info(f"Completed training on {timeframe} timeframe")
                logger.info("----------------------------------------")
            first_run = False
            timeframe_model_path = get_timeframe_model_file_path(f"{timeframe}-{str(uuid.uuid4())}")
            agent.save(timeframe_model_path)
        
        final_model_path = get_final_model_file_path(str(uuid.uuid4()))
        agent.save(final_model_path)
        logger.info(f"Saved final model for {timeframe} timeframe at {timeframe_model_path}")

        return agent, final_model_path
    
    finally:
        pass







def process_episode_results(agent, return_dict, best_rewards, no_improvement, best_model_path, 
                            test_data, initial_balance, batch_size, scaler, episode_base, 
                            num_episodes, patience, performance_threshold, timeframe, model_acceptable_percentage=5):
    logger = get_logger()[0]
    episode_metrics = []
    model_acceptable = False

    for episode, result in return_dict.items():
        if result is not None:
                    reward, metrics = result
                    episode_metrics.append(metrics)
                    logger.info(f"Episode {episode}: Processed result with reward {reward}")
        else:
            logger.warning(f"Episode {episode}: Invalid result format: {result}")

    if not episode_metrics:
        logger.warning(f"Episode {episode_base}: No valid metrics to process. This might indicate all processes failed.")
        return best_rewards, no_improvement, best_model_path, False, False

    avg_metrics = calculate_average_metrics(episode_metrics)
    
    if 'total_reward' not in avg_metrics:
        logger.warning(f"Episode {episode_base}: 'total_reward' not found in metrics. Available keys: {avg_metrics.keys()}, episode_metrics:{episode_metrics}")
        return best_rewards, no_improvement, best_model_path, False, False

    current_reward = avg_metrics['total_reward']
    
    if not best_rewards or current_reward > max(best_rewards):
        best_rewards.append(current_reward)
        no_improvement = 0
        #best_model_path = get_model_file_path(f'episode-{episode_base+num_episodes}_{str(uuid.uuid4())}')
        #agent.save(best_model_path)
    else:
        no_improvement += 1
    
    # Test on test data
    test_return_dict = {}
    run_episode(test_data, agent, 0, initial_balance, batch_size, scaler, test_return_dict)
    if 0 in test_return_dict and test_return_dict[0] is not None:
        test_reward, test_metrics = test_return_dict[0]
    else:
        test_reward = float('-inf')

    report_progress(agent, episode_base+num_episodes, timeframe, test_reward, current_reward, max(best_rewards), 0, avg_metrics, test_metrics)
    
    # Update epsilon based on test metrics
    if 'total_pnl' in test_metrics:
        new_epsilon = agent.update_epsilon(test_metrics['total_pnl'])
        logger.info(f"Updated epsilon to {new_epsilon} based on test PnL of {test_metrics['total_pnl']}")
        if (((test_metrics['total_pnl']*100)/initial_balance)>=model_acceptable_percentage):
            model_acceptable = True
            # Save checkpoint
            checkpoint_filename = f'checkpoint-episode-{episode_base+num_episodes}_{str(uuid.uuid4())}'
            checkpoint_path = get_checkpoint_path(checkpoint_filename)
            agent.save(checkpoint_path)
            logger.info(f"Saved checkpoint at episode {episode_base+num_episodes}: {checkpoint_path}")

    # Log more detailed information
    logger.info(f"Episode {episode_base+num_episodes}: Epsilon = {agent.epsilon}")
    logger.info(f"Episode {episode_base+num_episodes}: Average Loss = {avg_metrics.get('loss', 'N/A')}")
    logger.info(f"Episode {episode_base+num_episodes}: Action Distribution = {avg_metrics.get('action_distribution', 'N/A')}")
    logger.info(f"Episode {episode_base+num_episodes}: Average Reward = {avg_metrics.get('average_reward', 'N/A')}")
    logger.info(f"Episode {episode_base+num_episodes}: All metrics: {avg_metrics}")

    should_stop = no_improvement >= patience
    if should_stop:
        logger.info(f"Early stopping at episode {episode_base+num_episodes}")

    return best_rewards, no_improvement, best_model_path, should_stop, model_acceptable

def calculate_average_metrics(episode_metrics):
    if not episode_metrics or not isinstance(episode_metrics[0], dict):
        return {}
    
    avg_metrics = {}
    for key in episode_metrics[0].keys():
        values = [metrics[key] for metrics in episode_metrics if isinstance(metrics, dict) and key in metrics]
        if values:
            if isinstance(values[0], (int, float, np.number)):
                avg_metrics[key] = float(np.mean(values))  # Ensure it's a Python float
            elif isinstance(values[0], list):
                avg_metrics[key] = np.mean(values, axis=0).tolist()
            else:
                avg_metrics[key] = values[-1]
    return avg_metrics


def finalize_training(agent, best_model_path, test, initial_balance, batch_size, scaler, best_rewards, performance_threshold, start_index, end_index, retrain_attempts, max_retrain_attempts, model_acceptable_percentage=5):
    logger = get_logger()[0]
    
    if best_model_path is None:
        best_model_path = get_model_file_path(f"final_{str(uuid.uuid4())}")
        agent.save(best_model_path)
    
    final_test_return_dict = {}
    run_episode(test, agent, 0, initial_balance, batch_size, scaler, final_test_return_dict)
    if 0 in final_test_return_dict and final_test_return_dict[0] is not None:
        final_test_reward, final_test_metrics = final_test_return_dict[0]
    else:
        final_test_reward = float('-inf')
    
    logger.info(f"Final test results: Reward: {final_test_reward}, Metrics: {final_test_metrics}")
    
    model_acceptable = False
    # Update epsilon based on test metrics
    if 'total_pnl' in final_test_metrics:
        if (((final_test_metrics['total_pnl']*100)/initial_balance)>=model_acceptable_percentage):
            logger.info(f"Final test performance is good. Moving to next batch.")
            model_acceptable = True
            start_index = end_index
            retrain_attempts = 0
    if not model_acceptable:
        logger.info(f"Test performance is poor. Retraining attempt {retrain_attempts + 1}/{max_retrain_attempts}")
        agent.learning_rate *= 0.5
        agent.epsilon = min(agent.epsilon * 1.5, 1.0)
        best_rewards = []
        best_model_path = None
        retrain_attempts += 1

    '''
    positive_best_rewards = [r for r in best_rewards if r > 0]
    avg_best_reward = sum(positive_best_rewards) / len(positive_best_rewards) if positive_best_rewards else 0

    if not isinstance(final_test_reward,float):
        logger.info(f'final_test_reward is not float: {final_test_reward}, type:{type(final_test_reward)}')
        final_test_reward = float(final_test_reward)
    if not isinstance('avg_best_reward', float):
        logger.info(f'avg_best_reward is not float: {avg_best_reward}, type:{type(avg_best_reward)}')
        avg_best_reward = float(avg_best_reward)

    if final_test_reward >= float(avg_best_reward * performance_threshold):
        logger.info(f"Final test performance is good. Moving to next batch.")
        start_index = end_index
        retrain_attempts = 0
    else:
        logger.info(f"Test performance is poor. Retraining attempt {retrain_attempts + 1}/{max_retrain_attempts}")
        agent.learning_rate *= 0.5
        agent.epsilon = min(agent.epsilon * 1.5, 1.0)
        best_rewards = []
        best_model_path = None
        retrain_attempts += 1
    '''
    return agent, best_model_path, start_index, retrain_attempts, best_rewards, model_acceptable

def base_train_agent(data, agent, state_size, action_size, episodes=500, batch_size=1024, 
                     initial_balance=100000, patience=5, max_retrain_attempts=5, 
                     performance_threshold=0.8, timeframe="1h", run_episodes_func=None, num_processes=PROCESS_NUM):
    logger = get_logger()[0]
    scaler = MinMaxScaler()
    scaler.fit(data.values)
    scaler_info(scaler, data)
    best_rewards = []
    best_model_path = None
    retrain_attempts = 0
    
    start_index = 0
    previous_test_data = None

    while start_index < len(data):
        
        no_improvement = 0
        should_stop = False
        if retrain_attempts==0:
            train, test, end_index = split_data(data, start_index, batch_size)

            if previous_test_data is not None:
                train = pd.concat([previous_test_data, train])

        for i in range(0, episodes, num_processes):
            return_dict = run_episodes_func(train, agent, i, num_processes, initial_balance, batch_size, scaler)
            model_acceptable = False
            best_rewards, no_improvement, best_model_path, should_stop, model_acceptable = process_episode_results(
                agent, return_dict, best_rewards, no_improvement, best_model_path, 
                test, initial_balance, batch_size, scaler, i, num_processes, 
                patience, performance_threshold, timeframe, model_acceptable_percentage=5
            )
            '''
            If model has PNL over 5% it acceptable and it should go on
            '''
            if model_acceptable:
                break

            if should_stop:
                break

            if (i + num_processes) % 5 == 0:
                agent.update_target_model()
            
            if i % 100 == 0 and i > 0:
                agent.update_learning_rate(0.95)
        
        agent, best_model_path, start_index, retrain_attempts, best_rewards, model_acceptable = finalize_training(
            agent, best_model_path, test, initial_balance, batch_size, scaler,
            best_rewards, performance_threshold, start_index, end_index,
            retrain_attempts, max_retrain_attempts, model_acceptable_percentage=6.18
        )
        previous_test_data = test
        if retrain_attempts >= max_retrain_attempts:
            logger.warning(f"Reached maximum number of retraining attempts ({max_retrain_attempts}). Stopping training.")
            retrain_attempts = 0
            start_index = end_index
            agent.learning_rate *= 0.001
            agent.epsilon = 1.0

        
    return agent, best_model_path

def run_episodes_standard(train, agent, start_episode, num_episodes, initial_balance, batch_size, scaler):
    return_dict = {}
    for i in range(num_episodes):
        episode = start_episode + i
        run_episode(train, agent, episode, initial_balance, batch_size, scaler, return_dict)
    return return_dict


def run_episodes_multiprocessing(train, agent, start_episode, num_episodes, initial_balance, batch_size, scaler):
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    processes = []
    for i in range(num_episodes):
        episode = start_episode + i
        process = Process(
            target=run_episode,
            args=(train, agent, episode, initial_balance, batch_size, scaler, return_dict)
        )
        processes.append(process)
        process.start()
    
    for process in processes:
        process.join()
    
    return dict(return_dict)




def train_agent_standard(*args, **kwargs):
    return base_train_agent(*args, **kwargs, run_episodes_func=run_episodes_standard)


def train_agent_multiprocessing(*args, **kwargs):
    return base_train_agent(*args, **kwargs, run_episodes_func=run_episodes_multiprocessing)


