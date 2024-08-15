from sklearn.preprocessing import MinMaxScaler
import multiprocessing
import uuid
from logger import setup_logger
from TradingEmulator import TradingEmulator
from custom_ta import calculate_technical_indicators  # Import from custom_ta library
from data_processing import split_data
from utils import report_progress, get_last_trained_timeframe, find_latest_checkpoint, tensor_to_serializable, calculate_market_volatility
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
import ray
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
        
        for t in range(1, len(scaled_data)):
            logger.debug(f"Episode {episode+1}: Starting step {t}")
            action = agent.act(state)
            logger.debug(f"Episode {episode+1}, Step {t}: Action taken: {action}")

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

'''
def run_episode_wrapper(train, agent, episode, initial_balance, batch_size, scaler, return_dict, process_status):
    try:
        process_status[episode] = {'status': 'running', 'start_time': time.time()}
        run_episode(train, agent, episode, initial_balance, batch_size, scaler, return_dict)
        process_status[episode]['status'] = 'completed'
        process_status[episode]['end_time'] = time.time()
    except Exception as e:
        logger.error(f"Error in episode {episode}: {str(e)}")
        process_status[episode]['status'] = 'failed'
        process_status[episode]['error'] = str(e)
        process_status[episode]['end_time'] = time.time()
        return_dict[episode] = ('error', str(e))


def train_agent(data, agent, state_size, action_size, episodes=500, batch_size=1024, initial_balance=100000, patience=50, num_processes=4, max_retrain_attempts=5, performance_threshold=0.8, timeframe="1h"):
    logger = get_logger()[0]
    scaler = MinMaxScaler()
    scaler.fit(data.values)
    
    best_rewards = []
    no_improvement = 0
    best_model_path = None
    retrain_attempts = 0
    
    start_index = 0
    while start_index < len(data) and retrain_attempts < max_retrain_attempts:
        train, test, end_index = split_data(data, start_index, batch_size)

        for i in range(0, episodes, num_processes):
            processes = []
            manager = multiprocessing.Manager()
            return_dict = manager.dict()
            process_status = manager.dict()

            for j in range(num_processes):
                if i + j < episodes:
                    episode = i + j
                    process = multiprocessing.Process(
                        target=run_episode_wrapper, 
                        args=(train, agent, episode, initial_balance, batch_size, scaler, return_dict, process_status)
                    )
                    processes.append(process)
                    process.start()
            
            for process in processes:
                process.join()
            # Log the status of each process
            for episode, status in process_status.items():
                duration = status.get('end_time', time.time()) - status['start_time']
                if status['status'] == 'completed':
                    logger.info(f"Episode {episode} completed in {duration:.2f} seconds")
                elif status['status'] == 'failed':
                    logger.warning(f"Episode {episode} failed after {duration:.2f} seconds. Error: {status.get('error', 'Unknown error')}")
                

            logger.info(f"Episode {i}: All processes completed")
            logger.info(f"Episode {i}: return_dict keys: {list(return_dict.keys())}")

            # Collect and process metrics
            episode_metrics = []
            for episode, result in return_dict.items():
                if result is not None:
                    reward, metrics = result
                    episode_metrics.append(metrics)
                    logger.info(f"Episode {episode}: Processed result with reward {reward}")
                 
        
                else:
                    logger.warning(f"Episode {episode}: No result returned")
        
            logger.info(f"Episode {i}: episode_metrics length: {len(episode_metrics)}")
            
            if not episode_metrics:
                logger.warning(f"Episode {i}: No metrics to process. This might indicate all processes failed.")
                continue  # Skip to the next iteration
            # Calculate average metrics
            if episode_metrics:
                logger.info(f"Episode {i+num_processes}: Metrics keys: {episode_metrics[0].keys()}")
    
            
                avg_metrics = {}
                for key in episode_metrics[0].keys():
                    values = [metrics[key] for metrics in episode_metrics if key in metrics]
                    if isinstance(values[0], (int, float, tf.Tensor)):
                        avg_metrics[key] = tensor_to_serializable(tf.reduce_mean(values))
                    elif isinstance(values[0], list):
                        avg_metrics[key] = tensor_to_serializable([tf.reduce_mean(x) for x in zip(*values)])
                    else:
                        avg_metrics[key] = values[-1]
                
                current_reward = avg_metrics['total_reward']
                
                
                if not best_rewards or current_reward > max(best_rewards):
                    best_rewards.append(current_reward)
                    no_improvement = 0
                    best_model_path = get_model_file_path(f'episode-{i+1}_{str(uuid.uuid4())}')
                    agent.save(best_model_path)
                else:
                    no_improvement += 1
                
               
                # Test on test data
                test_return_dict = {}
                run_episode(test, agent, 0, initial_balance, batch_size, scaler, test_return_dict)
                if 0 in test_return_dict and test_return_dict[0] is not None:
                    test_reward, test_metrics = test_return_dict[0]
                else:
                    test_reward = float('-inf')
           
                report_progress(agent, i+num_processes, timeframe, test_reward, current_reward, max(best_rewards), retrain_attempts, avg_metrics, test_metrics)
                
                # Save checkpoint
                checkpoint_filename = f'checkpoint-episode-{i+num_processes}_{str(uuid.uuid4())}'
                checkpoint_path = get_checkpoint_path(checkpoint_filename)
                agent.save(checkpoint_path)
                logger.info(f"Saved checkpoint at episode {i+num_processes}: {checkpoint_path}")

                # Log more detailed information
                logger.info(f"Episode {i+num_processes}: Epsilon = {agent.epsilon}")
                logger.info(f"Episode {i+num_processes}: Average Loss = {avg_metrics.get('loss', 'N/A')}")
                logger.info(f"Episode {i+num_processes}: Action Distribution = {avg_metrics.get('action_distribution', 'N/A')}")
                logger.info(f"Episode {i+num_processes}: Average Reward = {avg_metrics.get('average_reward', 'N/A')}")
                logger.info(f"Episode {i+num_processes}: All metrics: {avg_metrics}")

            else:
                logger.info(f"Episode {i}: No metrics to process")


            if no_improvement >= patience:
                logger.info(f"Early stopping at episode {i+num_processes}")
                break
            
            if (i + num_processes) % 5 == 0:
                agent.update_target_model()
            
            if i % 100 == 0 and i > 0:
                agent.update_learning_rate(0.95)  # Reduce learning rate by 5% every 100 episodes
        
        if best_model_path is None:
            best_model_path = get_model_file_path(f"final_{str(uuid.uuid4())}")
            agent.save(best_model_path)
        
        # Perform final test after training loop
        final_test_return_dict = {}
        run_episode(test, agent, 0, initial_balance, batch_size, scaler, final_test_return_dict)
        if 0 in final_test_return_dict and final_test_return_dict[0] is not None:
            final_test_reward, final_test_metrics = final_test_return_dict[0]
        else:
            final_test_reward = float('-inf')
            final_test_metrics = {}
        
        logger.info(f"Final test results: Reward: {final_test_reward}, Metrics: {final_test_metrics}")

        # Calculate average of positive best rewards
        positive_best_rewards = [r for r in best_rewards if r > 0]
        avg_best_reward = sum(positive_best_rewards) / len(positive_best_rewards) if positive_best_rewards else 0
        
        if final_test_reward >= avg_best_reward * performance_threshold:
            logger.info(f"Final test performance is good. Moving to next batch.")
            start_index = end_index
            retrain_attempts = 0
        else:
            logger.info(f"Test performance is poor. Retraining attempt {retrain_attempts + 1}/{max_retrain_attempts}")
            agent.learning_rate *= 0.5
            agent.epsilon = min(agent.epsilon * 1.5, 1.0)
            best_rewards = []
            no_improvement = 0
            best_model_path = None
            retrain_attempts += 1
    
    if retrain_attempts >= max_retrain_attempts:
        logger.warning(f"Reached maximum number of retraining attempts ({max_retrain_attempts}). Stopping training.")
    
    return agent, best_model_path

def train_agent_standard(data, agent, state_size, action_size, episodes=500, batch_size=1024, initial_balance=100000, patience=50, num_processes=4, max_retrain_attempts=5, performance_threshold=0.8, timeframe="1h"):
    logger = get_logger()[0]
    scaler = MinMaxScaler()
    scaler.fit(data.values)
    
    best_rewards = []
    no_improvement = 0
    best_model_path = None
    retrain_attempts = 0
    
    start_index = 0
    while start_index < len(data) and retrain_attempts < max_retrain_attempts:
        train, test, end_index = split_data(data, start_index, batch_size)

        for episode in range(episodes):
            return_dict = {}
            run_episode(train, agent, episode, initial_balance, batch_size, scaler, return_dict)
            
            if episode in return_dict and return_dict[episode] is not None:
                reward, metrics = return_dict[episode]
                logger.info(f"Episode {episode}: Processed result with reward {reward}")
                
                current_reward = metrics['total_reward']
                
                if not best_rewards or current_reward > max(best_rewards):
                    best_rewards.append(current_reward)
                    no_improvement = 0
                    best_model_path = get_model_file_path(f'episode-{episode+1}_{str(uuid.uuid4())}')
                    agent.save(best_model_path)
                else:
                    no_improvement += 1
                
                # Test on test data
                test_return_dict = {}
                run_episode(test, agent, 0, initial_balance, batch_size, scaler, test_return_dict)
                if 0 in test_return_dict and test_return_dict[0] is not None:
                    test_reward, test_metrics = test_return_dict[0]
                else:
                    test_reward = float('-inf')
           
                report_progress(agent, episode, timeframe, test_reward, current_reward, max(best_rewards), retrain_attempts, metrics, test_metrics)
                
                # Save checkpoint
                if episode % 10 == 0:  # Save checkpoint every 10 episodes
                    checkpoint_filename = f'checkpoint-episode-{episode}_{str(uuid.uuid4())}'
                    checkpoint_path = get_checkpoint_path(checkpoint_filename)
                    agent.save(checkpoint_path)
                    logger.info(f"Saved checkpoint at episode {episode}: {checkpoint_path}")

                # Log more detailed information
                logger.info(f"Episode {episode}: Epsilon = {agent.epsilon}")
                logger.info(f"Episode {episode}: Average Loss = {metrics.get('loss', 'N/A')}")
                logger.info(f"Episode {episode}: Action Distribution = {metrics.get('action_distribution', 'N/A')}")
                logger.info(f"Episode {episode}: Average Reward = {metrics.get('average_reward', 'N/A')}")
                logger.info(f"Episode {episode}: All metrics: {metrics}")

            else:
                logger.warning(f"Episode {episode}: No valid result returned")

            if no_improvement >= patience:
                logger.info(f"Early stopping at episode {episode}")
                break
            
            if episode % 5 == 0:
                agent.update_target_model()
            
            if episode % 100 == 0 and episode > 0:
                agent.update_learning_rate(0.95)  # Reduce learning rate by 5% every 100 episodes
        
        if best_model_path is None:
            best_model_path = get_model_file_path(f"final_{str(uuid.uuid4())}")
            agent.save(best_model_path)
        
        # Perform final test after training loop
        final_test_return_dict = {}
        run_episode(test, agent, 0, initial_balance, batch_size, scaler, final_test_return_dict)
        if 0 in final_test_return_dict and final_test_return_dict[0] is not None:
            final_test_reward, final_test_metrics = final_test_return_dict[0]
        else:
            final_test_reward = float('-inf')
            final_test_metrics = {}
        
        logger.info(f"Final test results: Reward: {final_test_reward}, Metrics: {final_test_metrics}")

        # Calculate average of positive best rewards
        positive_best_rewards = [r for r in best_rewards if r > 0]
        avg_best_reward = sum(positive_best_rewards) / len(positive_best_rewards) if positive_best_rewards else 0
        
        if final_test_reward >= avg_best_reward * performance_threshold:
            logger.info(f"Final test performance is good. Moving to next batch.")
            start_index = end_index
            retrain_attempts = 0
        else:
            logger.info(f"Test performance is poor. Retraining attempt {retrain_attempts + 1}/{max_retrain_attempts}")
            agent.learning_rate *= 0.5
            agent.epsilon = min(agent.epsilon * 1.5, 1.0)
            best_rewards = []
            no_improvement = 0
            best_model_path = None
            retrain_attempts += 1
    
    if retrain_attempts >= max_retrain_attempts:
        logger.warning(f"Reached maximum number of retraining attempts ({max_retrain_attempts}). Stopping training.")
    
    return agent, best_model_path


def fine_tune_agent(data, agent, best_model_path, state_size, action_size, episodes=10, batch_size=64, initial_balance=100000, num_processes=4, timeframe="1h", performance_threshold=0.8, multiprocessing=False):
    # โหลดโมเดลที่ดีที่สุด
    agent.load(best_model_path)
    
    # ฝึกโมเดลเพิ่มเติม
    if multiprocessing:
        agent, best_model_path = train_agent(data, agent, state_size, action_size, episodes=episodes, batch_size=batch_size, initial_balance=initial_balance, num_processes=num_processes, timeframe=timeframe, performance_threshold=performance_threshold)
    
    else:
        agent, best_model_path = train_agent_standard(data, agent, state_size, action_size, episodes=episodes, batch_size=batch_size, initial_balance=initial_balance, num_processes=num_processes, timeframe=timeframe, performance_threshold=performance_threshold)
    
    return agent, best_model_path

def train_on_timeframes(timeframes, state_size, action_size, performance_threshold=0.8, use_process_num = PROCESS_NUM, multiprocessing=False):
    last_trained_timeframe = get_last_trained_timeframe()
    start_training = False if last_trained_timeframe else True
    logger = get_logger()[0]
    logger.info(f"Training on timeframes with state_size: {state_size}, action_size: {action_size}")
    
    agent = DQNAgent(state_size, action_size)

    for timeframe, df in timeframes:
        if start_training or timeframe == last_trained_timeframe:
            start_training = True
            logger.info(f"Training on {timeframe} timeframe")
 
            df = df.dropna()
            
            # สร้าง agent ใหม่สำหรับแต่ละ timeframe
            
            logger.info(f"Created new DQNAgent for {timeframe} timeframe")
            
            logger.info(f'Total CPU:{PROCESS_NUM}, Using:{use_process_num}')
            latest_checkpoint = find_latest_checkpoint()
            if latest_checkpoint:
                agent, best_model_path = fine_tune_agent(df, agent, latest_checkpoint, state_size, action_size, timeframe=timeframe, performance_threshold=performance_threshold, num_processes=use_process_num, multiprocessing=multiprocessing)
            else:
                if multiprocessing:
                    agent, best_model_path = train_agent(df, agent, state_size, action_size, timeframe=timeframe, performance_threshold=performance_threshold, num_processes=use_process_num)
                else:
                    agent, best_model_path = train_agent_standard(df, agent, state_size, action_size, timeframe=timeframe, performance_threshold=performance_threshold, num_processes=use_process_num)

            logger.info(f"Completed training on {timeframe} timeframe")
            logger.info("----------------------------------------")
        # บันทึกโมเดลสำหรับ timeframe นี้ (ย้ายมาอยู่ในระดับเดียวกับ if)
        timeframe_model_path = get_timeframe_model_file_path(f"{timeframe}-{str(uuid.uuid4())}")
        agent.save(timeframe_model_path)
    
    final_model_path = get_final_model_file_path({str(uuid.uuid4())})
    agent.save(final_model_path)
    logger.info(f"Saved final model for {timeframe} timeframe at {timeframe_model_path}")

'''

def fine_tune_agent(data, agent, best_model_path, state_size, action_size, episodes=10, batch_size=64, initial_balance=100000, num_processes=4, timeframe="1h", performance_threshold=0.8, training_method='standard'):
    # Load the best model
    agent.load(best_model_path)
    
    # Choose the appropriate training function based on the method
    train_func = {
        'standard': train_agent_standard,
        'threading': train_agent_threading,
        'multiprocessing': train_agent_multiprocessing,
        'ray': train_agent_ray,
        'threadpool': train_agent_threadpool,
        'queue': train_agent_queue
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

def train_on_timeframes(timeframes, state_size, action_size, performance_threshold=0.8, use_process_num=PROCESS_NUM, training_method='standard'):
    last_trained_timeframe = get_last_trained_timeframe()
    start_training = False if last_trained_timeframe else True
    logger = get_logger()[0]
    logger.info(f"Training on timeframes with state_size: {state_size}, action_size: {action_size}")
    
    ray_initialized = False
    if training_method == 'ray' and not ray.is_initialized():
        ray.init()
        ray_initialized = True
    
    try:
        agent = DQNAgent(state_size, action_size, use_ray=(training_method == 'ray'))

        # Choose the appropriate training function based on the method
        train_func = {
            'standard': train_agent_standard,
            'threading': train_agent_threading,
            'multiprocessing': train_agent_multiprocessing,
            'ray': train_agent_ray,
            'threadpool': train_agent_threadpool,
            'queue': train_agent_queue
        }.get(training_method, train_agent_standard)

        for timeframe, df in timeframes:
            if start_training or timeframe == last_trained_timeframe:
                start_training = True
                logger.info(f"Training on {timeframe} timeframe")
     
                df = df.dropna()
                
                logger.info(f"Created new DQNAgent for {timeframe} timeframe")
                
                logger.info(f'Total CPU:{PROCESS_NUM}, Using:{use_process_num}')
                latest_checkpoint = find_latest_checkpoint()
                if latest_checkpoint:
                    agent, best_model_path = fine_tune_agent(
                        df, agent, latest_checkpoint, state_size, action_size, 
                        timeframe=timeframe, 
                        performance_threshold=performance_threshold, 
                        num_processes=use_process_num, 
                        training_method=training_method
                    )
                else:
                    agent, best_model_path = train_func(
                        df, agent, state_size, action_size, 
                        timeframe=timeframe, 
                        performance_threshold=performance_threshold, 
                        num_processes=use_process_num
                    )

                logger.info(f"Completed training on {timeframe} timeframe")
                logger.info("----------------------------------------")

            timeframe_model_path = get_timeframe_model_file_path(f"{timeframe}-{str(uuid.uuid4())}")
            agent.save(timeframe_model_path)
        
        final_model_path = get_final_model_file_path(str(uuid.uuid4()))
        agent.save(final_model_path)
        logger.info(f"Saved final model for {timeframe} timeframe at {timeframe_model_path}")

        return agent, final_model_path
    
    finally:
        if ray_initialized:
            ray.shutdown()







def process_episode_results(agent, return_dict, best_rewards, no_improvement, best_model_path, 
                            test_data, initial_balance, batch_size, scaler, episode_base, 
                            num_episodes, patience, performance_threshold, timeframe):
    logger = get_logger()[0]
    episode_metrics = []
    for episode, result in return_dict.items():
        if result is not None:
                    reward, metrics = result
                    episode_metrics.append(metrics)
                    logger.info(f"Episode {episode}: Processed result with reward {reward}")
        else:
            logger.warning(f"Episode {episode}: Invalid result format: {result}")

    if not episode_metrics:
        logger.warning(f"Episode {episode_base}: No valid metrics to process. This might indicate all processes failed.")
        return best_rewards, no_improvement, best_model_path, False

    avg_metrics = calculate_average_metrics(episode_metrics)
    
    if 'total_reward' not in avg_metrics:
        logger.warning(f"Episode {episode_base}: 'total_reward' not found in metrics. Available keys: {avg_metrics.keys()}")
        return best_rewards, no_improvement, best_model_path, False

    current_reward = avg_metrics['total_reward']
    
    if not best_rewards or current_reward > max(best_rewards):
        best_rewards.append(current_reward)
        no_improvement = 0
        best_model_path = get_model_file_path(f'episode-{episode_base+num_episodes}_{str(uuid.uuid4())}')
        agent.save(best_model_path)
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

    return best_rewards, no_improvement, best_model_path, should_stop

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


def finalize_training(agent, best_model_path, test, initial_balance, batch_size, scaler, best_rewards, performance_threshold, start_index, end_index, retrain_attempts, max_retrain_attempts):
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

    positive_best_rewards = [r for r in best_rewards if r > 0]
    avg_best_reward = sum(positive_best_rewards) / len(positive_best_rewards) if positive_best_rewards else 0
    
    if final_test_reward >= avg_best_reward * performance_threshold:
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

    return agent, best_model_path, start_index, retrain_attempts, best_rewards

def base_train_agent(data, agent, state_size, action_size, episodes=500, batch_size=1024, 
                     initial_balance=100000, patience=50, max_retrain_attempts=5, 
                     performance_threshold=0.8, timeframe="1h", run_episodes_func=None, num_processes=PROCESS_NUM):
    logger = get_logger()[0]
    scaler = MinMaxScaler()
    scaler.fit(data.values)
    
    best_rewards = []
    no_improvement = 0
    best_model_path = None
    retrain_attempts = 0
    
    start_index = 0
    while start_index < len(data) and retrain_attempts < max_retrain_attempts:
        train, test, end_index = split_data(data, start_index, batch_size)

        for i in range(0, episodes, num_processes):
            return_dict = run_episodes_func(train, agent, i, num_processes, initial_balance, batch_size, scaler)

            best_rewards, no_improvement, best_model_path, should_stop = process_episode_results(
                agent, return_dict, best_rewards, no_improvement, best_model_path, 
                test, initial_balance, batch_size, scaler, i, num_processes, 
                patience, performance_threshold, timeframe
            )

            if should_stop:
                break

            if (i + num_processes) % 5 == 0:
                agent.update_target_model()
            
            if i % 100 == 0 and i > 0:
                agent.update_learning_rate(0.95)
        
        agent, best_model_path, start_index, retrain_attempts, best_rewards = finalize_training(
            agent, best_model_path, test, initial_balance, batch_size, scaler,
            best_rewards, performance_threshold, start_index, end_index,
            retrain_attempts, max_retrain_attempts
        )
    
    if retrain_attempts >= max_retrain_attempts:
        logger.warning(f"Reached maximum number of retraining attempts ({max_retrain_attempts}). Stopping training.")
    
    return agent, best_model_path

def run_episodes_standard(train, agent, start_episode, num_episodes, initial_balance, batch_size, scaler):
    return_dict = {}
    for i in range(num_episodes):
        episode = start_episode + i
        run_episode(train, agent, episode, initial_balance, batch_size, scaler, return_dict)
    return return_dict

def run_episodes_threading(train, agent, start_episode, num_episodes, initial_balance, batch_size, scaler):
    return_dict = {}
    threads = []
    for i in range(num_episodes):
        episode = start_episode + i
        thread = threading.Thread(
            target=run_episode,
            args=(train, agent, episode, initial_balance, batch_size, scaler, return_dict)
        )
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()
    
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

@ray.remote
def run_episode_ray(train, agent, episode, initial_balance, batch_size, scaler):
    return_dict = {}
    try:
        run_episode(train, agent, episode, initial_balance, batch_size, scaler, return_dict)
    except Exception as e:
        return_dict[episode] = ('error', str(e))
    return return_dict

def run_episodes_ray(train, agent, start_episode, num_episodes, initial_balance, batch_size, scaler):
    futures = [run_episode_ray.remote(train, agent, start_episode + i, initial_balance, batch_size, scaler) 
               for i in range(num_episodes)]
    results = ray.get(futures)
    return_dict = {}
    for result in results:
        return_dict.update(result)
    return return_dict

def run_episodes_threadpool(train, agent, start_episode, num_episodes, initial_balance, batch_size, scaler):
    return_dict = {}
    with ThreadPoolExecutor(max_workers=num_episodes) as executor:
        future_to_episode = {executor.submit(run_episode, train, agent, start_episode + i, initial_balance, batch_size, scaler, return_dict): i 
                             for i in range(num_episodes)}
        for future in as_completed(future_to_episode):
            future.result()  # This will raise an exception if the task failed
    return return_dict

def run_episode_queue(data, agent, episode, initial_balance, batch_size, scaler, queue):
    return_dict = {}
    run_episode(data, agent, episode, initial_balance, batch_size, scaler, return_dict)
    queue.put((episode, return_dict.get(episode)))

def run_episodes_queue(train, agent, start_episode, num_episodes, initial_balance, batch_size, scaler):
    queue = Queue()
    processes = []
    for i in range(num_episodes):
        episode = start_episode + i
        process = Process(
            target=run_episode_queue,
            args=(train, agent, episode, initial_balance, batch_size, scaler, queue)
        )
        processes.append(process)
        process.start()
    
    for process in processes:
        process.join()
    
    return_dict = {}
    while not queue.empty():
        episode, result = queue.get()
        if result is not None:
            return_dict[episode] = result
    
    return return_dict


def train_agent_standard(*args, **kwargs):
    return base_train_agent(*args, **kwargs, run_episodes_func=run_episodes_standard)

def train_agent_threading(*args, **kwargs):
    return base_train_agent(*args, **kwargs, run_episodes_func=run_episodes_threading)

def train_agent_multiprocessing(*args, **kwargs):
    return base_train_agent(*args, **kwargs, run_episodes_func=run_episodes_multiprocessing)

def train_agent_ray(*args, **kwargs):
    return base_train_agent(*args, **kwargs, run_episodes_func=run_episodes_ray)

def train_agent_threadpool(*args, **kwargs):
    return base_train_agent(*args, **kwargs, run_episodes_func=run_episodes_threadpool)


def train_agent_queue(*args, **kwargs):
    return base_train_agent(*args, **kwargs, run_episodes_func=run_episodes_queue)