from sklearn.preprocessing import MinMaxScaler
import multiprocessing
import uuid
from logger import setup_logger
from TradingEmulator import TradingEmulator
from custom_ta import calculate_technical_indicators  # Import from custom_ta library
from data_processing import split_data
from utils import report_progress, get_last_trained_timeframe, find_latest_checkpoint
from agent import DQNAgent
from logger import get_logger
from config import get_model_file_path, get_final_model_file_path,get_timeframe_model_file_path, PROCESS_NUM, get_checkpoint_path
import traceback
import numpy as np

logger, _ = get_logger()


def run_episode(data, agent, episode, initial_balance, batch_size, scaler, return_dict):
    logger = get_logger()[0]
    try:
        emulator = TradingEmulator(initial_balance=initial_balance)
        previous_balance = initial_balance  # Initialize previous_balance
    
        scaled_data = scaler.transform(data.values)
        logger.info(f"Episode {episode+1}: scaled_data shape: {scaled_data.shape}")
        logger.info(f"Episode {episode+1}: agent.state_size: {agent.state_size}")
        
        if len(scaled_data) < 1:
            logger.error(f"Episode {episode+1}: Not enough data points. scaled_data length: {len(scaled_data)}")
            return_dict[episode] = None
            return
        
        state = scaled_data[0].reshape(1, agent.state_size)  # Reshape to (1, state_size)
        logger.info(f"Episode {episode+1}: Initial state shape: {state.shape}")
        
        total_reward = 0
        
        for t in range(1, len(scaled_data)):
            action = agent.act(state)
            logger.debug(f"Episode {episode+1}, Step {t}: Action taken: {action}")
            
            # Execute action
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
            
            next_state = scaled_data[t].reshape(1, agent.state_size)  # Reshape to (1, state_size)
            #reward = emulator.balance - initial_balance
            reward = emulator.calculate_reward(initial_balance, previous_balance, action)
            previous_balance = emulator.balance
            done = t == len(scaled_data) - 1
            
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            emulator.update(data.iloc[t]['Close'])
            emulator.trailing_stop_loss(data.iloc[t]['Close'])
            
            if len(agent.memory) > batch_size:
                loss = agent.replay(batch_size)
                logger.debug(f"Episode {episode+1}, Step {t}: Replay loss: {loss}")
        
        if hasattr(emulator, 'get_performance_metrics') and callable(emulator.get_performance_metrics):
            metrics = emulator.get_performance_metrics()
        else:
            metrics = {}
        metrics['total_reward'] = total_reward
        
        return_dict[episode] = (total_reward, metrics)
        logger.info(f"Episode {episode+1} completed successfully. Total reward: {total_reward}")
    except Exception as e:
        logger.error(f"Error in episode {episode+1}: {e}")
        logger.debug(f"Memory size before replay: {len(agent.memory)}")
        logger.error(traceback.format_exc())
        return_dict[episode] = None


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
            
            for j in range(num_processes):
                if i + j < episodes:
                    process = multiprocessing.Process(target=run_episode, args=(train, agent, i+j, initial_balance, batch_size, scaler, return_dict))
                    processes.append(process)
                    process.start()
            
            for process in processes:
                process.join()
            
            # Collect and process metrics
            episode_metrics = []
            for episode, result in return_dict.items():
                if result is not None:
                    reward, metrics = result
                    episode_metrics.append(metrics)
            
            # Calculate average metrics
            if episode_metrics:
                avg_metrics = {}
                for key in episode_metrics[0].keys():
                    values = [metrics[key] for metrics in episode_metrics if key in metrics]
                    if isinstance(values[0], (int, float)):
                        avg_metrics[key] = sum(values) / len(values)
                    elif isinstance(values[0], list):
                        avg_metrics[key] = [sum(x) / len(x) for x in zip(*values)]
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
                    test_reward, _ = test_return_dict[0]
                else:
                    test_reward = float('-inf')
                
                report_progress(agent, i+num_processes, timeframe, test_reward, current_reward, max(best_rewards), retrain_attempts, avg_metrics)
                
                # Save checkpoint
                checkpoint_filename = f'checkpoint-episode-{i+num_processes}_{str(uuid.uuid4())}'
                checkpoint_path = get_checkpoint_path(checkpoint_filename)
                agent.save(checkpoint_path)
                logger.info(f"Saved checkpoint at episode {i+num_processes}: {checkpoint_path}")

                # Log more detailed information
                logger.info(f"Episode {i+num_processes}: Epsilon = {agent.epsilon}")
                logger.info(f"Episode {i+num_processes}: Average Loss = {avg_metrics.get('loss', 'N/A')}")
                logger.info(f"Episode {i+num_processes}: Action Distribution = {avg_metrics.get('action_distribution', 'N/A')}")
                logger.info(f"Episode {i+num_processes}: Average Reward = {avg_metrics.get('reward', 'N/A')}")

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


def fine_tune_agent(data, agent, best_model_path, state_size, action_size, episodes=10, batch_size=64, initial_balance=100000, num_processes=4, timeframe="1h", performance_threshold=0.8):
    # โหลดโมเดลที่ดีที่สุด
    agent.load(best_model_path)
    
    # ฝึกโมเดลเพิ่มเติม
    agent, best_model_path = train_agent(data, agent, state_size, action_size, episodes=episodes, batch_size=batch_size, initial_balance=initial_balance, num_processes=num_processes, timeframe=timeframe, performance_threshold=performance_threshold)
    
    return agent, best_model_path

def train_on_timeframes(timeframes, state_size, action_size, performance_threshold=0.8):
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
            use_process_num = 8
            logger.info(f'Total CPU:{PROCESS_NUM}, Using:{use_process_num}')
            latest_checkpoint = find_latest_checkpoint()
            if latest_checkpoint:
                agent, best_model_path = fine_tune_agent(df, agent, latest_checkpoint, state_size, action_size, timeframe=timeframe, performance_threshold=performance_threshold, num_processes=use_process_num)
            else:
                agent, best_model_path = train_agent(df, agent, state_size, action_size, timeframe=timeframe, performance_threshold=performance_threshold, num_processes=use_process_num)

            logger.info(f"Completed training on {timeframe} timeframe")
            logger.info("----------------------------------------")
        # บันทึกโมเดลสำหรับ timeframe นี้ (ย้ายมาอยู่ในระดับเดียวกับ if)
        timeframe_model_path = get_timeframe_model_file_path(f"{timeframe}-{str(uuid.uuid4())}")
        agent.save(timeframe_model_path)
    
    final_model_path = get_final_model_file_path({str(uuid.uuid4())})
    agent.save(final_model_path)
    logger.info(f"Saved final model for {timeframe} timeframe at {timeframe_model_path}")
