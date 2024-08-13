import numpy as np
from TradingEmulator import TradingEmulator

class TradingEnvironment:
    def __init__(self, data, initial_balance=100000, max_positions=5):
        self.data = data
        self.emulator = TradingEmulator(initial_balance=initial_balance)
        self.current_step = 0
        self.max_positions = max_positions
        self.action_size = 5 + 2 * max_positions  # Hold Both, Hold Long, Hold Short, Open Long, Open Short, Close Long (1-5), Close Short (1-5)

    def reset(self):
        self.emulator = TradingEmulator(initial_balance=self.emulator.initial_balance)
        self.current_step = 0
        return self._get_state()

    def step(self, action):
        # Execute the action
        self._execute_action(action)

        # Move to the next time step
        self.current_step += 1

        # Get the new state
        new_state = self._get_state()

        # Calculate the reward
        reward = self.emulator.calculate_reward(self.emulator.initial_balance, self.emulator.balance_history[-2], action)

        # Check if the episode is done
        done = self.current_step >= len(self.data) - 1

        # Get additional info
        info = self.emulator.get_performance_metrics()

        return new_state, reward, done, info

    def _get_state(self):
        # Return the current state (market data + agent's position)
        return np.concatenate([
            self.data.iloc[self.current_step].values,
            [self.emulator.position['long'], self.emulator.position['short']]
        ])

    def _execute_action(self, action):
        current_price = self.data.iloc[self.current_step]['Close']
        if action == 0:  # Hold Both
            pass
        elif action == 1:  # Hold Long (close short if exists)
            if self.emulator.position['short'] > 0:
                self.emulator.close_position('short', self.emulator.position['short'], current_price)
        elif action == 2:  # Hold Short (close long if exists)
            if self.emulator.position['long'] > 0:
                self.emulator.close_position('long', self.emulator.position['long'], current_price)
        elif action == 3:  # Open Long
            self.emulator.open_position('long', 1, current_price)
        elif action == 4:  # Open Short
            self.emulator.open_position('short', 1, current_price)
        else:  # Partial Close
            close_action = (action - 5) // self.max_positions
            close_amount = ((action - 5) % self.max_positions + 1) / self.max_positions
            if close_action == 0:  # Close Long
                self.emulator.close_position('long', self.emulator.position['long'] * close_amount, current_price)
            else:  # Close Short
                self.emulator.close_position('short', self.emulator.position['short'] * close_amount, current_price)

        self.emulator.update(current_price)
        self.emulator.trailing_stop_loss(current_price)

    def render(self):
        # This method can be implemented to visualize the trading process
        pass

    def get_action_space(self):
        return self.action_size

    def get_state_space(self):
        return len(self._get_state())

    def get_current_price(self):
        return self.data.iloc[self.current_step]['Close']

    def get_balance(self):
        return self.emulator.balance

    def get_position(self):
        return self.emulator.position

    def get_unrealized_pnl(self):
        return self.emulator.unrealized_pnl

    def get_realized_pnl(self):
        return self.emulator.balance - self.emulator.initial_balance

    def get_total_pnl(self):
        return self.get_realized_pnl() + self.get_unrealized_pnl()

    def get_max_drawdown(self):
        return self.emulator.max_drawdown

    def get_sharpe_ratio(self):
        return self.emulator.calculate_sharpe_ratio()

    def get_trading_history(self):
        return self.emulator.trading_history

    def get_current_step(self):
        return self.current_step

    def get_total_steps(self):
        return len(self.data)

    def is_done(self):
        return self.current_step >= len(self.data) - 1