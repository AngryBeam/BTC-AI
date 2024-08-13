from logger import get_logger
import numpy as np
class TradingEmulator:
    def __init__(self, initial_balance=100000, commission=0.002, max_loss_percentage=0.3, max_capital_load=0.8):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.commission = commission
        self.position = {'long': 0, 'short': 0}
        self.open_orders = []
        self.max_profit = 0
        self.max_drawdown = 0
        self.unrealized_pnl = 0
        self.max_loss = initial_balance * max_loss_percentage
        self.max_capital_load = max_capital_load * initial_balance
        self.balance_history = [initial_balance]
        self.capital_loaded = {'long': 0, 'short': 0}
        self.max_capital_loaded = 0
        self.closed_position = {'long': 0, 'short': 0}
        self.max_position = {'long': 0, 'short': 0}
        self.trading_count = {'long': 0, 'short': 0}
        self.returns = []
        logger, _ = get_logger()
        self.logger = logger

    def open_position(self, side, amount, price, tp=None, sl=None):
        try:
            #self.logger.info(f"Opening {side} position: amount={amount}, price={price}")
            transaction_value = amount * price
            commission = amount * self.commission
            total_cost = transaction_value + commission
            total_loaded = self.capital_loaded['long'] + self.capital_loaded['short'] + total_cost

            if total_loaded <= self.max_capital_load:
                self.balance -= total_cost  # Deduct total cost from balance for both long and short

                if side == 'long':
                    self.position['long'] += amount
                    self.capital_loaded['long'] += total_cost
                    self.trading_count['long'] += 1
                    if sl is None:
                        sl = price * 0.95  # Default 5% stop loss for long positions
                else:
                    self.position['short'] += amount
                    self.capital_loaded['short'] += total_cost
                    self.trading_count['short'] += 1
                    if sl is None:
                        sl = price * 1.05  # Default 5% stop loss for short positions
                
                order = {'side': side, 'amount': amount, 'entry_price': price, 'tp': tp, 'sl': sl}
                self.open_orders.append(order)
                self.max_capital_loaded = max(self.max_capital_loaded, total_loaded)
                self.max_position['long'] = max(self.max_position['long'], self.position['long'])
                self.max_position['short'] = max(self.max_position['short'], self.position['short'])
            #self.logger.info(f"Position opened: {self.position}")
        except Exception as e:
            self.logger.error(f"Error opening position: {str(e)}")
            raise

    def close_position(self, side, amount, price):
        try:
            #self.logger.info(f"Closing {side} position: amount={amount}, price={price}")
            if (side == 'long' and self.position['long'] >= amount) or (side == 'short' and self.position['short'] >= amount):
                transaction_value = amount * price
                commission = transaction_value * self.commission
                total_return = transaction_value - commission

                self.balance += total_return  # Add total return to balance for both long and short

                if side == 'long':
                    self.position['long'] -= amount
                    self.capital_loaded['long'] -= total_return
                    self.closed_position['long'] += amount
                else:  # short
                    self.position['short'] -= amount
                    self.capital_loaded['short'] -= total_return
                    self.closed_position['short'] += amount

                # Update open orders
                self.open_orders = [order for order in self.open_orders if order['side'] != side or order['amount'] > amount]
                if self.open_orders and self.open_orders[-1]['side'] == side:
                    self.open_orders[-1]['amount'] -= amount
            #self.logger.info(f"Position closed: {self.position}")
        except Exception as e:
            self.logger.error(f"Error closing position: {str(e)}")
            raise

    def update(self, current_price):
        try:
            #self.logger.info(f"Updating with current price: {current_price}")
            for order in self.open_orders:
                if order['side'] == 'long':
                    if order['tp'] and current_price >= order['tp']:
                        self.close_position('long', order['amount'], order['tp'])
                    elif order['sl'] and current_price <= order['sl']:
                        self.close_position('long', order['amount'], order['sl'])
                else:
                    if order['tp'] and current_price <= order['tp']:
                        self.close_position('short', order['amount'], order['tp'])
                    elif order['sl'] and current_price >= order['sl']:
                        self.close_position('short', order['amount'], order['sl'])
            
            self.open_orders = [order for order in self.open_orders if order['amount'] > 0]
            
            self.unrealized_pnl = sum([(current_price - order['entry_price']) * order['amount'] 
                                    for order in self.open_orders if order['side'] == 'long']) + \
                                sum([(order['entry_price'] - current_price) * order['amount'] 
                                    for order in self.open_orders if order['side'] == 'short'])
            
            total_pnl = self.balance - self.initial_balance + self.unrealized_pnl
            self.max_profit = max(self.max_profit, total_pnl)
            self.max_drawdown = min(self.max_drawdown, total_pnl)
            self.max_position['long'] = max(self.max_position['long'], self.position['long'])
            self.max_position['short'] = max(self.max_position['short'], self.position['short'])
            # Check for max loss
            if total_pnl <= -self.max_loss:
                self.close_all_positions(current_price)
            #self.logger.info(f"Updated balance: {self.balance}, Unrealized P/L: {self.unrealized_pnl}")

            self.balance_history.append(self.balance)

            if len(self.balance_history) > 1:
                return_value = (self.balance - self.balance_history[-2]) / self.balance_history[-2]
                self.returns.append(return_value)
        except Exception as e:
            self.logger.error(f"Error updating: {str(e)}")
            raise

    def close_all_positions(self, current_price):
        for order in self.open_orders:
            self.close_position(order['side'], order['amount'], current_price)
        self.open_orders = []

    def trailing_stop_loss(self, current_price):
        try:
            #self.logger.info(f"Checking trailing stop loss: current_price={current_price}")
            for order in self.open_orders:
                if order['side'] == 'long':
                    new_sl = current_price * 0.95  # 5% trailing stop loss
                    if new_sl > order['sl']:
                        order['sl'] = new_sl
                else:
                    new_sl = current_price * 1.05  # 5% trailing stop loss
                    if new_sl < order['sl']:
                        order['sl'] = new_sl
            #self.logger.info(f"After trailing stop loss check: {self.position}")
        except Exception as e:
            self.logger.error(f"Error in trailing stop loss: {str(e)}")
            raise
    def get_performance_metrics(self):
        return {
            'balance': self.balance,
            'unrealized_pnl': self.unrealized_pnl,
            'total_pnl': self.balance - self.initial_balance + self.unrealized_pnl,
            'max_profit': self.max_profit,
            'max_drawdown': self.max_drawdown,
            'capital_loaded_long': self.capital_loaded['long'],
            'capital_loaded_short': self.capital_loaded['short'],
            'total_capital_loaded': sum(self.capital_loaded.values()),
            'max_capital_loaded': self.max_capital_loaded,
            'open_trade_long': self.position['long'],
            'open_trade_short': self.position['short'],
            'close_trade_long': self.closed_position['long'],
            'close_trade_short': self.closed_position['short'],
            'max_open_position_long': self.max_position['long'],
            'max_open_position_short': self.max_position['short'],
            'trading_count_long': self.trading_count['long'],
            'trading_count_short': self.trading_count['short'],
            'balance_history': self.balance_history,
            'returns': self.returns,
            'sharpe_ratio': self.calculate_sharpe_ratio(),
        }
    
    def calculate_sharpe_ratio(self, risk_free_rate=0):
       if len(self.returns) < 2:
           return 0
       returns_array = np.array(self.returns)
       return (np.mean(returns_array) - risk_free_rate) / np.std(returns_array)
    
    
    def calculate_reward(self, initial_balance, previous_balance, action):
        current_balance = self.balance
        profit = current_balance - previous_balance
        
        # Calculate returns
        returns = (current_balance - initial_balance) / initial_balance
        
        # Estimate risk (using the returns we're already tracking)
        risk = np.std(self.returns) if len(self.returns) > 1 else 0
        
        # Calculate Sharpe-like ratio (assuming risk-free rate is 0 for simplicity)
        sharpe = returns / risk if risk != 0 else 0
        
        # Drawdown penalty
        max_balance = max(self.balance_history)
        drawdown = (max_balance - current_balance) / max_balance
        drawdown_penalty = -drawdown * 10  # Adjust multiplier as needed
        
        # Holding reward/penalty
        holding_factor = 0.0001  # Adjust as needed
        holding_reward = holding_factor * profit if self.position['long'] > 0 or self.position['short'] > 0 else 0
        
        # Combine all factors
        reward = profit + sharpe + drawdown_penalty + holding_reward
        
        return reward