from logger import setup_logger


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
        
        self.capital_loaded = {'long': 0, 'short': 0}
        self.max_capital_loaded = 0

     

    def open_position(self, side, amount, price, tp=None, sl=None):
        commission = amount * self.commission
        total_loaded = self.capital_loaded['long'] + self.capital_loaded['short'] + amount
        if total_loaded <= self.max_capital_load:
            if side == 'long':
                self.position['long'] += amount
                self.balance -= amount + commission
                self.capital_loaded['long'] += amount
                if sl is None:
                    sl = price * 0.95  # Default 5% stop loss for long positions
            else:
                self.position['short'] += amount
                self.balance += amount - commission
                self.capital_loaded['short'] += amount
                if sl is None:
                    sl = price * 1.05  # Default 5% stop loss for short positions
            
            order = {'side': side, 'amount': amount, 'entry_price': price, 'tp': tp, 'sl': sl}
            self.open_orders.append(order)
            self.max_capital_loaded = max(self.max_capital_loaded, total_loaded)

    def close_position(self, side, amount, price):
        commission = amount * self.commission
        if side == 'long':
            self.position['long'] -= amount
            self.balance += amount - commission
            self.capital_loaded['long'] -= amount
        else:
            self.position['short'] -= amount
            self.balance -= amount + commission
            self.capital_loaded['short'] -= amount

    def update(self, current_price):
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

        # Check for max loss
        if total_pnl <= -self.max_loss:
            self.close_all_positions(current_price)

    def close_all_positions(self, current_price):
        for order in self.open_orders:
            self.close_position(order['side'], order['amount'], current_price)
        self.open_orders = []

    def trailing_stop_loss(self, current_price):
        for order in self.open_orders:
            if order['side'] == 'long':
                new_sl = current_price * 0.95  # 5% trailing stop loss
                if new_sl > order['sl']:
                    order['sl'] = new_sl
            else:
                new_sl = current_price * 1.05  # 5% trailing stop loss
                if new_sl < order['sl']:
                    order['sl'] = new_sl
    
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
            'max_capital_loaded': self.max_capital_loaded
        }