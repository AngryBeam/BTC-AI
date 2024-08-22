import numpy as np
from collections import defaultdict

class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.value = 0

class MCTS:
    def __init__(self, agent, num_simulations=1000, exploration_weight=1.0, exploration_constant=1.41, max_depth=50):
        self.agent = agent
        self.num_simulations = num_simulations
        self.exploration_weight = exploration_weight
        self.Q = defaultdict(int)  # total reward of each state-action pair
        self.N = defaultdict(int)  # total visit count for each state-action pair
        self.exploration_constant = exploration_constant
        self.max_depth = max_depth

    def search(self, root_state):
        root = MCTSNode(root_state)
        
        for _ in range(self.num_simulations):
            node = self.select(root)
            reward = self.simulate(node.state)
            self.backpropagate(node, reward)
        
        return self.best_action(root)

    def select(self, node):
        while node.children:
            if not all(n in self.N for n, _ in node.children.items()):
                return self.expand(node)
            else:
                node = self.ucb_select(node)
        return self.expand(node)

    def expand(self, node):
        actions = self.get_possible_actions(node.state)
        for action in actions:
            if action not in node.children:
                new_state = self.get_next_state(node.state, action)
                new_node = MCTSNode(new_state, parent=node)
                node.children[action] = new_node
                return new_node
        return node

    def simulate(self, state):
        '''
        while not self.is_terminal(state):
            actions = self.get_possible_actions(state)
            action = np.random.choice(actions)
            state = self.get_next_state(state, action)
        return self.get_reward(state)
        '''
        depth = 0
        while not self.agent.is_terminal(state) and depth < self.max_depth:
            actions = self.agent.get_possible_actions(state)
            action = np.random.choice(actions)
            state = self.agent.get_next_state(state, action)
            depth += 1
        return self.agent.get_reward(state)

    def backpropagate(self, node, reward):
        while node:
            node.visits += 1
            node.value += reward
            node = node.parent

    def best_action(self, node):
        return max(node.children, key=lambda a: node.children[a].value / node.children[a].visits)

    def ucb_select(self, node):
        '''
        log_N_vertex = np.log(node.visits)
        
        def ucb(n):
            if self.N[(node, n)] == 0:
                return float('inf')
            return self.Q[(node, n)] / self.N[(node, n)] + self.exploration_weight * np.sqrt(
                log_N_vertex / self.N[(node, n)])
        
        return max(node.children, key=ucb)
        '''
        log_N_vertex = np.log(node.visits)
        
        def ucb(n):
            if self.N[(node, n)] == 0:
                return float('inf')
            return self.Q[(node, n)] / self.N[(node, n)] + self.exploration_constant * np.sqrt(
                log_N_vertex / self.N[(node, n)])
        
        return max(node.children, key=ucb)

    def get_possible_actions(self, state):
        return self.agent.get_possible_actions(state)

    def get_next_state(self, state, action):
        return self.agent.get_next_state(state, action)

    def is_terminal(self, state):
        return self.agent.is_terminal(state)

    def get_reward(self, state):
        return self.agent.get_reward(state)