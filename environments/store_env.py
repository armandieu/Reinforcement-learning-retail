#!/usr/bin/env python

from environments.environment import BaseEnvironment
from collections import deque
import numpy as np
from copy import deepcopy
from scipy.stats import poisson
class Environment(BaseEnvironment):
    """Implements the environment for an RLGlue environment

    Note:
        env_init, env_start, env_step, env_cleanup, and env_message are required
        methods.
    """

    # def __init__(self):


    def env_init(self, env_info={}):
        """Setup for the environment called when the experiment first starts.

        Note:
            Initialize a tuple with the reward, first state observation, boolean
            indicating if it's terminal.
        """
        # These are needed if want to implement the paper A Simulation Environment and Reinforcement Learning
        # Method for Waste Reduction
        self.start_stock = env_info.get('initial_stock', 20) # State initial 20 items and life time of 3
        self.max_stock = env_info.get('capacity', 20)
        self.min_stock = 0

        self.max_loss = env_info.get('max_loss', 100)
        self.cum_profit = 0
        self.lags_to_return = env_info.get('lags_to_return',1) # Lags of states to return
        self.history = deque(maxlen=self.lags_to_return)
        # Stock
        self.stock = [] 
        self.sales = 0
        self.waste = 0
        self.total_waste = 0
        self.total_waste_cost = 0
        self.expiration_time = env_info.get('expiration_time', 30) # Expiration in days or 30
        self.expiration_data =[]
        self.cost_expired = env_info.get('expiration_cost', 1)

        # Time from making an order and receiving it
        self.lead_time = 5
        self.current_state = None
        # Otherwise a simple one
        self.capacity = env_info.get('capacity', 20) # Stock capacity
        self.maintenance = env_info.get('maintenance_cost') # Maintenance cost per unit
        self.cost = env_info.get('buying_price') # Buying price per unit
        self.selling_price = env_info.get('selling_price')  # Selling price per unit
        self.demand = env_info.get('weekly_demand') # Weekly average demand
        self.max_time = env_info.get('max_time') # Max time to run
        self.time = 0
        self.actual_demand = 0
        self.action_space = list(range(0, self.capacity))
    def _make_fast_order(self, action):
        # Return order cost
        pass
    def _waste(self):
        # Update waste
        pass
    def _reduceShelfLives(self):
        pass
    # Return sales, availability
    def _generateDemand(self):
        # Method to simulate demand
        pass
    def _addStock(self, units):
        # Method to re-stock certain quantity
        pass
    def env_start(self):
        """The first method called when the episode starts, called before the
        agent starts.

        Returns:
            The first state observation from the environment.
        """
        self.current_state = (self.start_stock, self.expiration_time)
        self.expiration_data = [{'qty': self.start_stock, 'exp': self.expiration_time}]
        self.time = 0
        self.cum_profit = 0
        self.sales = 0
        self.waste = 0
        self.total_waste = 0
        self.total_waste_cost = 0
        self.reward_obs_term = (0.0, self.observation(self.current_state), False)
        self.history = deque(maxlen=self.lags_to_return)
        return self.reward_obs_term[1]

    def env_step(self, action):
        """A step taken by the environment.

        Args:
            action: The action taken by the agent

        Returns:
            (float, state, Boolean): a tuple of the reward, state observation,
                and boolean indicating if it's terminal.
        """

        new_state = deepcopy(self.current_state)
        reward, state = self.sample(action)
        expir = self.expiration_data[0]['exp'] if len(self.expiration_data) > 0 else self.expiration_time # Older item expiration time
        new_state =(state, expir)
        
        self.time += 1
        self.cum_profit += reward
        is_terminal = False
        # Condition for terminal state
        # Either achieved max_time or cum profit is less than max loss
        if self.time > self.max_time or self.cum_profit < -self.max_loss: 
            is_terminal = True
        self.history.append(self.current_state)
        self.current_state = new_state
        # Last expiration times and values
        # print(self.history[:self.lags_to_return])
        # print([lag['exp'] for lag in self.expiration_data])


        self.reward_obs_term = (reward, self.observation(self.current_state), is_terminal)

        return self.reward_obs_term
    def update_expiration_qty(self, actual, demand, action):
        if(demand >= (actual + action)):
            # Reset expiration as we sold every product available
            self.expiration_data = []
        else:
            # Update the self.expiration_data accordingly
            demand_left = demand
            i = 0
            while i < len(self.expiration_data):
                val = self.expiration_data[i]
                if demand_left >= val['qty']:
                    # Remove the entire quantity
                    demand_left -= val['qty']
                    self.expiration_data.pop(i)
                else:
                    # Subtract the sold quantity
                    self.expiration_data[i]['qty'] -= demand_left
                    demand_left = 0  # No more demand left to fulfill
                    break
                if demand_left == 0:  # If completed the demand
                    break
                i += 1
    
    def update_expiration_days(self, days=1):
        sum_negatives = 0
        to_remove = []
        for i, exp_item in enumerate(self.expiration_data):
            exp_item['exp'] -= days  # Decrease expiration days by one
            if exp_item['exp'] <= 0:
                sum_negatives += exp_item['qty']
                to_remove.append(i)
        # Remove items with non-positive expiration days
        for index in reversed(to_remove):
            self.expiration_data.pop(index)
        return sum_negatives
    
    def add_item(self, qty):
        self.expiration_data.append({'qty':qty, 'exp':self.expiration_time})

    def sample(self, action):
        """ Utility function to sample reward and next state. """
        self.actual_demand = poisson.rvs(self.demand)

        reward = -self.maintenance*self.current_state[0] \
                    + self.selling_price*min([self.actual_demand, self.current_state[0] + action])
        
        # Add expiration information to products re-stocked
        if(action > 0):
            self.add_item(action)
        self.update_expiration_qty(self.current_state[0], self.actual_demand, action)
        expired_products = self.update_expiration_days()
        # Update reward on expired products
        self.waste = expired_products
        reward -= expired_products*self.cost_expired
        self.total_waste += self.waste
        self.total_waste_cost += self.waste*self.cost_expired
        # Update reward if not enough stock
        if (self.actual_demand > self.current_state[0]+ action):
            reward -= self.selling_price*(self.actual_demand - (self.current_state[0]+ action))
        next_state = max([(self.current_state[0] + action)-self.actual_demand, 0])
        return reward, next_state
    
    def observation(self, state):
        if self.lags_to_return > 1:
            padded_history = np.zeros((self.lags_to_return,2))
            if len(self.history) < self.lags_to_return:
                if(len(self.history)> 0):
                    padded_history[-len(self.history):] = np.array(self.history)
            else:
                padded_history[:] = np.array(self.history)
            padded_history = np.concatenate([padded_history[:,0], padded_history[:,1]])
            return padded_history
        else:
            return state  # Return the actual quantity in stock

    def env_cleanup(self):
        """Cleanup done after the environment ends"""
        pass

    def env_message(self, message):
        """A message asking the environment for information

        Args:
            message (string): the message passed to the environment

        Returns:
            string: the response (or answer) to the message
        """
        if message == "what is the current reward?":
            return "{}".format(self.reward_obs_term[0])

        # else
        return "I don't know how to respond to your message"
    # Added from https://github.com/rs-benatti/Retail_Store_Mangement_Reinforcement_Learning/blob/main/Retail_Store.ipynb example
    def reward_function(self):
        """ Computes the action-depend reward function r(s,a). """
        r = np.zeros((self.capacity+1, self.capacity+1))
        for s in range(self.capacity+1):
            for a in range(self.capacity+1):
                # Note: computing the expectation of the truncated Poisson distribution using the survival function
                r[s, a] = -self.maintenance*s -self.cost*a + self.selling_price*sum(poisson.sf(np.linspace(0, min(s+a, self.capacity)-1, num=min(s+a, self.capacity)), self.demand))
        return r
    def transition_function(self):
        """ Computes the action-depend transition probabilities p(s,a,s'). """
        p = np.zeros((self.capacity+1, self.capacity+1, self.capacity+1))
        for s in range(self.capacity+1):
            for a in range(self.capacity+1):
                for i in range(min(s+a, self.capacity)):
                    p[s, a, min(s+a, self.capacity)-i] = poisson.pmf(i, self.demand)
                p[s, a, 0] = poisson.sf(min(s+a, self.capacity)-1, self.demand)
        return p

    def reward_policy(self, pi):
        """ Computes the reward function r_pi(s) associated with a policy. """
        r = self.reward_function()
        r_pi = np.sum(np.multiply(r, pi), axis=1)
        return r_pi

    def transition_policy(self, pi):
        """ Computes the transition probabilities p_pi(s,s') associated with a policy. """
        p = self.transition_function()
        p_pi = np.zeros((self.capacity+1, self.capacity+1))
        for s in range(self.capacity+1):
            p_pi[s,:] = np.matmul(np.transpose(pi[s,:]), p[s,:,:])
        return p_pi

    def value_policy(self, pi, gamma):
        """ Computes the value function of a policy, with discount gamma (using matrix inversion). """
        r_pi = self.reward_policy(pi)
        p_pi = self.transition_policy(pi)
        v_pi = np.linalg.solve(np.eye(self.capacity+1) - gamma* p_pi, r_pi)
        return v_pi
