import numpy as np
import utils


class Agent:
    def __init__(self, actions, Ne=40, C=40, gamma=0.7, display_width=18, display_height=10):
        # HINT: You should be utilizing all of these
        self.actions = actions
        self.Ne = Ne  # used in exploration function
        self.C = C
        self.gamma = gamma
        self.display_width = display_width
        self.display_height = display_height
        self.reset()
        # Create the Q Table to work with
        self.Q = utils.create_q_table()
        self.N = utils.create_q_table()
        
    def train(self):
        self._train = True
        
    def eval(self):
        self._train = False

    # At the end of training save the trained model
    def save_model(self, model_path):
        utils.save(model_path, self.Q)
        utils.save(model_path.replace('.npy', '_N.npy'), self.N)

    # Load the trained model for evaluation
    def load_model(self, model_path):
        self.Q = utils.load(model_path)

    def reset(self):
        # HINT: These variables should be used for bookkeeping to store information across time-steps
        # For example, how do we know when a food pellet has been eaten if all we get from the environment
        # is the current number of points? In addition, Q-updates requires knowledge of the previously taken
        # state and action, in addition to the current state from the environment. Use these variables
        # to store this kind of information.
        self.points = 0
        self.s = None
        self.a = None
    
    def update_n(self, state, action):
        # TODO - MP11: Update the N-table.
        #if (state is not None and action is not None):
            self.N[state][action] += 1
        #pass

    def update_q(self, s, a, r, s_prime):
        # TODO - MP11: Update the Q-table.
        #if (s is not None and a is not None and s_prime is not None) or r == -1:
            # if self.N[s + (a,)] < self.Ne:
            #     f_value = 1
            # else:
            #     f_value = self.Q[s + (a,)]

            # a_t = np.argmax(f_value)

            alpha = self.C / (self.C + self.N[s][a])

            self.Q[s][a] = self.Q[s][a] + alpha * (r + self.gamma * np.max(self.Q[s_prime]) - self.Q[s][a])
        #pass        

    def act(self, environment, points, dead):
        '''
        :param environment: a list of [snake_head_x, snake_head_y, snake_body, food_x, food_y, rock_x, rock_y] to be converted to a state.
        All of these are just numbers, except for snake_body, which is a list of (x,y) positions 
        :param points: float, the current points from environment
        :param dead: boolean, if the snake is dead
        :return: chosen action between utils.UP, utils.DOWN, utils.LEFT, utils.RIGHT

        Tip: you need to discretize the environment to the state space defined on the webpage first
        (Note that [adjoining_wall_x=0, adjoining_wall_y=0] is also the case when snake runs out of the playable board)
        '''
        s_prime = self.generate_state(environment)

        # TODO - MP12: write your function here

        # some important bookkeeping
        reward = -0.1
        priority_order = [utils.RIGHT, utils.LEFT, utils.DOWN, utils.UP]   # correct priority order

        if self._train is True:
            # analyzing information coming in from the environment - are we dead
            if dead is True:
                reward = -1
                self.update_n(self.s, self.a)
                self.update_q(self.s, self.a, reward, s_prime)
                
                # need to reset the game after a death
                self.reset()
                
                #print("ACTION: ", actions[0])
                return priority_order[0]    # can return anything since we reset anyway
        
            if self.s is not None and self.a is not None:
                reward = -0.1
                if points > self.points:
                    reward = 1   # did we eat food pellet
                
                self.update_n(self.s, self.a)
                self.update_q(self.s, self.a, reward, s_prime)
        else:
            if dead is True:
                reward = -1
                self.update_n(self.s, self.a)
                self.update_q(self.s, self.a, reward, s_prime)
                
                # need to reset the game after a death
                self.reset()
                
                #print("ACTION: ", actions[0])
                return priority_order[0]    # can return anything since we reset anyway

        # check if we are in training phase - update q and n tables
        # if self._train:
        #     self.update_n(self.s, self.a)
        #     self.update_q(self.s, self.a, reward, s_prime)
        
        # exploration stuff
        act = self.optimal_action(priority_order, s_prime)
        
        # updating 
        self.s = s_prime
        self.a = act
        self.points = points
        
        return act

    
    def optimal_action(self, actions_array, s_prime):
        action = [self.exploration_policy(direction, s_prime) for direction in actions_array]
        index = np.argmax(action)
        return actions_array[index]


    def exploration_policy(self, direction, s_prime):
        food_dir_x, food_dir_y, adjoining_wall_x, adjoining_wall_y, adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right = s_prime
        
        # if we're not in training phase, then we return the self.Q
        if self._train:
            if self.N[(food_dir_x, food_dir_y, adjoining_wall_x, adjoining_wall_y, adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right, direction)] < self.Ne:
                return 1
            else:
                return self.Q[(food_dir_x, food_dir_y, adjoining_wall_x, adjoining_wall_y, adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right, direction)]
        else:
            return self.Q[(food_dir_x, food_dir_y, adjoining_wall_x, adjoining_wall_y, adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right, direction)]
        

    def generate_state(self, environment):
        '''
        :param environment: a list of [snake_head_x, snake_head_y, snake_body, food_x, food_y, rock_x, rock_y] to be converted to a state.
        All of these are just numbers, except for snake_body, which is a list of (x,y) positions 
        '''
        # TODO - MP11: Implement this helper function that generates a state given an environment 
        snake_head_x, snake_head_y, snake_body, food_x, food_y, rock_x, rock_y = environment

        # head and food
        if food_x < snake_head_x:
            food_dir_x = 1
        else:
            if food_x > snake_head_x:
                food_dir_x = 2
            else:
                food_dir_x = 0
        
        if food_y < snake_head_y:
            food_dir_y = 1
        else:
            if food_y > snake_head_y:
                food_dir_y = 2
            else:
                food_dir_y = 0

        # head and wall
        adjoining_wall_x = 0
        adjoining_wall_y = 0

        if snake_head_x == 1 or (snake_head_x == rock_x + 2 and snake_head_y == rock_y) or (snake_head_x == self.display_width - 2 and snake_head_x == rock_x + 2 and snake_head_y == rock_y):
            adjoining_wall_x = 1
        elif snake_head_x == self.display_width - 2 or (snake_head_x == rock_x - 1 and snake_head_y == rock_y):
            adjoining_wall_x = 2

        if snake_head_y == 1 or (snake_head_y == rock_y + 1 and snake_head_x == rock_x) or (snake_head_y == rock_y + 1 and snake_head_x == rock_x + 1) or (snake_head_y == self.display_height - 2 and snake_head_y == rock_y + 1 and snake_head_x == rock_x) or (snake_head_y == self.display_height - 2 and snake_head_y == rock_y + 1 and snake_head_x == rock_x + 1):
            adjoining_wall_y = 1
        elif snake_head_y == self.display_height - 2 or (snake_head_y == rock_y - 1 and snake_head_x == rock_x):
            adjoining_wall_y = 2

        
        # head and body
        if (snake_head_x, snake_head_y - 1) in snake_body:
            adjoining_body_top = 1
        else:
            adjoining_body_top = 0

        if (snake_head_x, snake_head_y + 1) in snake_body:
            adjoining_body_bottom = 1
        else:
            adjoining_body_bottom = 0

        if (snake_head_x - 1, snake_head_y) in snake_body:
            adjoining_body_left = 1
        else:
            adjoining_body_left = 0

        if (snake_head_x + 1, snake_head_y) in snake_body:
            adjoining_body_right = 1
        else:
            adjoining_body_right = 0

        state = (food_dir_x, food_dir_y, adjoining_wall_x, adjoining_wall_y, adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right)
        #print("STATE: ", state)

        return state
        #return None
