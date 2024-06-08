import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import random
import time
from collections import deque
from statistics import mean
from Ac_constants import AcConstants
from collections import deque
import numpy as np


from ObjectiveFunction import objective

# ALTERATION: NO jump reward



AC = AcConstants()
class WingOptimEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    def __init__(self):
        super(WingOptimEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space =spaces.Box(low=-1, high=1,
                                            shape=(9,), dtype=np.float32)
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(low=-10, high=20,
                                            shape=(9, ), dtype=np.float64)

    def adjustment(self, action):
        # 1Process Action into new design vector
        # u_bounds = [15, 3, 2.5, 1, 20, 5, 1, 10, 0.20]
        # l_bounds = [8, 0.5, 0.5, 0.1, 0, 1.4, 0, -10, 0.11] #Changed
        kink_lower_bound = AC.kink_lb
        kink_upper_bound = AC.kink_ub


        def take_action(idx, action_stat, current_var):
            u_bounds = AC.upper_bound
            l_bounds = AC.lower_bound
            action_dist = 75

            action_step = (u_bounds[idx] - l_bounds[idx]) / action_dist
            action_step = action_stat * action_step # Negative?
            new_var = current_var + action_step
            if new_var > u_bounds[idx]:
                new_var = u_bounds[idx]
                self.bound_counter = 1
            if new_var < l_bounds[idx]:
                new_var = l_bounds[idx]
                self.bound_counter = 1

            return new_var

        def update_design_vector(action_stat):
            for idx in range(len(self.design_vector)):
                self.design_vector[idx] = take_action(idx, action_stat[idx], self.design_vector[idx])

            return self.design_vector


        self.design_vector = update_design_vector(action)


        # wingspan
        if self.design_vector[5] - (AC.fuselage_width/2) < kink_lower_bound * 0.5 * (self.design_vector[0]-AC.fuselage_width):
            self.design_vector[5] = kink_upper_bound * 0.5 * (self.design_vector[0]-AC.fuselage_width)
            self.constraint_counter = 1
        if self.design_vector[5] - (AC.fuselage_width/2) > kink_upper_bound * 0.5 * (self.design_vector[0]-AC.fuselage_width):
            self.design_vector[5] = kink_upper_bound * 0.5 * (self.design_vector[0]-AC.fuselage_width)
            self.constraint_counter = 1
            return

        # Root chord
        if 0.5 * self.design_vector[1] > self.design_vector[2]:
            self.design_vector[2] = 0.5 * self.design_vector[1]#CHanged
            self.constraint_counter = 1
        if self.design_vector[1] < self.design_vector[2]:
            self.design_vector[2] = self.design_vector[1] #Changed
            self.constraint_counter = 1

        # Kink chord
        if self.design_vector[2] > self.design_vector[1]:
            self.design_vector[1] = self.design_vector[2]
            self.constraint_counter = 1
        if self.design_vector[2] < 0.5 * self.design_vector[1]:
            self.design_vector[2] = 0.5 * self.design_vector[1]
            self.constraint_counter = 1
        if self.design_vector[2] < self.design_vector[3]:
            self.design_vector[3] = self.design_vector[2]
            self.constraint_counter = 1

        # Tip chord
        if self.design_vector[3] > self.design_vector[2]:
            self.design_vector[2] = self.design_vector[3]
            self.constraint_counter = 1


        # Yoffset kin
        if self.design_vector[5] - (AC.fuselage_width/2) > kink_upper_bound * 0.5 * (self.design_vector[0]- AC.fuselage_width):
            self.design_vector[5] = kink_upper_bound * 0.5 * (self.design_vector[0]- AC.fuselage_width)
            self.constraint_counter = 1
        if self.design_vector[5] - (AC.fuselage_width/2) < kink_lower_bound * 0.5 * (self.design_vector[0]-AC.fuselage_width):
            self.design_vector[5] = kink_lower_bound * 0.5 * self.design_vector[0]
            self.constraint_counter = 1

        # Zoffset tip
        # Twist
        # Thickness chord


    def step(self, action):
        #1Reset the Evaluation
        self.reward = 0
        self.reward_obtained = 0
        self.reward_penalty = 0
        self.step_counter += 1
        self.bound_counter = 0
        self.constraint_counter = 0
        self.adjustment(action)

        #________________________________________________________
        #2New design vector calculations
        current_objective, wing_mass_percentage , wing_volume, wing_volume_product, wing_loading = objective(self.design_vector)

        #________________________________________________________
        #3 Generate REWARDS based on objective

        #define Scalable penalty and reward terms
        P1 = -0.2                       #Bounds hits
        P2 = -0.5                       #Geometric constraints
        X3 = [6, 2, 1, 0]               #Mass constraint [1-2]
        X3_2 = [3, 2, 1, 0]             # Mass constraint [1-2]
        X4 = [5, 2, 1, 0]               # Volume constraint [1-2]
        X5 = [5, 2, 1, 0]               #Loading constraint
        P6 = -5                         # AVL error
        X7 = [1.5, -0.1]                # Valid wing, improved from previous, no new optim [0.2 - 2]
        X8 = [5, -0.2]                  # Valid wing, same as previous, no new optim  [0.5 -2]
        X9 = [10, -0.4]                 # Valid wing, worse than previous, no new optim  [0.5 - 2]
        P10 = -0.05                     # Same exact optimum

        R1 = [20, 1, 3, 0, 40, 1.0, 4]  # Reward for optimizing [1-5]
        R2 = [28, 4]                       # Final reward [+-100]

        # Limit counters
        optimization_counter_limit = 50
        mass_counter_limit = 25
        volume_counter_limit = 25
        loading_counter_limit = 20
        AVL_counter_limit = 10

        # Condition NEG
        if current_objective < self.score:

            # NEG A: Constraints Hit
            # Hitting Bounds
            if self.bound_counter == 1:
                self.reward_penalty += P1

            if current_objective < 2:
                # NEG A1: Geometry Constraint hit
                if  self.constraint_counter == 1:
                    self.reward_penalty += P2

                # NEG A2: Wing Weight / Volume incorrect / Wing Loading too high
                if current_objective == 1.005 or current_objective == 1.006 or current_objective == 1.007:  # wing weight incorrect
                    #Wing mass
                    if wing_mass_percentage > AC.wing_mass_ub:
                        self.reward_penalty += - X3[2] * ((X3[0] * (wing_mass_percentage - AC.wing_mass_ub) /  AC.wing_mass_ub + 1) **X3[1]) + X3[3]
                        self.mass_counter += 1
                    if wing_mass_percentage < AC.wing_mass_lb and wing_mass_percentage > 0:
                        self.reward_penalty += - X3_2[2] * ((X3_2[0] * (AC.wing_mass_lb - wing_mass_percentage) / AC.wing_mass_lb + 1) **X3_2[1]) + X3_2[3]
                        self.mass_counter += 1
                    if self.mass_counter >= mass_counter_limit:
                        self.done = True

                    #wing volume
                    if wing_volume < wing_volume_product and wing_volume > 0:
                        self.reward_penalty +=  - X4[2] * ((X4[0] * (wing_volume_product - wing_volume) / wing_volume_product + 1) ** X4[1]) + X4[3]
                        self.volume_counter += 1
                        if self.volume_counter >= volume_counter_limit :
                            self.done = True

                    #wing loading
                    if wing_loading > 1.1 * AC.wing_loading_ref :
                        self.reward_penalty += - X5[2] * ((X5[0] * (wing_loading - (1.1 * AC.wing_loading_ref)) / (1.1 * AC.wing_loading_ref) + 1) ** X5[1]) + X5[3]
                        if self.loading_counter >= loading_counter_limit:
                            self.done = True


                # NEG A4: Weight =/= Lift
                if current_objective == 1e-8 : # AVL error
                    self.reward_penalty += P6
                    self.AVL_counter += 1

                if self.AVL_counter >= AVL_counter_limit:
                    self.done = True


            # NEG B: Valid CL/CD but non max:
            if current_objective > 2:

                # NEG B1: MAX > Current > Previous
                # No done counter added
                if current_objective > self.previous_objective:
                    self.reward_penalty += AC.transfer_learning * ((current_objective - self.score) / AC.objective_start) * ( X7[0]) + X7[1]

                # NEG B2: Current == Previous
                if current_objective == self.previous_objective:
                    self.done_counter += 1
                    self.reward_penalty += AC.transfer_learning * ((current_objective - self.score) / AC.objective_start) * (X8[0]) + X8[1]

                # NEG B3: Previous > Current
                if current_objective < self.previous_objective:
                    self.done_counter += 1
                    self.reward_penalty += AC.transfer_learning * ((current_objective - self.score) / AC.objective_start) * (X9[0]) + X9[1]


            if self.done_counter >= optimization_counter_limit:
                self.done = True


        # Condition NEG C: Current objective == MAX
        if current_objective == self.score:
            self.reward_penalty += P10

            #Termination Reset
            self.done_counter = 0
            self.AVL_counter = 0
            self.mass_counter = 0
            self.volume_counter = 0
            self.loading_counter = 0

        #Condition POS : New MAX CL/CD obtained
        if current_objective > self.score:
            self.reward_obtained =  ((AC.transfer_learning * R1[0] * ((current_objective - AC.objective_start)/AC.objective_start) + R1[1]) ** R1[2] - R1[3]) #* ((AC.transfer_learning * R1[4] * ((current_objective - self.score)/self.objective_start) + R1[5]) ** R1[6])

            # Store best result
            self.score = current_objective
            self.best_design_vector = np.copy(self.design_vector)

            #Termination Reset
            self.done_counter = 0
            self.AVL_counter = 0
            self.mass_counter = 0
            self.volume_counter = 0
            self.loading_counter = 0

        #Get final reward
        if self.done == True:
            self.reward_obtained += (AC.transfer_learning * ((self.score - AC.objective_start) / AC.objective_start) * R2[0] + 1)** R2[1]

        #________________________________________________________
        #4 Return results
        self.reward = self.reward_obtained + self.reward_penalty
        self.cummulative_reward += self.reward
        self.total_reward += self.reward_obtained
        self.total_penalty += self.reward_penalty
        self.previous_objective = current_objective
        self.observation = self.design_vector
        self.observation = np.array(self.observation)

        truncated = False

        #TODO: get constraint termination info

        info = {'Current CLCD: ': current_objective, 'Done Counter': self.done_counter ,'Action':action , 'Design Vector': self.design_vector,'Step Counter:':self.step_counter,'Heavy Wing cntr': self.mass_counter,'volume counter': self.volume_counter,'No lift cntr': self.AVL_counter, 'bound cntr':self.bound_counter,'Best CL/CD': self.score, 'Best Design': self.best_design_vector, 'Cummulative reward':self.cummulative_reward, 'Total reward':self.total_reward, 'Total penalty':self.total_penalty}
        return self.observation, self.reward, self.done,truncated, info

    def reset(self, seed = None, options=None):
        self.done = False
        self.done_counter = 0
        self.AVL_counter = 0
        self.score = AC.objective_start
        self.previous_objective = AC.objective_start
        self.reward = 0
        self.reward_penalty = 0
        self.objective = 0
        self.step_counter = 0
        self.bound_counter = 0
        self.mass_counter = 0
        self.volume_counter = 0
        self.loading_counter = 0
        #self.evalcounter = 0
        self.cummulative_reward= 0
        self.total_reward = 0
        self.total_penalty = 0

        # Initial conditions> Design Vector
        self.wing_span = AC.wing_span_start
        self.chord_root = AC.chord_root_start
        self.chord_kink = AC.chord_kink_start
        self.chord_tip = AC.chord_tip_start
        self.sweep_le = AC.sweep_le_start
        self.yoffset_kink = AC.yoffset_kink_start
        self.zoffset_tip = AC.zoffset_tip_start
        self.twist = AC.twist_start
        self.thickness_chord = AC.thickness_chord_start

        self.design_vector = [self.wing_span, self.chord_root, self.chord_kink, self.chord_tip, self.sweep_le, self.yoffset_kink, self.zoffset_tip, self.twist, self.thickness_chord]
        self.objective_start = AC.objective_start
        self.best_design_vector = self.design_vector


        self.observation = [self.wing_span, self.chord_root, self.chord_kink, self.chord_tip, self.sweep_le, self.yoffset_kink, self.zoffset_tip, self.twist, self.thickness_chord]
        self.observation = np.array(self.observation)


        info = {'Current CLCD: ': self.objective_start}
        return self.observation, info # reward, done, info can't be included

