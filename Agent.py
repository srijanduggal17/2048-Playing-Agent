import numpy as np
import pandas as pd
import time
from Game import Game
from helpers_2048 import *
import torch
from DQN import *

class Agent (object):
    def __init__(self, sleep_time=.05, headless=False, printmode=False):
        self.move_names = ["Down", "Up", "Left", "Right"]
        self.policies = {
            "random": self.random_policy,
            "combination": self.combination_policy,
            "bigger_combo": self.priority_bigger_policy,
            "designed": self.designed_policy,
            "pretrained": self.pretrained_policy
        }
        self.sleep_time = sleep_time
        self.headless = headless
        self.printmode = printmode
    
    # Play a game and collect experience
    def collect_experience(self, num_episodes=1):
        game = Game(self.sleep_time, self.headless)
        cur_episode = self.experience_game(game)

        experiences = {
            1: cur_episode
        }
        
        count = 1
        while count < num_episodes:
            print('{} episodes complete'.format(count))
            count += 1
            game.reset()
            cur_episode = self.experience_game(game)
            experiences[count] = cur_episode
            
        return experiences
    
    def experience_game(self, game):
        experience = []
#         (state, action, nextstate, reward)
        start_time = time.time()
        continue_playing = True
        cur_state = game.initial_state
        while not isinstance(continue_playing, str):
            choices = np.array([0, 1, 2, 3])
            rand_move = np.random.choice(choices)
            
            while not isvalid(cur_state, self.move_names[rand_move]):
                choices = choices[choices != rand_move]
                if len(choices) == 0:
                    continue_playing = "Invalid";
                    break;
                rand_move = np.random.choice(choices)
            
            if not isinstance(continue_playing, str):
                chosen_move = self.move_names[rand_move]
                continue_playing, next_state, reward = game.move(chosen_move, give_reward=True)
                cur_experience = (cur_state, chosen_move, next_state, reward)
                experience.append(cur_experience)
                cur_state = next_state

        end_time = time.time()
        game.calculate_results(end_time - start_time)
        
        return experience
        
    
    # Plays a game following a policy
    def play_game(self, policy):
        game = Game(self.sleep_time, self.headless)
        completed_game = self.policies[policy](game)
        return completed_game
    
    # Plays a game following a policy using a driver that is already open
    def replay_game(self, game, policy):
        game.reset()
        completed_game = self.policies[policy](game)
        return completed_game
    
    # Function to evaluate the performance of a policy
    def evaluate(self, policy, num_games=30):
        cur_game = self.play_game('random')
        df_overall = pd.DataFrame(columns = cur_game.results.keys())
        df_overall = df_overall.append(cur_game.results, ignore_index=True)
        
        for i in range(2, num_games+1):
            cur_game = self.replay_game(cur_game, 'random')
            df_overall = df_overall.append(cur_game.results, ignore_index=True)
            if i%5 == 0:
                print('Finished Game {}'.format(i))
        
        cur_game.end_driver()
        
        statistics = {
            'num_moves_mean': df_overall.num_moves.mean(),
            'num_moves_unc': df_overall.num_moves.std()/np.sqrt(num_games),
            'biggest_tile_mean': df_overall.biggest_tile.mean(),
            'biggest_tile_unc': df_overall.biggest_tile.std()/np.sqrt(num_games),
            'sum_of_tiles_mean': df_overall.sum_of_tiles.mean(),
            'sum_of_tiles_unc': df_overall.sum_of_tiles.std()/np.sqrt(num_games),
            'policy': 'random'
        }
        
        return df_overall, statistics
    
    # Pretrained policy using experiences collected from random moves
    def pretrained_policy(self, game):
        move_ops = ['Left', 'Right', 'Up', 'Down']
        start_time = time.time()
        continue_playing = True
        cur_state = game.initial_state
        
        policy_network = DQN(4,4,4)
        policy_network.load_state_dict(torch.load('trained_model.pt'))
        policy_network.eval()
        
        while not isinstance(continue_playing, str):
            network_out = policy_network(process_state(cur_state))
            ranked_actions = np.argsort(network_out.data.numpy()[0])
            chosen_action = ranked_actions[0]
            chosen_move = move_ops[chosen_action]
            
            while not isvalid(cur_state, chosen_move):
                ranked_actions = ranked_actions[ranked_actions != chosen_action]
                if len(ranked_actions) == 0:
                    continue_playing = "Invalid";
                    break;
                chosen_action = ranked_actions[0]
                chosen_move = move_ops[chosen_action]
                
            if not isinstance(continue_playing, str):
                continue_playing, cur_state = game.move(chosen_move)

        end_time = time.time()
        game.calculate_results(end_time - start_time)
        
        return game
    
    # Designed policy
    def designed_policy(self, game):
        start_time = time.time()
        continue_playing = True
        cur_state = game.initial_state
        
        while not isinstance(continue_playing, str):
            is_good_1, next_move_1 = compare_ops_1(cur_state, 0)
            is_good_2, next_move_2 = compare_ops_1(cur_state, 1)
            is_good_3, next_move_3 = compare_ops_1(cur_state, 2)

            # Move biggest number in top row to the right if possible
            if not max_is_rightmost(cur_state[0]):
                continue_playing, cur_state = game.move("Right")
                if self.printmode: print('Rule 1 Right: Max 1st row')

            elif not isrowfull(cur_state[0]):
                continue_playing, cur_state = rule_9(cur_state, game, ['2', ': 1st row not full'], self.printmode)     

            elif are_combinable(cur_state[0])[0]:
                continue_playing, cur_state = game.move("Right")
                if self.printmode: print('Rule 3 Right: Combo 1st row')

            elif is_good_1:
                outstr = 'Rule 4 '
                for move in next_move_1:
                    continue_playing, cur_state = game.move(move)
                    outstr += move + ' '
                if self.printmode: print(outstr)

            elif are_combinable(cur_state[1])[0]:
                continue_playing, cur_state = game.move("Right")
                if self.printmode: print('Rule 5 Right: Combo 2nd row')

            elif not max_is_rightmost(cur_state[1]):
                continue_playing, cur_state = game.move("Right")
                if self.printmode: print('Rule 6 Right: Max 2nd row')

            elif isrowfull(cur_state[1]):
                if is_good_2:
                    outstr = 'Rule 7a '
                    for move in next_move_2:
                        continue_playing, cur_state = game.move(move)
                        outstr += move + ' '
                    if self.printmode: print(outstr)

                elif are_combinable(cur_state[2])[0]:
                    continue_playing, cur_state = game.move("Right")
                    if self.printmode: print('Rule 7b Right: Combo 3rd row')

                elif not max_is_rightmost(cur_state[2]):
                    continue_playing, cur_state = game.move("Right")
                    if self.printmode: print('Rule 7c Right: Max 3rd row')

                elif isrowfull(cur_state[2]):
                    if is_good_3:
                        outstr = 'Rule 8a '
                        for move in next_move_3:
                            continue_playing, cur_state = game.move(move)
                            outstr += move + ' '
                        if self.printmode: print(outstr)

                    elif are_combinable(cur_state[3])[0]:
                        continue_playing, cur_state = game.move("Right")
                        if self.printmode: print('Rule 8b Right: Combo 4th row')

                    elif not max_is_rightmost(cur_state[3]):
                        continue_playing, cur_state = game.move("Right")
                        if self.printmode: print('Rule 8c Right: Max 4th row')

                    else:
                        continue_playing, cur_state = rule_9(cur_state, game, ['9',''], self.printmode)
                else:
                    continue_playing, cur_state = rule_9(cur_state, game, ['9',''], self.printmode)

            else:
                continue_playing, cur_state = rule_9(cur_state, game, ['9',''], self.printmode)     
        
        end_time = time.time()
        game.calculate_results(end_time - start_time)
        
        return game
        
        
    # Random policy definition
    def random_policy(self, game):
        start_time = time.time()
        continue_playing = True
        cur_state = game.initial_state
        while not isinstance(continue_playing, str):
            choices = np.array([0, 1, 2, 3])
            rand_move = np.random.choice(choices)
            
            while not isvalid(cur_state, self.move_names[rand_move]):
                choices = choices[choices != rand_move]
                if len(choices) == 0:
                    continue_playing = "Invalid";
                    break;
                rand_move = np.random.choice(choices)
            
            if not isinstance(continue_playing, str):
                chosen_move = self.move_names[rand_move]
                continue_playing, cur_state = game.move(chosen_move)

        end_time = time.time()
        game.calculate_results(end_time - start_time)
        
        return game
    
    # Policy that greedily chooses to combine if possible
    def combination_policy(self, game):
        start_time = time.time()
        continue_playing = True
        cur_state = game.initial_state

        # First move is random
        choices = np.array([0, 1, 2, 3])
        rand_move = np.random.choice(choices)        
        while not isvalid(cur_state, self.move_names[rand_move]):
            choices = choices[choices != rand_move]
            rand_move = np.random.choice(choices)

        chosen_move = self.move_names[rand_move]
        continue_playing, cur_state = game.move(chosen_move)

        # Move to combine tiles in rows/cols until the game is over
        while not isinstance(continue_playing, str):
            # Initialize action booleans
            combine_along_row = False
            combine_along_col = False

            # Loop over each row
            for row in cur_state:
                # Check if row has combinable tiles
                if are_combinable(row)[0]:
                    combine_along_row = True
                    break

            # Loop over each col
            for col in np.transpose(cur_state):
                # Check if row has combinable tiles
                if are_combinable(col)[0]:
                    combine_along_col = True
                    break

            # Choose move
            if combine_along_row == combine_along_col:
                choices = np.array([0, 1, 2, 3])
                rand_move = np.random.choice(choices)        
                while not isvalid(cur_state, self.move_names[rand_move]):
                    choices = choices[choices != rand_move]
                    rand_move = np.random.choice(choices)
                chosen_move = self.move_names[rand_move]
                continue_playing, cur_state = game.move(chosen_move)
                
            elif combine_along_col:
                choices = np.array([0,1])
                rand_move = np.random.choice(choices)        
                chosen_move = self.move_names[rand_move]
                continue_playing, cur_state = game.move(chosen_move)
            else:
                choices = np.array([2,3])
                rand_move = np.random.choice(choices)        
                chosen_move = self.move_names[rand_move]
                continue_playing, cur_state = game.move(chosen_move)

        end_time = time.time()
        game.calculate_results(end_time - start_time)
        
        return game

    # Policy that greedily chooses to combine the biggest number if possible
    def priority_bigger_policy(self, game):
        start_time = time.time()
        continue_playing = True
        cur_state = game.initial_state

        # First move is random
        choices = np.array([0, 1, 2, 3])
        rand_move = np.random.choice(choices)        
        while not isvalid(cur_state, self.move_names[rand_move]):
            choices = choices[choices != rand_move]
            rand_move = np.random.choice(choices)

        chosen_move = self.move_names[rand_move]
        continue_playing, cur_state = game.move(chosen_move)
        
        # Move to combine tiles in rows/cols until the game is over
        while not isinstance(continue_playing, str):
            # Initialize actions
            combine_along_row = []
            combine_along_col = []

            # Loop over each row
            for row in cur_state:
                # Check if row has combinable tiles
                combinable, val = are_combinable(row)
                if combinable:
                    combine_along_row.append(val)

            # Loop over each col
            for col in np.transpose(cur_state):
                # Check if row has combinable tiles
                combinable, val = are_combinable(col)
                if combinable:
                    combine_along_col.append(val)

            # Choose move
            if (len(combine_along_row) == 0) and (len(combine_along_col) == 0):
                # Make random move
                choices = np.array([0, 1, 2, 3])
                rand_move = np.random.choice(choices)        
                while not isvalid(cur_state, self.move_names[rand_move]):
                    choices = choices[choices != rand_move]
                    rand_move = np.random.choice(choices)
                chosen_move = self.move_names[rand_move]
                continue_playing, cur_state = game.move(chosen_move)
                
            elif len(combine_along_row) == 0:
                # Make random vertical move
                choices = np.array([0,1])
                rand_move = np.random.choice(choices)        
                chosen_move = self.move_names[rand_move]
                continue_playing, cur_state = game.move(chosen_move)
                
            elif len(combine_along_col) == 0:
                # Make random horizontal move
                choices = np.array([2,3])
                rand_move = np.random.choice(choices)        
                chosen_move = self.move_names[rand_move]
                continue_playing, cur_state = game.move(chosen_move)
                
            elif max(combine_along_row) == max(combine_along_col):
                # Make random move
                choices = np.array([0, 1, 2, 3])
                rand_move = np.random.choice(choices)        
                while not isvalid(cur_state, self.move_names[rand_move]):
                    choices = choices[choices != rand_move]
                    rand_move = np.random.choice(choices)
                chosen_move = self.move_names[rand_move]
                continue_playing, cur_state = game.move(chosen_move)
                
            elif max(combine_along_col) > max(combine_along_row):
                # Make random vertical move
                choices = np.array([0,1])
                rand_move = np.random.choice(choices)        
                chosen_move = self.move_names[rand_move]
                continue_playing, cur_state = game.move(chosen_move)
                
            else:
                # Make random horizontal move
                choices = np.array([2,3])
                rand_move = np.random.choice(choices)        
                chosen_move = self.move_names[rand_move]
                continue_playing, cur_state = game.move(chosen_move)

        end_time = time.time()
        game.calculate_results(end_time - start_time)
        
        return game