from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.action_chains import ActionChains
from bs4 import BeautifulSoup
import time
import numpy as np

class Game (object):
    def __init__(self, sleep_time=.05, headless=False):
        self.driver = self.start_driver(headless)
        _, self.initial_state, self.cur_body = self.get_state()
        self.moves = []
        self.states = []
        self.moves_list = [Keys.DOWN, Keys.UP, Keys.LEFT, Keys.RIGHT]
        self.move_names = ["Down", "Up", "Left", "Right"]
        self.sleep_time = sleep_time
        self.results = None
    
    def start_driver(self, headless=False):
        options = webdriver.ChromeOptions()
        if headless:
            options.add_argument('headless')
            driver = webdriver.Chrome('./chromedriver_win32/chromedriver.exe', options=options)
            driver.get('https://play2048.co/')
        else:
            driver = webdriver.Chrome('./chromedriver_win32/chromedriver.exe')
            driver.get('https://play2048.co/')
            driver.set_window_position(635, 0)
            driver.set_window_size(655, 690)
            cookie_dismiss = driver.find_element_by_class_name('cookie-notice-dismiss-button')
            cookie_dismiss.click()
            driver.execute_script("document.getElementsByClassName('app-notice')[0].style.display = 'None'") 
            driver.execute_script("window.scrollTo(0, 250)") 
        
        return driver
    
    def reset(self):
        self.driver.execute_script("document.getElementsByClassName('retry-button')[0].click()")
        time.sleep(.5)
        _, self.initial_state, self.cur_body = self.get_state()
        self.moves = []
        self.states = []
        self.results = None
    
    def end_driver(self):
        self.driver.close()
    
    def move(self, move, give_reward=False):
        chosen_ndx = self.move_names.index(move)
        chosen_move = self.moves_list[chosen_ndx]
        self.cur_body.send_keys(chosen_move)
        self.moves.append(move)
        time.sleep(self.sleep_time)
        continue_playing, cur_state, self.cur_body = self.get_state()
        reward = self.get_reward()
        self.states.append(cur_state)
        if give_reward:
            return continue_playing, cur_state, reward
        else:
            return continue_playing, cur_state
        
    def get_reward(self):
        current_soup = BeautifulSoup(self.driver.page_source, 'html.parser')
        reward_container = current_soup.find("div", class_="score-addition")
        if (reward_container):
            reward = int(reward_container.contents[0][1:])
            return reward
        else: return 0

        
    def get_state(self):
        current_soup = BeautifulSoup(self.driver.page_source, 'html.parser')
        game_message = current_soup.find("div", class_="game-message")
        game_over = 'game-over' in game_message['class']
        game_won = 'game-won' in game_message['class']
        body = None
        state = np.zeros((4,4))
        continue_playing = True

        if game_won:
            continue_playing = "Game Won"
        elif game_over:
            continue_playing = "Game Over"
            
        game_state = current_soup.find("div", class_="tile-container")    
        body = self.driver.find_element_by_tag_name('body')
        # Coordinates start at top left of board
        for tile in game_state.contents:
            current_classes = tile['class']
            tile_value = current_classes[1].split('-')[1]
            tile_positions = current_classes[2].split('-')[-2:]
            tile_positions = [int(x) for x in tile_positions]
            row = tile_positions[1] - 1
            col = tile_positions[0] - 1
            state[row, col] = tile_value


        return continue_playing, state, body
    
    def print_states(self):
        for state in self.states:
            print(state)
            
    def calculate_results(self, elapsed_time):
        self.results = {
            'num_moves': len(self.moves),
            'biggest_tile': np.max(self.states[-1]),
            'sum_of_tiles': np.sum(self.states[-1]),
            'game_time': elapsed_time
        }