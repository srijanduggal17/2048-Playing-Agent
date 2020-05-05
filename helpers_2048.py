import numpy as np

# Check if a move is valid in a certain state
def isvalid(state, move):      
    cur_state = np.copy(state)

    if move == "Up":
        if is_row_combinable(cur_state.T):
            return True
        board_state = cur_state.T
    elif move == "Down":
        if is_row_combinable(cur_state.T):
            return True
        board_state = np.flip(cur_state.T, axis=1)
    elif move == "Left":
        if is_row_combinable(cur_state):
            return True
        board_state = cur_state     
    elif move == "Right":
        if is_row_combinable(cur_state):
            return True
        board_state = np.flip(cur_state, axis=1)
    else:
        print("Chosen Move is not an Option")
        return False
    
    num_zeros = count_zeros_left(board_state)
    result = num_zeros != 0
    return result

def is_row_combinable(cur_state):        
    # Loop over each row
    for row in cur_state:
        # Remove empty tiles
        row_collapsed = row[row != 0]

        # Check if row has combinable tiles
        if are_combinable(row_collapsed)[0]:
            return True


# Checks if there are any tiles that can
#  be combined given a list of tiles
# Returns a boolean and the biggest tile value that can be combined 
def are_combinable(row):
    row_collapsed = row[row != 0]
    # row_collapsed: list of tiles without empty spaces
    if len(row_collapsed) != 0:
        if len(set(row_collapsed)) < len(row_collapsed):
            # Find if there are duplicate tiles
            # adjacent to each other
            rolled_row = np.roll(row_collapsed,1)
            rolled_row[0] = 0
            diff = row_collapsed - rolled_row
            ndx_of_zero = np.where(diff == 0)
            collapsable_vals = rolled_row[ndx_of_zero]
            if (0 in row_collapsed - rolled_row):
                return True, max(collapsable_vals)

            # If there are no duplicate tiles adjacent to each other
            else: return False, None

        # If there are no duplicate tiles in the list
        else: return False, None

    # If there are no tiles in the list
    else: return False, None

# Given a state, counts the number of zeros
# to the left of each nonzero tile
def count_zeros_left(state):
    cur_sum = 0
    for row in state:                
        # Find position of nonzero tiles
        nonzero_ndx = np.nonzero(row)[0]

        # Sum the number of zeros to the left
        for n in nonzero_ndx:
            cur_sum += np.sum(row[:n] == 0)
    return cur_sum

# Checks if the biggest number in a row 
# is in the rightmost possible position
def max_is_rightmost(row): 
    flipped = np.flip(row)
    
    # Find position of max tile
    max_ndx = np.argmax(flipped)
    
    # Count zeros to the left of max tile
    zeros_left = np.sum(flipped[:max_ndx] == 0)
    return zeros_left == 0

# Checks if a row is full
def isrowfull(row):
    return np.sum(row == 0) == 0

def up_combinable_with_row(cur_state, rownum):
    target_row = cur_state[rownum]
    
    # Collapse columns below
    rows_below = cur_state[rownum+1:]
    virtual_row_below = np.zeros((1,4)).squeeze()
    for ndx in range(4):
        col = rows_below.T[ndx]
        collapsed = col[col != 0]
        if len(collapsed) == 0:
            virtual_row_below[ndx] = 0
        else:
            virtual_row_below[ndx] = collapsed[0]
            
    # Check if combinable
    diff = target_row - virtual_row_below
    combos = np.nonzero(diff == 0)[0]
    if len(combos) == 0:
        combinable = False
        combo_sum = None
    elif np.max(target_row[combos]) == 0:
        combinable = False
        combo_sum = None
    else:
        combinable = True
        combo_sum = np.sum(target_row[combos])
    return combinable, combo_sum

def anticipated_state(cur_state, move):
    if move == "Left":
        board = cur_state
    elif move == "Right":
        board = np.flip(cur_state, axis=1)
    elif move == "Up":
        board = cur_state.T
    else:
        board = np.flip(cur_state.T, axis=1)
    
    anticipated = np.zeros_like(cur_state)
    for i in range(4):
        anticipated[i] = collapse_row(board[i])
    
    if move == "Right":
        anticipated = np.flip(anticipated, axis=1)
    elif move == "Up":
        anticipated = anticipated.T
    elif move == "Down":
        anticipated = np.flip(anticipated, axis=1).T
    return anticipated
        
def collapse_row(row):
    # Remove zeros from row
    removed_zeros = row[row != 0]
    
    # Check which indices can be combined in row
    diff = removed_zeros[:-1] - removed_zeros[1:]
    combo_indices = np.where(diff == 0)[0]
    
    # Initialize output row
    out_row = np.zeros((1,4)).squeeze()
    out_row[:len(removed_zeros)] = removed_zeros
    
    while len(combo_indices) != 0:
        ndx_to_replace = combo_indices[0]
        val = out_row[ndx_to_replace]
        out_row[ndx_to_replace] = 2*val
        out_row[ndx_to_replace + 1:-1] = out_row[ndx_to_replace+2:]
        out_row[-1] = 0

        # Check which indices can be combined in row
        diff = out_row[ndx_to_replace+1:-1] - out_row[ndx_to_replace+2:]
        combo_indices = ndx_to_replace+1 + np.where(diff == 0)[0]
        
    return out_row

def compare_ops_1(cur_state, target_row):
    options = [["Up"], ["Left", "Up"], ["Right", "Up"], ["Left", "Left", "Up"], ["Right", "Right", "Up"]]
    ops = [0,0,0,0,0]
    
    combine_0, val_0 = up_combinable_with_row(cur_state, target_row)
    if combine_0:
        ops[0] = val_0
    
    ant_state_1 = anticipated_state(cur_state, "Left")
    combine_1, val_1 = up_combinable_with_row(ant_state_1, target_row)
    if combine_1:
        ops[1] = val_1
    
    ant_state_2 = anticipated_state(cur_state, "Right")
    combine_2, val_2 = up_combinable_with_row(ant_state_2, target_row)
    if combine_2:
        ops[2] = val_2
        
    ant_state_3 = anticipated_state(ant_state_1, "Left")
    combine_3, val_3 = up_combinable_with_row(ant_state_3, target_row)
    if combine_3:
        ops[3] = val_3

    ant_state_4 = anticipated_state(ant_state_2, "Right")
    combine_4, val_4 = up_combinable_with_row(ant_state_4, target_row)
    if combine_4:
        ops[4] = val_4
    
    best_op_ndx = np.argmax(ops)
    is_good = ops[best_op_ndx] != 0
    next_move = None
    if is_good:
        next_move = options[best_op_ndx]
    
    return is_good, next_move

def rule_9(cur_state, game, rule_str, printmode):
    if isvalid(cur_state, "Up"):
        continue_playing, cur_state = game.move("Up")
        if printmode: print('Rule {} Up{}'.format(rule_str[0], rule_str[1]))

    elif isvalid(cur_state, "Right"):
        continue_playing, cur_state = game.move("Right")
        if printmode: print('Rule {} Right{}'.format(rule_str[0], rule_str[1]))

    elif isvalid(cur_state, "Left"):
        continue_playing, cur_state = game.move("Left")
        if printmode: print('Rule {} Left{}'.format(rule_str[0], rule_str[1]))

    else:
        continue_playing, cur_state = game.move("Down")
        continue_playing, cur_state = game.move("Up")
        if printmode: print('Rule {} Down-Up{}'.format(rule_str[0], rule_str[1]))
    return continue_playing, cur_state