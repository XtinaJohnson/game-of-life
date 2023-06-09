import numpy as np


# Set to True to see debugging messages.
verbose = True

# Print debugging messages.
def show(s):
    if verbose:
        print(s)

# Create the initial 2-D array with random "live" (1) and "dead" (0) cells.
# The random sample is generated from elements of a.
# size is actually the output shape, with n * n samples drawn.
# n = 5
# grid1 = np.random.choice(a=np.array([0, 1]), size=(n, n))
# print(grid1)

# Iterate through the 2-D array and print each value.
# for x in np.nditer(grid1):
#     print(x)


# Any live cell with fewer than two live neighbors dies as if caused by underpopulation.
# Any live cell with two or three live neighbors lives on to the next generation.
# Any live cell with more than three live neighbors dies, as if by overpopulation.
# Any dead cell with exactly three live neighbors becomes a live cell, as if by reproduction.


def get_next_state(state, live_neighbors):
    next_state = state
    if state == 1:  # live
        if live_neighbors < 2:
            next_state = 0  # dies (underpopulation)
        if live_neighbors == 2 or live_neighbors == 3:
            next_state = 1  # lives to next gen
        if live_neighbors > 3:
            next_state = 0  # dies (overpopulation)
    if state == 0:  # dead
        if live_neighbors == 3:
            next_state = 1  # becomes alive (reproduction)
    return(next_state)


def count_neighbors(up, upper_left, left, lower_left, down, lower_right, right, upper_right):
    live_neighbors = 0
    if up == 1:
        live_neighbors += 1
    if upper_left == 1:
        live_neighbors += 1
    if left == 1:
        live_neighbors += 1
    if lower_left == 1:
        live_neighbors += 1
    if down == 1:
        live_neighbors += 1
    if lower_right == 1:
        live_neighbors += 1
    if right == 1:
        live_neighbors += 1
    if upper_right == 1:
        live_neighbors += 1
    return(live_neighbors)

def print_neighbors(up, upper_left, left, lower_left, down, lower_right, right, upper_right):
    print("up:", up)
    print("upper-left:", upper_left)
    print("left:", left)
    print("lower-left:", lower_left)
    print("down:", down)
    print("lower-right:", lower_right)
    print("right:", right)
    print("upper-right:", upper_right)

    

def get_neighbors_top_left_corner(grid, i, j):
    up = grid[row - 1, j]
    upper_left = grid[row - 1, col - 1]
    left = grid[i, col - 1]
    lower_left = grid[i + 1, col - 1]
    down = grid[i + 1, j]
    lower_right = grid[i + 1, j + 1]
    right = grid[i, j + 1]
    upper_right = grid[row - 1, j + 1]
    return(up, upper_left, left, lower_left, down, lower_right, right, upper_right)

def get_neighbors_top_right_corner(grid, i, j):
    # print("In get_neighbors_top_right_corner:")
    # print(grid)
    up = grid[row - 1, j]
    upper_left = grid[row - 1, j - 1]
    left = grid[i, j - 1]
    lower_left = grid[i + 1, j - 1]
    down = grid[i + 1, j]
    lower_right = grid[i + 1, 0]
    right = grid[i, 0]
    upper_right = grid[row - 1, 0]
    return(up, upper_left, left, lower_left, down, lower_right, right, upper_right)

def get_neighbors_bottom_left_corner(grid, i, j):
    up = grid[i - 1, j]
    upper_left = grid[i - 1, col - 1]
    left = grid[i, col - 1]
    lower_left = grid[0, col - 1]
    down = grid[0, 0]
    lower_right = grid[0, j + 1]
    right = grid[i, j + 1]
    upper_right = grid[i - 1, j + 1]
    return(up, upper_left, left, lower_left, down, lower_right, right, upper_right)
    
    
def get_neighbors_bottom_right_corner(grid, i, j):
    up = grid[i - 1, j]
    upper_left = grid[i - 1, j - 1]
    left = grid[i, j - 1]
    lower_left = grid[0, j - 1]
    down = grid[0, j]
    lower_right = grid[0, 0]
    right = grid[row - 1, 0]
    upper_right = grid[i - 1, 0]
    return(up, upper_left, left, lower_left, down, lower_right, right, upper_right)


def get_neighbors_top_row(grid, i, j): 
    up = grid[row - 1, j]
    upper_left = grid[row - 1, j - 1]
    left = grid[0, j - 1]
    lower_left = grid[i + 1, j - 1]
    down = grid[i + 1, j]
    lower_right = grid[i + 1, j + 1]
    right = grid[0, j + 1]
    upper_right = grid[row - 1, j + 1]
    return(up, upper_left, left, lower_left, down, lower_right, right, upper_right)
        
def get_neighbors_bottom_row(grid, i, j):
    up = grid[i - 1, j]
    upper_left = grid[i - 1, j - 1]
    left = grid[i, j - 1]
    lower_left = grid[0, j - 1]
    down = grid[0, j]
    lower_right = grid[0, j + 1]
    right = grid[i, j + 1]
    upper_right = grid[i - 1, j + 1]
    return(up, upper_left, left, lower_left, down, lower_right, right, upper_right)

def get_neighbors_left_column(grid, i, j):
    up = grid[i - 1, j]
    upper_left = grid[i - 1, col - 1]
    left = grid[i, col - 1]
    lower_left = grid[i + 1, col - 1]
    down = grid[i + 1, j]
    lower_right = grid[i + 1, j + 1]
    right = grid[i, j + 1]
    upper_right = grid[i - 1, j + 1]
    return(up, upper_left, left, lower_left, down, lower_right, right, upper_right)

def get_neighbors_right_column(grid, i, j):
    up = grid[i - 1, j]
    upper_left = grid[i - 1, j - 1]
    left = grid[i, j - 1]
    lower_left = grid[i + 1, j - 1]
    down = grid[i + 1, j]
    lower_right = grid[i + 1, 0]
    right = grid[i, 0]
    upper_right = grid[i - 1, 0]
    return(up, upper_left, left, lower_left, down, lower_right, right, upper_right)

def get_neighbors_body(grid, i, j):
    up = grid[i - 1, j]
    upper_left = grid[i - 1, j - 1]
    left = grid[i, j - 1]
    lower_left = grid[i + 1, j - 1]
    down = grid[i + 1, j]
    lower_right = grid[i + 1, j + 1]
    right = grid[i, j + 1]
    upper_right = grid[i - 1, j + 1]
    return(up, upper_left, left, lower_left, down, lower_right, right, upper_right)


def change_grid(grid):
    # np.ndenumerate() gets the index and value of each grid item! Useful.
    new_grid = np.full(shape=(row, col), fill_value=7)
    # print("This generation's starting grid:")
    # print(grid)
    # print("This generation's new grid (starting point):")
    # print(new_grid)

    for idx, val in np.ndenumerate(grid):
        # print(idx)
        i, j = idx
        # print(i, j, val)
        if i == 0:  # top row with corners
            if j == 0:
                # print("Top-left corner:")
                # print(i, j, val)
                # print("Sending this to get_neighbors_top_left_corner:")
                # print(grid)
                up, upper_left, left, lower_left, down, lower_right, right, upper_right = get_neighbors_top_left_corner(grid, i, j)
                # print_neighbors(up, down, left, right)
                live_neighbors = count_neighbors(up, upper_left, left, lower_left, down, lower_right, right, upper_right)
                # print("live neighbors:", live_neighbors)
                next_state = get_next_state(val, live_neighbors)
                # print("next state:", next_state)
                new_grid[i, j] = next_state
                # print(new_grid)
                
            elif j == col - 1:
                # print("Top-right corner:")
                # print(i, j, val)
                # print("Sending this to get_neighbors_top_right_corner:")
                # print(grid)
                up, upper_left, left, lower_left, down, lower_right, right, upper_right = get_neighbors_top_right_corner(grid, i, j)
                # print_neighbors(up, down, left, right)
                live_neighbors = count_neighbors(up, upper_left, left, lower_left, down, lower_right, right, upper_right)
                # print("live neighbors:", live_neighbors)
                next_state = get_next_state(val, live_neighbors)
                # print("next state:", next_state)
                new_grid[i, j] = next_state
                # print("New grid so far:")
                # print(new_grid)
            else: 
                # print("Top row (non-corners):")
                # print(i, j, val)
                up, upper_left, left, lower_left, down, lower_right, right, upper_right = get_neighbors_top_row(grid, i, j)
                # print_neighbors(up, down, left, right)
                live_neighbors = count_neighbors(up, upper_left, left, lower_left, down, lower_right, right, upper_right)
                # print("live neighbors:", live_neighbors)
                next_state = get_next_state(val, live_neighbors)
                # print("next state:", next_state)
                new_grid[i, j] = next_state
                # print(new_grid)
        elif i == row - 1:  # bottom row with corners
            if j == 0:
                # print("Bottom-left corner:")
                # print(i, j, val)
                up, upper_left, left, lower_left, down, lower_right, right, upper_right = get_neighbors_bottom_left_corner(grid, i, j)
                # print_neighbors(up, down, left, right)
                live_neighbors = count_neighbors(up, upper_left, left, lower_left, down, lower_right, right, upper_right)
                # print("live neighbors:", live_neighbors)
                next_state = get_next_state(val, live_neighbors)
                # print("next state:", next_state)
                new_grid[i, j] = next_state
                # print(new_grid)
            elif j == col - 1:
                # print("Bottom-right corner:")
                # print(i, j, val)
                up, upper_left, left, lower_left, down, lower_right, right, upper_right = get_neighbors_bottom_right_corner(grid, i, j)
                # print_neighbors(up, down, left, right)
                live_neighbors = count_neighbors(up, upper_left, left, lower_left, down, lower_right, right, upper_right)
                # print("live neighbors:", live_neighbors)
                next_state = get_next_state(val, live_neighbors)
                new_grid[i, j] = next_state
                # print(new_grid)
            else:
                # print("Bottom row (non-corners):")
                # print(i, j, val)
                up, upper_left, left, lower_left, down, lower_right, right, upper_right = get_neighbors_bottom_row(grid, i, j)
                # print_neighbors(up, down, left, right)
                live_neighbors = count_neighbors(up, upper_left, left, lower_left, down, lower_right, right, upper_right)
                # print("live neighbors:", live_neighbors)
                next_state = get_next_state(val, live_neighbors)
                # print("next state:", next_state)
                new_grid[i, j] = next_state
                # print(new_grid)
        elif j == 0:
            # print("Left column (non-corners):")
            # print(i, j, val)
            up, upper_left, left, lower_left, down, lower_right, right, upper_right = get_neighbors_left_column(grid, i, j)
            # print_neighbors(up, down, left, right)
            live_neighbors = count_neighbors(up, upper_left, left, lower_left, down, lower_right, right, upper_right)
            # print("live neighbors:", live_neighbors)
            next_state = get_next_state(val, live_neighbors)
            # print("next state:", next_state)
            new_grid[i, j] = next_state
            # print(new_grid)
        elif j == col - 1:
            # print("Right column (non-corners):")
            # print(i, j, val)
            up, upper_left, left, lower_left, down, lower_right, right, upper_right = get_neighbors_right_column(grid, i, j)
            # print_neighbors(up, down, left, right)
            live_neighbors = count_neighbors(up, upper_left, left, lower_left, down, lower_right, right, upper_right)
            # print("live neighbors:", live_neighbors)
            next_state = get_next_state(val, live_neighbors)
            # print("next state:", next_state)
            new_grid[i, j] = next_state
            # print(new_grid)
        else:
            # print("Body:")
            # print(i, j, val)
            up, upper_left, left, lower_left, down, lower_right, right, upper_right = get_neighbors_body(grid, i, j)
            # print_neighbors(up, down, left, right)
            live_neighbors = count_neighbors(up, upper_left, left, lower_left, down, lower_right, right, upper_right)
            # print("live neighbors:", live_neighbors)
            next_state = get_next_state(val, live_neighbors)
            # print("next state:", next_state)
            new_grid[i, j] = next_state
            # print(new_grid)
    return(new_grid)

def process_generations(initial_grid, n):
    current_grid = initial_grid
    print("Initial grid")
    print(current_grid)
    for i in range(n):
        print("Generation", i)
        next_grid = change_grid(current_grid)
        print("Next grid")
        print(next_grid)
        current_grid = next_grid
    return(current_grid)

# Create the initial 2-D array with random "live" (1) and "dead" (0) cells.
# The random sample is generated from elements of a.
# size is actually the output shape, with row * col samples drawn.
# row = 4
# col = 4

# Random grid
# Get row, col from .shape
# initial_grid = np.random.choice(a=np.array([0, 1]), size=(10, 10))
# row, col = initial_grid.shape
# process_generations(initial_grid, 10)

# arange grid for testing
# row = 5
# col = 5
# arr = np.array(np.arange(row * col))
# initial_grid = np.reshape(a=arr, newshape=(row, col))
# up, upper_left, left, lower_left, down, lower_right, right, upper_right = get_neighbors_body(initial_grid, 1, 1)
# print(initial_grid)
# print_neighbors(up, upper_left, left, lower_left, down, lower_right, right, upper_right)


# For bug fix
# initial_grid=np.array([[1, 0, 0, 1, 1, 0, 1, 1, 1],
# [1, 1, 0, 0, 0, 1, 1, 1, 1],
# [1, 0, 1, 1, 0, 0, 1, 0, 0],
# [0, 0, 1, 0, 0, 0, 1, 0, 1],
# [0, 0, 1, 0, 1, 1, 1, 0, 1],
# [1, 1, 0, 1, 0, 0, 1, 0, 1],
# [1, 1, 0, 0, 1, 0, 1, 0, 0],
# [0, 1, 1, 1, 0, 0, 0, 1, 1],
# [0, 0, 1, 0, 0, 0, 0, 0, 1]])
# arr = np.array(np.arange(row * col))
# grid = np.reshape(a=arr, newshape=(row, col))
# print(grid)

# Create an empty grid to be populated.
# new_grid = np.full(shape=(row, col), fill_value=7)
# print(new_grid)


  
block = np.array([[0, 0, 0, 0],
                  [0, 1, 1, 0],
                  [0, 1, 1, 0],
                  [0, 0, 0, 0]])

beehive = np.array([[0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 0, 0],
                    [0, 1, 0, 0, 1, 0],
                    [0, 0, 1, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0]])

loaf = np.array([[0, 0, 0, 0, 0, 0],
                 [0, 0, 1, 1, 0, 0],
                 [0, 1, 0, 0, 1, 0],
                 [0, 0, 1, 0, 1, 0],
                 [0, 0, 0, 1, 0, 0],
                 [0, 0, 0, 0, 0, 0]])

boat = np.array([[0, 0, 0, 0, 0],
                 [0, 1, 1, 0, 0],
                 [0, 1, 0, 1, 0],
                 [0, 0, 1, 0, 0],
                 [0, 0, 0, 0, 0]])

tub = np.array([[0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 1, 0, 1, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0]])



row, col = tub.shape


def test_still_life():
    start = process_generations(tub, 100)
    end = tub
    assert np.array_equal(start, end) 

