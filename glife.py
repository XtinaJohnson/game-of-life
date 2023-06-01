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


def count_neighbors(up, left, down, right):
    live_neighbors = 0
    if up == 1:
        live_neighbors += 1
    if left == 1:
        live_neighbors += 1
    if down == 1:
        live_neighbors += 1
    if right == 1:
        live_neighbors += 1
    return(live_neighbors)

def print_neighbors(up, left, down, right):
    print("up:", up)
    print("left:", left)
    print("down:", down)
    print("right:", right)

    

def get_neighbors_top_left_corner(grid, i, j):
    live_neighbors = 0
    up = grid[row - 1, j]
    left = grid[i, col - 1]
    down = grid[i + 1, j]
    right = grid[i, j + 1]
    return(up, left, down, right)

def get_neighbors_top_right_corner(grid, i, j):
    live_neighbors = 0
    up = grid[row - 1, j]
    left = grid[i, j - 1]
    down = grid[i + 1, j]
    right = grid[i, 0]
    return(up, left, down, right)

def get_neighbors_bottom_left_corner(grid, i, j):
    live_neighbors = 0
    up = grid[i - 1, j]
    left = grid[i, col - 1]
    down = grid[0, 0]
    right = grid[i, j + 1]
    return(up, left, down, right)
    
    
def get_neighbors_bottom_right_corner(grid, i, j):
    live_neighbors = 0
    up = grid[i - 1, j]
    left = grid[i, j - 1]
    down = grid[0, j]
    right = grid[row - 1, 0]
    return(up, left, down, right)


def get_neighbors_top_row(grid, i, j):  # consider that i is 0 in the top row...
    live_neighbors = 0
    up = grid[row - 1, j]
    left = grid[0, j - 1]
    down = grid[i + 1, j]
    right = grid[0, j + 1]
    return(up, left, down, right)
        
def get_neighbors_bottom_row(grid, i, j):
    live_neighbors = 0
    up = grid[i - 1, j]
    left = grid[i, j - 1]
    down = grid[0, j]
    right = grid[i, j + 1]
    return(up, left, down, right)

def get_neighbors_left_column(grid, i, j):
    live_neighbors = 0
    up = grid[i - 1, j]
    left = grid[i, col - 1]
    down = grid[i + 1, j]
    right = grid[i, j + 1]
    return(up, left, down, right)

def get_neighbors_right_column(grid, i, j):
    live_neighbors = 0
    up = grid[i - 1, j]
    left = grid[i, j - 1]
    down = grid[i + 1, j]
    right = grid[i, 0]
    return(up, left, down, right)

def get_neighbors_body(grid, i, j):
    live_neighbors = 0
    up = grid[i - 1, j]
    left = grid[i, j - 1]
    down = grid[i + 1, j]
    right = grid[i, j + 1]
    return(up, left, down, right)

# Create the initial 2-D array with random "live" (1) and "dead" (0) cells.
# The random sample is generated from elements of a.
# size is actually the output shape, with row * col samples drawn.
row = 3
col = 4
grid = np.random.choice(a=np.array([0, 1]), size=(row, col))
# arr = np.array(np.arange(row * col))
# grid = np.reshape(a=arr, newshape=(row, col))
print(grid)

# Create an empty grid to be populated.
new_grid = np.full(shape=(row, col), fill_value=7)
print(new_grid)

# np.ndenumerate() gets the index and value of each grid item! Useful.
for idx, val in np.ndenumerate(grid):
    # print(idx)
    i, j = idx
    # print(i, j, val)
    if i == 0:  # top row with corners
        if j == 0:
            print("Top-left corner:")
            print(i, j, val)
            up, down, left, right = get_neighbors_top_left_corner(grid, i, j)
            print_neighbors(up, down, left, right)
            live_neighbors = count_neighbors(up, down, left, right)
            print("live neighbors:", live_neighbors)
            next_state = get_next_state(val, live_neighbors)
            print("next state:", next_state)
            new_grid[i, j] = next_state
            print(new_grid)
            
        elif j == col - 1:
            print("Top-right corner:")
            print(i, j, val)
            up, down, left, right = get_neighbors_top_right_corner(grid, i, j)
            print_neighbors(up, down, left, right)
            live_neighbors = count_neighbors(up, down, left, right)
            print("live neighbors:", live_neighbors)
            next_state = get_next_state(val, live_neighbors)
            print("next state:", next_state)
            new_grid[i, j] = next_state
            print(new_grid)
        else: 
            print("Top row (non-corners):")
            print(i, j, val)
            up, down, left, right = get_neighbors_top_row(grid, i, j)
            print_neighbors(up, down, left, right)
            live_neighbors = count_neighbors(up, down, left, right)
            print("live neighbors:", live_neighbors)
            next_state = get_next_state(val, live_neighbors)
            print("next state:", next_state)
            new_grid[i, j] = next_state
            print(new_grid)
    elif i == row - 1:  # bottom row with corners
        if j == 0:
            print("Bottom-left corner:")
            print(i, j, val)
            up, down, left, right = get_neighbors_bottom_left_corner(grid, i, j)
            print_neighbors(up, down, left, right)
            live_neighbors = count_neighbors(up, down, left, right)
            print("live neighbors:", live_neighbors)
            next_state = get_next_state(val, live_neighbors)
            print("next state:", next_state)
            new_grid[i, j] = next_state
            print(new_grid)
        elif j == col - 1:
            print("Bottom-right corner:")
            print(i, j, val)
            up, down, left, right = get_neighbors_bottom_right_corner(grid, i, j)
            print_neighbors(up, down, left, right)
            live_neighbors = count_neighbors(up, down, left, right)
            print("live neighbors:", live_neighbors)
            next_state = get_next_state(val, live_neighbors)
            print("next state:", next_state)
            new_grid[i, j] = next_state
            print(new_grid)
        else:
            print("Bottom row (non-corners):")
            print(i, j, val)
            up, down, left, right = get_neighbors_bottom_row(grid, i, j)
            print_neighbors(up, down, left, right)
            live_neighbors = count_neighbors(up, down, left, right)
            print("live neighbors:", live_neighbors)
            next_state = get_next_state(val, live_neighbors)
            print("next state:", next_state)
            new_grid[i, j] = next_state
            print(new_grid)
    elif j == 0:
        print("Left column (non-corners):")
        print(i, j, val)
        up, down, left, right = get_neighbors_left_column(grid, i, j)
        print_neighbors(up, down, left, right)
        live_neighbors = count_neighbors(up, down, left, right)
        print("live neighbors:", live_neighbors)
        next_state = get_next_state(val, live_neighbors)
        print("next state:", next_state)
        new_grid[i, j] = next_state
        print(new_grid)
    elif j == col - 1:
        print("Right column (non-corners):")
        print(i, j, val)
        up, down, left, right = get_neighbors_right_column(grid, i, j)
        print_neighbors(up, down, left, right)
        live_neighbors = count_neighbors(up, down, left, right)
        print("live neighbors:", live_neighbors)
        next_state = get_next_state(val, live_neighbors)
        print("next state:", next_state)
        new_grid[i, j] = next_state
        print(new_grid)
    else:
        print("Body:")
        print(i, j, val)
        up, down, left, right = get_neighbors_body(grid, i, j)
        print_neighbors(up, down, left, right)
        live_neighbors = count_neighbors(up, down, left, right)
        print("live neighbors:", live_neighbors)
        next_state = get_next_state(val, live_neighbors)
        print("next state:", next_state)
        new_grid[i, j] = next_state
        print(new_grid)

  

