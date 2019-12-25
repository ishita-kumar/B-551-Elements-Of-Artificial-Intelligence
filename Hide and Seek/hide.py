#!/usr/local/bin/python3
#
# hide.py : a simple friend-hider
#
# Submitted by : Ishita Kumar (ishkumar@iu.edu)
#
# Based on skeleton code by D. Crandall and Z. Kachwala, 2019
#
# The problem to be solved is this:
# Given a campus map, find a placement of F friends so that no two can find one another.
#

import sys


# Parse the map from a given filename
def parse_map(filename):
	with open(filename, "r") as f:
		return [[char for char in line] for line in f.read().split("\n")]

# Count total # of friends on board
def count_friends(board):
    return sum([ row.count('F') for row in board ] )

# Return a string with the board rendered in a human-friendly format
def printable_board(board):
    return "\n".join([ "".join(row) for row in board])

# Add a friend to the board at the given position, and return a new board (doesn't change original)
def add_friend(board, row, col):
    return board[0:row] + [board[row][0:col] + ['F',] + board[row][col+1:]] + board[row+1:]

# Validate whether F is safe to put on row,col 
def valid_neighbors(board, r, c):
    if board[r][c] == '&' or board[r][c] == 'F' or board[r][c] == '#' or board[r][c] == '@':
        return False
    i = r
    # Checking the neighbour in South direction
    while i != len(board) - 1:
        if board[i+1][c] == "&":
            break
        elif board[i+1][c] == "F":
            return False
        i = i + 1

    # Checking the neighbour in North direction
    i = r
    while i >= 0:
        if board[i-1][c] == "&":
            break
        elif board[i-1][c] == "F":
            return False
        i = i - 1

    # Checking the neighbour in the East direction
    j = c
    while j != len(board[0]) - 1:
        if board[r][j+1] == "&":
            break
        elif board[r][j+1] == "F":
            return False
        j = j + 1
            
    # Checking the neighbour in the West direction
    j = c
    while j >= 0:
        if board[r][j-1] == "&":
            return True
        elif board[r][j-1] == "F":
            return False
        j = j - 1

    return True
    

                    
# Get list of successors of given board state
def successors(board):
    return [ add_friend(board, r, c) for r in range(0, len(board)) for c in range(0,len(board[0])) if board[r][c] == '.' and valid_neighbors(board,r,c)]

# check if board is a goal state
def is_goal(board):
    return count_friends(board) == K 

# Solve n-rooks!
def solve(initial_board):
    visited = []
    fringe = [initial_board]
    while len(fringe) > 0:
        popval = fringe.pop()
        visited.append(popval)
        for s in successors(popval):
            if s in visited:
                continue   
            if is_goal(s):
                return(s)  
            fringe.append(s)
    return False

# Main Function
if __name__ == "__main__":
    IUB_map=parse_map(sys.argv[1])

    # This is K, the number of friends
    K = int(sys.argv[2])
    print ("Starting from initial board:\n" + printable_board(IUB_map) + "\n\nLooking for solution...\n")
    solution = solve(IUB_map)
    print ("Here's what we found:")
    print (printable_board(solution) if solution else "None")


