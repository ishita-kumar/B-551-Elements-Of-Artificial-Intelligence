#!/usr/local/bin/python3
#
# find_luddy.py : a simple maze solver
#
# Submitted by : Ishita Kumar (ishkumar@iu.edu)
#
# Based on skeleton code by Z. Kachwala, 2019
#

import sys
import json

# Parse the map from a given filename
def parse_map(filename):
	with open(filename, "r") as f:
		return [[char for char in line] for line in f.read().split("\n")]

# Check if a row,col index pair is on the map
def valid_index(pos, n, m):
	return 0 <= pos[0] < n  and 0 <= pos[1] < m

# Find the possible moves from position (row, col)
def moves(map, row, col):
	moves=((row+1,col), (row-1,col), (row,col-1), (row,col+1))
	# Return only moves that are within the board and legal (i.e. on the sidewalk ".")
	return [ move for move in moves if valid_index(move, len(map), len(map[0])) and (map[move[0]][move[1]] in ".@" ) ]

# Check Path direction in each hop
def sub_tup(curr_move, move):
	if [move[0]-curr_move[0],move[1]-curr_move[1]] == [1,0]:
		return 'S'
	elif [move[0]-curr_move[0],move[1]-curr_move[1]] == [-1,0]:
		return 'N'
	elif [move[0]-curr_move[0],move[1]-curr_move[1]] == [0,1]:
		return 'E'
	elif [move[0]-curr_move[0],move[1]-curr_move[1]] == [0,-1]:
		return 'W'

# Perform search on the map
def search1(IUB_map):
	# Inititalise visited array to store traversed node
	visited = []
	you_loc=[(row_i,col_i) for col_i in range(len(IUB_map[0])) for row_i in range(len(IUB_map)) if IUB_map[row_i][col_i]=="#"][0]
	fringe=[(you_loc,0, '')]
	while fringe:
		(curr_move, curr_dist, path)=fringe.pop()
		visited.append(curr_move)
		
		for move in moves(IUB_map, *curr_move):
			if move in visited:
				continue
			elif IUB_map[move[0]][move[1]]=="@":
				return curr_dist+1, path + sub_tup(curr_move, move)
			else:
				temp_path = sub_tup(curr_move, move)
				fringe.append((move, curr_dist + 1, path + temp_path))
	return False

# Main Function
if __name__ == "__main__":
	IUB_map=parse_map(sys.argv[1])
	print("Shhhh... quiet while I navigate!")
	solution = search1(IUB_map)
	print("Here's the solution I found:")
	if solution:
		print(str(solution[0]) + ' ' + solution[1])
	else:
		print("Inf")

