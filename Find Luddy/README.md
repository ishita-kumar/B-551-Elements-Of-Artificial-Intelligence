# Assignment 0

 ### Aim and Learning Objectives

 ```
- [x] Solve search problem statements in AI. 
- [x] Get well versed with Python
```
 
### Part 1 : Finding your way

**Problem Statement**
A map of size N x M size row and columns represents a campus. The symbol # represents your current location, @ is the location of Luddy hall (where you have to get to) and . represents  sidewalks which are free to walk on. Our goal is to find shortest path between current location (#) and Luddy Hall (@) 

**Approach**
 Find Luddy poses a search problem where one must navigate through the N x M matrix to find the goal state. The skeleton code handles most of the functions such as matrix initialization to printing the output. However it misses the core functions of actually returning a cost function and the path to get to it. Keeping this in mind, the following modifications to the skeleton code have been made:
1.  We create a **visited array** which actually checks whether a certain state in the fringe has already been traversed . Since the original code was leading us into an infinite loop. The current move is appended into the visited array which keeps note of the moves already traversed and does not repeat them in later operations. 
2. To find the path through which we can reach our goal state (5,6) we must find the path via calculating the absolute distance traveled by each row column in one hop. If row, col is our current location and the next location can  be row+1, col(move down) row-1, col(move up), row, col+1(move right) row, col -1(move left). The **sub_tup** function checks this condition,taking in curr_move and move as its arguments and measuring the absolute distance measured in each hop.

Output: (16, 'NNNEESSSEENNEESS')



```
. . . . & & & 
. & & & . . .
. . . . & . .
. & . & . . .
. & . & . & .
# & . . . & @
```

 Path Traversed: 5,0 &rarr;4,0&rarr;   3,0&rarr;  2,0&rarr;  2,1&rarr;  2,2&rarr;  3,2&rarr;  4,2&rarr;  5,2&rarr;    5,3 &rarr;5,4&rarr; 4,4&rarr;   4,5&rarr; 4,6 &rarr;5,6  

**State Space**: A matrix of N rows and M columns wherein "#" represents current state, "." represents valid state, "&" represents a blocked path which you cannot traverse, and "@" represents goal state.

**Cost Function**: Each hop (going up, down, left or right) costs one unit.

**Goal State**:  Luddy Hall (5,6).

**Successor Function**: Moving up down left or right. for our first move  (5,0), succ(5,0) = (4,0).

**Valid State**: A state which conforms to going on a "." path 



