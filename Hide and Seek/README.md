# Assignment 0

 ### Aim and Learning Objectives

 ```
- [x] Solve search problem statements in AI. 
- [x] Get well versed with Python
```
 

### Part 2: Hide-and-seek


**Problem Statement**
A map of size N x M size row and columns represents a campus. There exist  finite number of friends K which need to be placed in a a manner such that no friends can see each other. Buildings (&) restrict the view while "." are roads which are openly visible however each row/column has a different lane.

**Approach**
Hide and seek is a search problem wherein we must arrange K number of friends such that none of them see each other. Again, the skeleton code contains enough to get us by the initial tasks of displaying our parsing the map, placing K number of friends and displaying them in a human friendly manner. The following modifications have been done to the existing code base: 
1.  We create a **visited array** which checks if the successor function has previously been through. A successor function is only valid if it follows going through the sidewalk or the road. The fringe value  contains the successor functions of the current move.  One pops this fringe value iteratively on visiting it, and simultaneously appends it into the visited array. If the successor function is in the visited array, it is not visited again. 
2. Just checking whether a state has already been visited doesn't suffice to satisfy our goal i.e Having no friends with-in each others eye-sight.  We must check  every time we place a Friend  into a row,column whether there exists a friend nearby. **Valid_neighbors** checks the neighbor above, below, left and right for the & and F characters. It returns a false value for when an "F" is discovered in the problem set.

The output state for K=9 Friends
```
. F . . & & &
. & & & F . .
. . . F & . .
F & . & . F .
. & F & F & F
# & . F . & @
```

**State Space**: A blank matrix of N rows and M columns

**Cost Function**: At first, I wasn't really sureof the cost path. AFter reading more about the cost path for N-queens problem space, and referrring to slide number 5 of [This presentation](https://www.cs.cmu.edu/~ggordon/780-spring09/slides/Search%20I.pdf) in an abstract way , it means to have an efficient solution wherein cost path is 0

**Goal State**:  Placing a set of K friends such that none of them are able to see each other.

**Successor Function**: Placing the K'th friend on position row,col

**Valid State**: A state in which F does not have another F in eye sight. i.e two F's are not in the same row or column without an "&"  in between them

**Code Analysis:**


| N | Time To run |
| :----: | :---: |
|8 | 0.0004 s|
|9|2 s|
|10|193 s|

### Challenges:
The loop runs slowly for n=10 and returns the solution None after 190 seconds. I was unsure of what states could directly be pruned such that the successor function would not consider them. I also tried to changed the way each valid state is checked (Going in the opposite direction) but that did not sseem to make a difference to the final output.
