The goal of the algorithm is to assemble a team of robots under a given budget in such a way that the skills are maximized. The skeleton code returned the most optimal solution, however it considered fractions of the robots, instead of whole robots.This was similar to fractional knapsack, a greedy approach.The easiest way to change this would be to break out of the loop for when the skill/profit value would not b a whole value but this would'nt return the optimal solution. Our solution makes use of a branch and bound 0/1 knapsack algorithm, and a brief abstraction of the algorithm is given below, -

**Search Space:** Collection of robots with their skills and weights. - **Successor Function:** Each successor function would have robots entry with the most optimal skill/weight ratio - 
**Cost Function:** The cost function is a value that determines the upper bound and max profit which examines the decision tree and returns the optimal solution -
 **Goal State:** An optimal list which maximizes skills within the given budget - 
**Initial State:** An empty set In *approx_solve*, We create a decision tree by sorting our names, skill, cost in a non decreasing ratio of skills/cost. *approx_solve* implements a queue called fringe.  Empty at first,the Fringe traverses the nodes of the decision tree. We first initialize the profit and weight `initial_node.profit = initial_node.weight = 0`. *new_node* stores the key values of children nodes. For every node in *new_node*, we compute the profit and bound value.The branch and bound methodology is essentially branching or rather dividing the given sub-problem into smaller problems and then formulating an upper bound to the optimal answer in every node. The branch and bound method implements a greedy solution to find out this upper bound on the current node. The get_bound method gets the `max_profit` value of every node in *new_node*. If profit is more than the current `max_profit` value then we store the `max_profit = profit`. If bound value is higher than the max profit value then `max_profit = bound`. This ensures that after checking at each level we only consider the branch with the most optimal skill/cost ratio. Any value above the `max_profit` value will not be considered, thus the decision tree is pruned at that stage.