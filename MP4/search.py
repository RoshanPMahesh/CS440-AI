import heapq
# You do not need any other imports

def best_first_search(starting_state):
    '''
    Implementation of best first search algorithm

    Input:
        starting_state: an AbstractState object

    Return:
        A path consisting of a list of AbstractState states
        The first state should be starting_state
        The last state should have state.is_goal() == True
    '''
    # we will use this visited_states dictionary to serve multiple purposes
    # - visited_states[state] = (parent_state, distance_of_state_from_start)
    #   - keep track of which states have been visited by the search algorithm
    #   - keep track of the parent of each state, so we can call backtrack(visited_states, goal_state) and obtain the path
    #   - keep track of the distance of each state from start node
    #       - if we find a shorter path to the same state we can update with the new state 
    # NOTE: we can hash states because the __hash__/__eq__ method of AbstractState is implemented
    visited_states = {starting_state: (None, 0)}

    # The frontier is a priority queue
    # You can pop from the queue using "heapq.heappop(frontier)"
    # You can push onto the queue using "heapq.heappush(frontier, state)"
    # NOTE: states are ordered because the __lt__ method of AbstractState is implemented
    frontier = []
    heapq.heappush(frontier, starting_state)
    
    # TODO(III): implement the rest of the best first search algorithm
    # HINTS:
    #   - add new states to the frontier by calling state.get_neighbors()
    #   - check whether you've finished the search by calling state.is_goal()
    #       - then call backtrack(visited_states, state)...
    #   - you can reuse the search code from mp3...
    # Your code here ---------------
    befs = []
    distance = 1
    while frontier:
        curr_state = heapq.heappop(frontier)

        # if we've finished the search, call backtrack to get path
        if curr_state.is_goal() is True:
            befs = backtrack(visited_states, curr_state)
            return befs

        # gets all of the neighbors of the current state and checks if visited and if distance is shorter than current state
        for neighbors in curr_state.get_neighbors():
            node_dist = neighbors.dist_from_start + neighbors.h

            # updates the list of visited states
            if (neighbors not in visited_states) or ((node_dist + neighbors.compute_heuristic()) < visited_states[neighbors][distance]):
                # puts the correct distance from the start state for the neighbor
                visited_states[neighbors] = (curr_state, node_dist + neighbors.compute_heuristic())              
                heapq.heappush(frontier, neighbors)
    # ------------------------------
    
    # if you do not find the goal return an empty list
    return []

# TODO(III): implement backtrack method, to be called by best_first_search upon reaching goal_state
# Go backwards through the pointers in visited_states until you reach the starting state
# You can reuse the backtracking code from MP3
# NOTE: the parent of the starting state is None
def backtrack(visited_states, goal_state):
    path = []
    # Your code here ---------------
    curr_state = goal_state

    # iterates through the states from current state to starting state
    while curr_state is not None:
        # puts the current state at the beginning of the list so that it's in order from starting to ending state
        path.insert(0, curr_state)
        curr_state = visited_states[curr_state][0]
    # ------------------------------
    return path