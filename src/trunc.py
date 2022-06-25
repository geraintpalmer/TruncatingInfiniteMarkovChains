import numpy as np

def find_steady_state_probs_DTMC(transition_matrix, state_space):
    """
    Finds the steady state probabilities by solving:
    \pi P = \pi
    \sum \pi = 1
    """
    size_mat = len(state_space)
    A = np.append(transition_matrix.transpose() - np.identity(size_mat), np.ones((1, size_mat)), axis=0)
    b = np.hstack((np.transpose(np.zeros(3)), [1]))
    sol = np.linalg.solve(np.transpose(A).dot(A), np.transpose(A).dot(b))
    probs = {state_space[i]: sol[i] for i in range(size_mat)}
    return probs

def find_steady_state_probs_CTMC(transition_rate_matrix, state_space):
    """
    Finds the steady state probabilities by solving:
    \pi Q = 0
    \sum \pi = 1
    """
    size_mat = len(state_space)
    A = np.vstack((transition_rate_matrix.transpose()[:-1], np.ones(size_mat)))
    b = np.vstack((np.zeros((size_mat - 1, 1)), [1]))
    sol = np.linalg.solve(A, b).transpose()[0]
    probs = {state_space[i]: sol[i] for i in range(size_mat)}
    return probs

def find_mean_time_to_absorption_CTMC(transition_rate_matrix, state_space):
    """
    Finds the mean time to absorption from each state
    """
    Q = transition_rate_matrix.copy()
    absorbing_states = [si for si, s in enumerate(state_space) if np.all(Q[si]==0)]
    transient_states = [si for si, s in enumerate(state_space) if si not in absorbing_states]
    n_transient = len(transient_states)
    time2absorb = np.linalg.solve(-Q[:, transient_states][transient_states], np.ones(n_transient))
    mean_times_to_absorption = {state_space[s]: time2absorb[si] for si, s in enumerate(transient_states)}
    return mean_times_to_absorption

def find_mean_time_to_absorption_DTMC(transition_matrix, state_space):
    """
    Finds the mean time to absorption from each state
    """
    P = transition_matrix.copy()
    absorbing_states = [si for si, s in enumerate(state_space) if np.all(P[si, si]==1)]
    transient_states = [si for si, s in enumerate(state_space) if si not in absorbing_states]
    n_transient = len(transient_states)
    time2absorb = np.linalg.solve(np.identity(n_transient)-P[:, transient_states][transient_states], np.ones(n_transient))
    mean_times_to_absorption = {state_space[s]: time2absorb[si] for si, s in enumerate(transient_states)}
    return mean_times_to_absorption

def discretise_transition_rate_matrix(transition_rate_matrix):
    """
    Discretises a CTMC rate matrix to a DTMC probabilities matrix
    """
    size_mat = len(transition_rate_matrix)
    time_step = 1 / (-transition_rate_matrix.diagonal()).max()
    transition_matrix = transition_rate_matrix * time_step + np.identity(size_mat)
    return transition_matrix

def find_hitting_probs(transition_matrix, state_space, state):
    """
    Finds the probability of ever reaching a transient state `state' from every transient state.
    """
    P = transition_matrix.copy()
    n_states = len(state_space)
    i = state_space.index(state)
    absorbing_states = [si for si, s in enumerate(state_space) if P[si, si]==1.0]
    transient_states = [si for si, s in enumerate(state_space) if si not in absorbing_states]
    it = transient_states.index(i)
    n_transient = len(transient_states)
    P[i] = np.zeros(n_states)
    A = P[:, transient_states][transient_states] - np.identity(n_transient)
    b = np.zeros(n_transient)
    b[it] = -1
    p = np.linalg.solve(A, b)
    hitting_probabilities = {state_space[s]: p[si] for si, s in enumerate(transient_states)}
    return hitting_probabilities

def make_mm1_matrix(arrival_rate, service_rate, bound):
    """
    Makes the discretised transition matrix for an M/M/1 queue,
    but there is a chance of closure (absorbtion) when the queue is empty.
    """
    transition_matrix = np.zeros((bound, bound))
    for s1 in range(bound):
        for s2 in range(bound):
            delta = s2 - s1
            if delta == 1:
                transition_matrix[s1, s2] = arrival_rate
            if delta == -1:
                transition_matrix[s1, s2] = service_rate
    transition_matrix = transition_matrix -(np.identity(bound) * transition_matrix.sum(axis=1))
    return transition_matrix

def make_modified_mm1_matrix(arrival_rate, service_rate, closure_rate, bound):
    """
    Makes the discretised transition matrix for an M/M/1 queue,
    but there is a chance of closure (absorbtion) when the queue is empty.
    """
    transition_matrix = np.zeros((bound, bound))
    for s1 in range(bound):
        for s2 in range(bound):
            delta = s2 - s1
            if delta == 1:
                transition_matrix[s1, s2] = arrival_rate
            if delta == -1:
                transition_matrix[s1, s2] = service_rate
    new_column = np.hstack(([closure_rate], np.zeros(bound - 1)))
    transition_matrix = np.c_[transition_matrix, new_column]
    transition_matrix = np.vstack((transition_matrix, np.zeros(bound + 1)))
    transition_matrix = transition_matrix -(np.identity(bound + 1) * transition_matrix.sum(axis=1))
    return transition_matrix

def make_gamblers_ruin_matrix(winning_prob, bound):
    """
    Makes the transition matrix for gamblers ruin:
      - A gambler plays a game with a given chance of winning
      - They bet `state', and their loose everything, or win +1
      - The gambler bets everything they have every time
    """
    transition_matrix = np.zeros((bound, bound))
    for s1 in range(bound):
        for s2 in range(bound):
            delta = s2 - s1
            if s1 == s2 == 0:
                transition_matrix[s1, s2] = 1.0
            elif s1 != 0:
                if delta == 1:
                    transition_matrix[s1, s2] = winning_prob
                if delta == -s1:
                    transition_matrix[s1, s2] = 1 - winning_prob
                if s2 == bound - 1 and s1 in [bound-2, bound-1]:
                    transition_matrix[s1, s2] = winning_prob
    return transition_matrix

def make_random_walk_matrix(prob_step_away, bound):
    """
    Makes the transition matrix for an absorbing random walk:
      - Take a step away from the origin with probability `prob_going_away`
      - Take a step closer to the origin with probability 1 - `prob_going_away`
      - Stop once you return to the origin.
    """
    transition_matrix = np.zeros((bound, bound))
    transition_matrix[0, 0] = 1
    for s in range(1, bound - 1):
        transition_matrix[s, s-1] = 1 - prob_step_away
        transition_matrix[s, s+1] = prob_step_away
    transition_matrix[-1][-2] = 1
    return transition_matrix
