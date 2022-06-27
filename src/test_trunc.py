import numpy as np
import trunc

def test_find_steady_state_probs_DTMC():
    """
    Example from Stewart book (pp.205):
    
    P = (
    (0.8,   0.15,    0.05),
    (0.7,   0.2,     0.1 ),
    (0.5,   0.3,     0.2 )
    )
    
    pi = (0.76250, 0.16875, 0.06875)
    """
    P = np.array(
        [
            [0.8, 0.15, 0.05],
            [0.7, 0.2, 0.1],
            [0.5, 0.3, 0.2]
        ]
    )
    pi = trunc.find_steady_state_probs_DTMC(
        transition_matrix=P,
        state_space=range(3)
    )
    expected_pi = (0.76250, 0.16875, 0.06875)
    assert round(pi[0], 5) == expected_pi[0]
    assert round(pi[1], 5) == expected_pi[1]
    assert round(pi[2], 5) == expected_pi[2]


def test_find_steady_state_probs_CTMC():
    """
    Example from Stewart book (pp.264):
    
    Q = (
    (-4,   4,   0,   0)
    ( 3,  -6,   3,   0)
    ( 0,   2,  -4,   2)
    ( 0,   0,   1,  -1)
    )
    
    pi = (0.12, 0.16, 0.24, 0.48)
    """
    Q = np.array(
        [
            [-4, 4, 0, 0],
            [3, -6, 3, 0],
            [0, 2, -4, 2],
            [0, 0, 1, -1]
        ]
    )
    pi = trunc.find_steady_state_probs_CTMC(
        transition_rate_matrix=Q,
        state_space=range(4)
    )
    expected_pi = (0.12, 0.16, 0.24, 0.48)
    assert round(pi[0], 5) == expected_pi[0]
    assert round(pi[1], 5) == expected_pi[1]
    assert round(pi[2], 5) == expected_pi[2]
    assert round(pi[3], 5) == expected_pi[3]


def test_find_mean_time_to_absorption_CTMC():
    """
    Example from Stewart book (pp.261):
    
    Q = (
    (-4,  4,  0,  0),
    ( 3, -6,  2,  1),
    ( 0,  2, -4,  2),
    ( 0,  0,  0,  0)
    )
    
    nu = (1.375, 1.125, 0.8125)
    """
    Q = np.array(
        [
            [-4, 4, 0, 0],
            [3, -6, 2, 1],
            [0, 2, -4, 2],
            [0, 0, 0, 0]
        ]
    )
    nu = trunc.find_mean_time_to_absorption_CTMC(
        transition_rate_matrix=Q,
        state_space=range(4)
    )
    expected_nu = (1.375, 1.125, 0.8125)
    assert round(nu[0], 5) == expected_nu[0]
    assert round(nu[1], 5) == expected_nu[1]
    assert round(nu[2], 5) == expected_nu[2]


def test_find_mean_time_to_absorption_DTMC():
    """
    Example from Stewart book (pp.221--223, modified, slightly by amalgamated many absorbing states):
    
    P = (
    (0.4, 0.2, 0.0, 0.4),
    (0.3, 0.3, 0.0, 0.4),
    (0.0, 0.0, 0.1, 0.9),
    (0.0, 0.0, 0.0, 1.0)
    )
    
    nu = (2.5, 2.5, 10/9)
    """
    P = np.array(
        [
            [0.4, 0.2, 0.0, 0.4],
            [0.3, 0.3, 0.0, 0.4],
            [0.0, 0.0, 0.1, 0.9],
            [0.0, 0.0, 0.0, 1.0]
        ]
    )
    nu = trunc.find_mean_time_to_absorption_DTMC(
        transition_matrix=P,
        state_space=range(4)
    )
    expected_nu = (2.5, 2.5, 1.11111)
    assert round(nu[0], 5) == expected_nu[0]
    assert round(nu[1], 5) == expected_nu[1]
    assert round(nu[2], 5) == expected_nu[2]


def test_discretise_transition_rate_matrix():
    """
    Example from Stewart book (pp.286):

    Q = (
    (-4,   4,   0,   0),
    ( 3,  -6,   3,   0),
    ( 0,   2,  -4,   2),
    ( 0,   0,   1,  -1)
    )

    P = (
    (1/3,  2/3,    0,    0),
    (1/2,    0,  1/2,    0),
    (  0,  1/3,  1/3,  1/3),
    (  0,    0,  1/6,  5/6)
    )
    """
    Q = np.array(
        [
            [-4,   4,   0,  0],
            [ 3,  -6,   3,  0],
            [ 0,   2,  -4,  2],
            [ 0,   0,   1, -1],
        ]
    )
    expected_P = np.array(
        [
            [1/3, 2/3, 0.0, 0.0],
            [1/2, 0.0, 1/2, 0.0],
            [0.0, 1/3, 1/3, 1/3],
            [0.0, 0.0, 1/6, 5/6],
        ]
    )
    P = trunc.discretise_transition_rate_matrix(
        transition_rate_matrix=Q,
    )
    assert np.allclose(P, expected_P)


def test_find_hitting_probs():
    """
    Made up example based on the maths of https://youtu.be/edTup9lQU90, but now treating transient states as absorbing states:
    
    P = (
      (1/5,  1/5,  1/5,  2/5,  0  ,  0  )
      (0  ,  1  ,  0  ,    0,  0  ,  0  )
      (0  ,  1/3,  0  ,  1/3,  1/3,  0  )
      (0  ,  0  ,  0  ,    1,  0  ,  0  )
      (1/2,  0  ,  0  ,    0,  0  ,  1/2)
      (0  ,  0  ,  1/2,  1/4,  1/4,  0  )
    )

    + There are two abosrbing states, state 1 and 3, so h_{10} = h_{30} = 0 by definition.
    + For $h_{00}$ we are already at state 0, so guaranteed hit, h_{00} = 1
    + To find h_{20}, h_{40}, and h_{50} we solve:
      
      h_{20} &= (1/3)h_{10} + (1/3)h_{30} + (1/3)h_{40}
      h_{40} &= (1/2)h_{00} + (1/2)h_{50}
      h_{50} &= (1/2)h_{20} + (1/4)h_{30} + (1/4)h_{40}
    
      which simplifies to:
      
      h_{20} &= (1/3)h_{40}
      h_{40} &= (1/2) +(1/2)h_{50}
      h_{50} &= (1/2)h_{40} +(1/4)h_{40}
      
    + This gives
      - h_{20} = 4/19 = 0.210526
      - h_{40} = 12/19 = 0.631579
      - h_{50} = 5/19 = 0.263158
    """
    P = np.array(
        [
            [1/5, 1/5, 1/5, 2/5, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1/3, 0.0, 1/3, 1/3, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [1/2, 0.0, 0.0, 0.0, 0.0, 1/2],
            [0.0, 0.0, 1/2, 1/4, 1/4, 0.0]
        ]
    )
    probs = trunc.find_hitting_probs(P, range(6), 0)
    assert round(probs[0], 6) == 1
    assert round(probs[2], 6) == 0.210526
    assert round(probs[4], 6) == 0.631579
    assert round(probs[5], 6) == 0.263158


def test_make_mm1_matrix():
    """
    With arrival_rate = 5, service_rate = 7, and a boundary of 4, we should get:

    Q = (
    (-5,   5,   0,   0),
    ( 7, -12,   5,   0),
    ( 0,   7, -12,   5),
    ( 0,   0,   7,  -7)
    )
    """
    expected_Q = np.array(
        [
            [-5, 5, 0, 0],
            [7, -12, 5, 0],
            [0, 7, -12, 5],
            [0, 0, 7, -7]
        ]
    )
    Q = trunc.make_mm1_matrix(
        arrival_rate=5,
        service_rate=7,
        bound=4
    )
    assert (Q == expected_Q).all()


def test_make_modified_mm1_matrix():
    """
    With arrival_rate = 5, service_rate = 7, closure rate = 3, and a boundary of 4, we should get:

    Q = (
    (-8,   5,   0,  0,  3),
    ( 7, -12,   5,  0,  0),
    ( 0,   7, -12,  5,  0),
    ( 0,   0,   7, -7,  0),
    ( 0,   0,   0,  0,  0)
    )
    """
    expected_Q = np.array(
        [
            [-8,   5,   0,  0,  3],
            [ 7, -12,   5,  0,  0],
            [ 0,   7, -12,  5,  0],
            [ 0,   0,   7, -7,  0],
            [ 0,   0,   0,  0,  0]
        ]
    )
    Q = trunc.make_modified_mm1_matrix(
        arrival_rate=5,
        service_rate=7,
        closure_rate=3,
        bound=4
    )
    assert (Q == expected_Q).all()


def test_make_gamblers_ruin_matrix():
    """
    With p=0.8, and a boundary of 6, we should get:

    P = (
    (1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    (0.2, 0.0, 0.8, 0.0, 0.0, 0.0),
    (0.2, 0.0, 0.0, 0.8, 0.0, 0.0),
    (0.2, 0.0, 0.0, 0.0, 0.8, 0.0),
    (0.2, 0.0, 0.0, 0.0, 0.0, 0.8),
    (0.2, 0.0, 0.0, 0.0, 0.0, 0.8)
    )
    """
    expected_P = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.2, 0.0, 0.8, 0.0, 0.0, 0.0],
            [0.2, 0.0, 0.0, 0.8, 0.0, 0.0],
            [0.2, 0.0, 0.0, 0.0, 0.8, 0.0],
            [0.2, 0.0, 0.0, 0.0, 0.0, 0.8],
            [0.2, 0.0, 0.0, 0.0, 0.0, 0.8]
        ]
    )
    P = trunc.make_gamblers_ruin_matrix(
        winning_prob=0.8,
        bound=6
    )
    assert np.allclose(P, expected_P)


def test_make_random_walk_matrix():
    """
    With p=0.2, and a boundary of 6, we should get:

    P = (
    (1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    (0.8, 0.0, 0.2, 0.0, 0.0, 0.0),
    (0.0, 0.8, 0.0, 0.2, 0.0, 0.0),
    (0.0, 0.0, 0.8, 0.0, 0.2, 0.0),
    (0.0, 0.0, 0.0, 0.8, 0.0, 0.2),
    (0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    )
    """
    expected_P = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.8, 0.0, 0.2, 0.0, 0.0, 0.0],
            [0.0, 0.8, 0.0, 0.2, 0.0, 0.0],
            [0.0, 0.0, 0.8, 0.0, 0.2, 0.0],
            [0.0, 0.0, 0.0, 0.8, 0.0, 0.2],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        ]
    )
    P = trunc.make_random_walk_matrix(
        prob_step_away=0.2,
        bound=6
    )
    assert np.allclose(P, expected_P)
