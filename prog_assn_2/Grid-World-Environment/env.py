from grid_world import GridWorld
import numpy as np

def create_standard_grid(start_state : np.ndarray = np.array([[0,4]]), transition_prob: float = 0.7, wind: bool = False):
    # Let's specify world parameters
    
    # size of the grid
    num_cols = 10
    num_rows = 10

    # coordinates of the cells representing obstruction
    obstructions = np.array([[0,7],[1,1],[1,2],[1,3],[1,7],[2,1],[2,3],
                            [2,7],[3,1],[3,3],[3,5],[4,3],[4,5],[4,7],
                            [5,3],[5,7],[5,9],[6,3],[6,9],[7,1],[7,6],
                            [7,7],[7,8],[7,9],[8,1],[8,5],[8,6],[9,1]])

    # list of bad states
    bad_states = np.array([[1,9],[4,2],[4,4],[7,5],[9,9]])

    # list of restart states
    restart_states = np.array([[3,7],[8,2]])

    # starting position
    start_state = start_state

    # list of goal states i.e. terminal states
    goal_states = np.array([[0,9],[2,2],[8,7]])

    # create the environment's model
    gw = GridWorld(num_rows=num_rows,
                num_cols=num_cols,
                start_state=start_state,
                goal_states=goal_states,
                wind = wind)
    
    gw.add_obstructions(obstructed_states=obstructions,
                        bad_states=bad_states,
                        restart_states=restart_states)
    
    gw.add_rewards(step_reward=-1,
                goal_reward=10,
                bad_state_reward=-6,
                restart_state_reward=-100)
    
    gw.add_transition_probability(p_good_transition=transition_prob,
                                bias=0.5)

    return gw.create_gridworld()

    


def create_four_room(start_state : np.ndarray = np.array([[8,0]]),goal_change : bool = True,transition_prob: float = 1.0):
    '''
    Creates a four-room gridworld environment.

    Inputs
    ------
    start_state : np.ndarray
        The starting state of the agent. Default is (8, 0).
    goal_change : bool
        Whether the goal state changes over time. Default is True.
    transition_prob : float
        The probability of transitioning to the intended state. Default is 1.0.

    Outputs
    -------
    gw : GridWorld
        The created gridworld environment.

    '''
    rows = 9
    cols = 9

    obstructions = np.array([
        [0,4],[2,4],[3,4],[4,4],[5,4],[6,4],[8,4],
        [5,0],[5,1],[5,3],[3,5],[3,7],[3,8],
        [2,1],[2,2],[3,2],[6,6],[6,7],[5,6]
    ])

    bad_states = np.array([[8,3]])

    restart_states = np.array([[2,3],[4,8]])

    goal_states = np.array([[0,8],[0,3],[6,8]])

    gw = GridWorld(num_rows=rows,
                num_cols=cols,
                start_state=start_state,
                goal_states=goal_states,
                goal_change = goal_change,
                wind = False,
                env = 'four_room')

    gw.add_obstructions(obstructed_states=obstructions,
                        bad_states=bad_states,
                        restart_states=restart_states)
    
    gw.add_rewards(step_reward=-1,
                goal_reward=10,
                bad_state_reward=-6,
                restart_state_reward=-100)
    
    gw.add_transition_probability(p_good_transition=transition_prob,
                                bias=0.5)

    return gw.create_gridworld()


