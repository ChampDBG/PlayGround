import numpy as np
cimport numpy as np
cimport cython

## defint function
cdef int TransMat(int now_state, int action):
    cdef int max_row = 4
    cdef int max_col = 12
    cdef int now_row = int(now_state/max_col)
    cdef int now_col = (now_state%max_col)

    if max_col < now_col or max_row < now_row or now_col < 0 or now_row < 0:
        print('index error')

    cdef int col = now_col
    cdef int row = now_row
    if action == 0 and now_row > 0:
        row -= 1
    elif action == 1 and now_col > 0:
        col -= 1
    elif action == 2 and (max_row-1) > now_row:
        row += 1
    elif action == 3 and (max_col-1) > now_col:
        col += 1
    cdef int next_state = row*max_col + col
    return next_state

def qlearn(np.ndarray[np.float64_t, ndim=2] action_value, np.ndarray[np.int_t, ndim=1] reward, int steps, float gamma, float alpha, float epsilon):
    # initialize setting
    cdef np.ndarray[np.float64_t, ndim=2] record = np.zeros([steps, 4])
    cdef int state = 36
    for step from 0 <= step < steps:
        # get next information
        action = GetAction(action_value, epsilon, state)
        next_state = TransMat(state, action)
        record[step,:] = state, action, reward[next_state], next_state
        # update action value
        action_value[state, action] = ValueUpdate('qlearn', action_value, record[step,:], alpha, gamma)
        # update for next state
        state = next_state
        if state > 36:
            break
    # episode reward
    cdef np.int64_t epi_reward = sum(record[:,2])
    return action_value, epi_reward

def sarsa(np.ndarray[np.float64_t, ndim=2] action_value, np.ndarray[np.int_t, ndim=1] reward, int steps, float gamma, float alpha, float epsilon):
    # initialize setting
    cdef np.ndarray[np.float64_t, ndim=2] record = np.zeros([steps, 5])
    cdef int state = 36
    action = GetAction(action_value, epsilon, state)
    for step from 0 <= step < steps:
        # get next information
        next_state = TransMat(state, action)
        next_action = GetAction(action_value, epsilon, next_state)
        record[step,:] = state, action, reward[next_state], next_state, next_action
        # update action value
        action_value[state, action] = ValueUpdate('sarsa', action_value, record[step,:], alpha, gamma)
        # update for next state
        state = next_state
        action = next_action
        if state > 36:
            break
    # episode reward
    cdef np.int64_t epi_reward = sum(record[:,2])
    return action_value, epi_reward

cdef int GetAction(np.ndarray[np.float64_t, ndim=2] action_value, float epsilon, int next_state):
    if np.random.rand(1) >= epsilon:
        policy = np.argmax(action_value, axis = 1)
        action = policy[next_state]
    else:
        action = np.random.randint(0,4,1)
    return action

def ValueUpdate(str method, np.ndarray[np.float64_t, ndim=2] action_value, np.ndarray[np.float64_t, ndim=1] record, float alpha, float gamma):
    cdef int state = (<int>record[0])
    cdef int action = (<int>record[1])
    cdef int reward = (<int>record[2])
    cdef int next_state = (<int>record[3])
    cdef float now_value = action_value[state, action]
    if method == 'qlearn':
        update_value = alpha*(reward + gamma*np.max(action_value[next_state,:]) - now_value)
    elif method == 'sarsa':
        next_action = (<int>record[4])
        update_value = alpha*(reward + gamma*action_value[next_state, next_action] - now_value)
    else:
        print('No this method.')
    value = now_value + update_value
    return value

def run_cliffwalk(int episodes, str method):
    # environment setting
    cdef np.ndarray[np.float64_t, ndim=2] ActionValue = np.zeros([48, 4])
    cdef np.ndarray[np.int64_t, ndim=1] Reward = np.full(48, -1)
    Reward[37:-1] = -100
    cdef list EpisodeReward = []
    # parameters setting
    cdef float Gamma = 0.99
    cdef float Epsilon = 0.1
    cdef int Steps = 1000
    cdef float Alpha = 0.05

    # Execute
    if method == 'qlearn':
        for episode from 0 <= episode < episodes:
            ActionValue, Epi_Reward = qlearn(ActionValue, Reward, Steps, Gamma, Alpha, Epsilon)
            EpisodeReward.append(Epi_Reward)
    elif method == 'sarsa':
        for episode from 0 <= episode < episodes:
            ActionValue, Epi_Reward = sarsa(ActionValue, Reward, Steps, Gamma, Alpha, Epsilon)
            EpisodeReward.append(Epi_Reward)
    else:
        print('No this method.')
    #EpisodeReward = np.array(EpisodeReward)
    #return EpisodeReward