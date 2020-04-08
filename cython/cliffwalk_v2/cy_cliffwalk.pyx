import numpy as np
cimport numpy as np

## environment and parameters setting
cdef float Gamma = 0.99
cdef float Epsilon = 0.1
cdef int Steps = 1000
cdef float Alpha = 0.05

## defint function
cdef int TransMat(int now_state, np.int64_t action):
    cdef int max_row = 4
    cdef int max_col = 12
    cdef int now_row = int(now_state/max_col)
    cdef int now_col = (now_state%max_col)

    if max_col < now_col or max_row < now_row or now_col < 0 or now_row < 0:
        print('index error')
    
    col = now_col
    row = now_row
    if action == 0 and now_row > 0:
        row -= 1
    elif action == 1 and now_col > 0:
        col -= 1
    elif action == 2 and (max_row-1) > now_row:
        row += 1
    elif action == 3 and (max_col-1) > now_col:
        col += 1
    next_state = row*max_col + col
    return next_state

def qlearn(steps = Steps, gamma = Gamma, alpha = Alpha, epsilon = Epsilon):
    # initialize setting
    cdef np.ndarray[np.float64_t, ndim=2] action_value = np.zeros([48, 4])
    cdef np.ndarray[np.int64_t, ndim=1] reward = np.full(48, -1)
    cdef np.ndarray[np.int64_t, ndim=2] record = np.zeros([steps, 4], dtype = np.int64)
    cdef int state = 36
    cdef np.int64_t action
    cdef int next_state
    reward[37:-1] = -100
    for step in range(steps):
        # get next information
        action = GetAction(action_value, epsilon, state)
        next_state = TransMat(state, action)
        record[step, :] = state, action, reward[next_state], next_state
        # update action value
        action_value[state, action] = ValueUpdate('qlearn', action_value, record[step], alpha, gamma)
        # update for next state
        state = next_state
        if state > 36:
            break
    # episode reward
    epi_reward = np.sum(record[:,2])
    return action_value, epi_reward

def sarsa(action_value, reward, steps, gamma, alpha, epsilon):
    # initialize setting
    record = []
    state = 36
    action = GetAction(action_value, epsilon, state)
    for step in range(steps):
        # get next information
        next_state = TransMat(state, action)
        next_action = GetAction(action_value, epsilon, next_state)
        record.append([state, action, reward[next_state], next_state, next_action])
        # update action value
        action_value[state, action] = ValueUpdate('sarsa', action_value, record[step], alpha, gamma)
        # update for next state
        state = next_state
        action = next_action
        if state > 36:
            break
    # episode reward
    record = np.array(record)
    epi_reward = np.sum(record[:,2])
    return action_value, epi_reward

def GetAction(np.ndarray[np.float64_t, ndim=2] action_value, float epsilon, int next_state):
    cdef np.ndarray[np.int64_t, ndim = 1] policy
    cdef np.int64_t action
    if np.random.rand(1) >= epsilon:
        policy = np.argmax(action_value, axis = 1)
        action = policy[next_state]
    else:
        action = np.random.randint(0,4,1)[0] # for consistent the output dtype
    return action

def ValueUpdate(str method, np.ndarray[np.float64_t, ndim=2] action_value, np.ndarray[np.int64_t, ndim=1] record, float alpha, float gamma):
    cdef np.int64_t state = record[0]
    cdef np.int64_t action = record[1]
    cdef np.int64_t reward = record[2]
    cdef np.int64_t next_state = record[3]
    cdef np.float64_t now_value = action_value[state, action]
    cdef np.float64_t update_value
    cdef np.float64_t value
    if method == 'qlearn':
        update_value = alpha*(reward + gamma*np.max(action_value[next_state,:]) - now_value)
        value = now_value + update_value
    elif method == 'sarsa':
        next_action = record[4]
        update_value = alpha*(reward + gamma*action_value[next_state, next_action] - now_value)
        value = now_value + update_value
    else:
        print('No this method.')
    return value