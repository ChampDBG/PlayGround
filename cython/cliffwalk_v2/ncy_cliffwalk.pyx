import numpy as np

## environment and parameters setting
Gamma = 0.99
Epsilon = 0.1
Steps = 1000
Alpha = 0.05

## defint function
def TransMat(now_state, action):
    max_row = 4
    max_col = 12
    now_row = int(now_state/max_col)
    now_col = (now_state%max_col)

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
    action_value = np.zeros([48, 4])
    reward = np.full(48, -1)
    reward[37:-1] = -100

    record = []
    state = 36
    for step in range(steps):
        # get next information
        action = GetAction(action_value, epsilon, state)
        next_state = TransMat(state, action)
        record.append([state, action, reward[next_state], next_state])
        # update action value
        action_value[state, action] = ValueUpdate('qlearn', action_value, record[step], alpha, gamma)
        # update for next state
        state = next_state
        if state > 36:
            break
    # episode reward
    record = np.array(record)
    epi_reward = np.sum(record[:,2])
    return action_value, epi_reward

def sarsa(steps = Steps, gamma = Gamma, alpha = Alpha, epsilon = Epsilon):
    # initialize setting
    action_value = np.zeros([48, 4])
    reward = np.full(48, -1)
    reward[37:-1] = -100
    
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

def GetAction(action_value, epsilon, next_state):
    if np.random.rand(1) >= epsilon:
        policy = np.argmax(action_value, axis = 1)
        action = policy[next_state]
    else:
        action = np.random.randint(0,4,1)
    return action

def ValueUpdate(method, action_value, record, alpha, gamma):
    state = record[0]
    action = record[1]
    reward = record[2]
    next_state = record[3]
    now_value = action_value[state, action]
    if method == 'qlearn':
        update_value = alpha*(reward + gamma*np.max(action_value[next_state,:]) - now_value)
    elif method == 'sarsa':
        next_action = record[4]
        update_value = alpha*(reward + gamma*action_value[next_state, next_action] - now_value)
    else:
        print('No this method.')
    value = now_value + update_value
    return value