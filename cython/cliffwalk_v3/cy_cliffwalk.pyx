import numpy as np
cimport numpy as np

def exec_cliffwalk(STEPS, METHOD):
    if METHOD == 'qlearn':
        run = cliffwalk(
                steps = STEPS, 
                method = METHOD, 
                Gamma = 0.99, 
                Epsilon = 0.1, 
                Alpha = 0.05, 
                state = 36, 
                action = np.random.randint(0,4,1)[0], 
                action_value = np.zeros([48, 4], dtype = np.float64), 
                next_state = 0, 
                next_action = 0,
                now_value = 0,
                reward = np.full(48, -1, dtype = np.int64), 
                record = np.zeros([STEPS, 4], dtype = np.int64)
            )
        run.qlearn()
    
    elif METHOD == 'sarsa':
        run = cliffwalk(
                steps = STEPS, 
                method = METHOD, 
                Gamma = 0.99, 
                Epsilon = 0.1, 
                Alpha = 0.05, 
                state = 36, 
                action = np.random.randint(0,4,1)[0], 
                action_value = np.zeros([48, 4], dtype = np.float64), 
                next_state = 0, 
                next_action = 0,
                now_value = 0,
                reward = np.full(48, -1, dtype = np.int64), 
                record = np.zeros([STEPS, 5], dtype = np.int64)
            )
        run.sarsa()

    else:
        print('You entered the undefined method, please enter "qlearn" or "sarsa".')

cdef class cliffwalk():
    cdef float Gamma
    cdef float Epsilon
    cdef float Alpha
    cdef int steps
    cdef str method
    cdef int state
    cdef np.float64_t[:,:] action_value
    cdef int action
    cdef int next_state
    cdef int next_action
    cdef np.float64_t now_value
    cdef np.int64_t[:] reward
    cdef np.int64_t[:,:] record
    def __init__(self, steps, method, Gamma, Epsilon, Alpha, state, action_value, action, next_state, next_action, now_value, reward, record):
        # environment parameters
        self.Gamma = Gamma
        self.Epsilon = Epsilon
        self.Alpha = Alpha
        self.method = method
        self.steps = steps
        # learning setting
        self.state = state
        self.action_value = action_value
        self.action = action
        self.next_state = next_state
        self.next_action = next_action
        self.now_value = self.action_value[self.state, self.action]
        self.reward = reward
        self.reward[37:-1] = -100
        self.record = record

    cdef int TransMat(self):
        cdef int max_row = 4
        cdef int max_col = 12
        cdef int now_row = int(self.state/max_col)
        cdef int now_col = (self.state%max_col)
        if max_col < now_col or max_row < now_row or now_col < 0 or now_row < 0:
            print('index error')
        col = now_col
        row = now_row
        if self.action == 0 and now_row > 0:
            row -= 1
        elif self.action == 1 and now_col > 0:
            col -= 1
        elif self.action == 2 and (max_row-1) > now_row:
            row += 1
        elif self.action == 3 and (max_col-1) > now_col:
            col += 1
        self.next_state = row*max_col + col
    
    cdef int GetAction(self):
        if np.random.rand(1) >= self.Epsilon:
            if self.method == 'qlearn':
                self.action = np.argmax(self.action_value, axis = 1)[self.state]
            elif self.method == 'sarsa':
                self.next_action = np.argmax(self.action_value, axis = 1)[self.next_state]
            else:
                print('There is no method: %s' % self.method)
            # Note
            # policy = np.argmax(self.action_value, axis = 1)
            # self.action = policy[self.next_state]
        else:
            if self.method == 'qlearn':
                self.action = np.random.randint(0,4,1)[0] # for consistent the output dtype in cython
            elif self.method == 'sarsa':
                self.next_action = np.random.randint(0,4,1)[0] # for consistent the output dtype in cython
            else:
                print('There is no method: %s' % self.method)
    
    cdef int ValueUpdate(self):
        cdef np.float64_t update_value = 0
        if self.method == 'qlearn':
            update_value = self.Alpha*(self.reward[self.next_state] + self.Gamma*np.max(self.action_value[self.next_state,:]) - self.now_value)
        elif self.method == 'sarsa':
            update_value = self.Alpha*(self.reward[self.next_state] + self.Gamma*self.action_value[self.next_state, self.next_action] - self.now_value)
        else:
            print('There is no method: %s.' % self.method)
        self.action_value[self.state, self.action] = self.now_value + update_value
    
    cdef int InsertRecord(self, int now_step):
        if self.method == 'qlearn':
            self.record[now_step, 0] = self.state
            self.record[now_step, 1] = self.action
            self.record[now_step, 2] = self.reward[self.next_state]
            self.record[now_step, 3] = self.next_state
        elif self.method == 'sarsa':
            self.record[now_step, 0] = self.state
            self.record[now_step, 1] = self.action
            self.record[now_step, 2] = self.reward[self.next_state]
            self.record[now_step, 3] = self.next_state
            self.record[now_step, 4] = self.next_action

    cdef float qlearn(self):
        cdef np.float64_t epi_reward
        if self.method == 'sarsa':
            print('Wrong function, please run with sarsa() !')
        for step from 0 <= step < self.steps:
            # get next information
            self.GetAction()
            self.TransMat()
            self.InsertRecord(now_step = step)
            # update action value
            self.ValueUpdate()
            # update for next state
            self.state = self.next_state
            if self.state > 36:
                break
        # episode reward
        epi_reward = np.sum(self.record[:,2])
        return epi_reward

    cdef float sarsa(self):
        cdef np.float64_t epi_reward
        if self.method == 'qlearn':
            print('Wrong function, please run with qlearn() !')
        for step from 0 <= step < self.steps:
            # get next information
            self.TransMat()
            self.GetAction()
            self.InsertRecord(now_step = step)
            # update action value
            self.ValueUpdate()
            # update for next state
            self.state = self.next_state
            self.action = self.next_action
            if self.state > 36:
                break
        # episode reward
        epi_reward = np.sum(self.record[:,2])
        return epi_reward