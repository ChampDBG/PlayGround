import numpy as np

class cliffwalk():
    def __init__(self, Steps, Method):
        # environment parameters
        self.Gamma = 0.99
        self.Epsilon = 0.1
        self.Alpha = 0.05
        self.steps = Steps
        self.method = Method
        # learning setting
        self.state = 36
        self.action_value = np.zeros([48, 4])
        self.action = np.random.randint(0,4,1)[0]
        self.next_state = 0
        self.now_value = self.action_value[self.state, self.action]
        self.reward = np.full(48, -1)
        self.reward[37:-1] = -100
        if self.method == 'qlearn':
            self.record = np.zeros([self.steps, 4])
        elif self.method == 'sarsa':
            self.record = np.zeros([self.steps, 5])
        else:
            print('There is no method: %s' % self.method)

    def TransMat(self):
        max_row = 4
        max_col = 12
        now_row = int(self.state/max_col)
        now_col = (self.state%max_col)
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
    
    def GetAction(self):
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
    
    def ValueUpdate(self):
        if self.method == 'qlearn':
            update_value = self.Alpha*(self.reward[self.next_state] + self.Gamma*np.max(self.action_value[self.next_state,:]) - self.now_value)
        elif self.method == 'sarsa':
            update_value = self.Alpha*(self.reward[self.next_state] + self.Gamma*self.action_value[self.next_state, self.next_action] - self.now_value)
        else:
            print('There is no method: %s.' % self.method)
        self.action_value[self.state, self.action] = self.now_value + update_value
    
    def qlearn(self):
        if self.method == 'sarsa':
            print('Wrong function, please run with sarsa() !')
        for step in range(self.steps):
            # get next information
            self.GetAction()
            self.TransMat()
            self.record[step, :] = self.state, self.action, self.reward[self.next_state], self.next_state
            # update action value
            self.ValueUpdate()
            # update for next state
            self.state = self.next_state
            if self.state > 36:
                break
            # episode reward
        epi_reward = np.sum(self.record[:,2])
        return self.action_value, epi_reward

    def sarsa(self):
        if self.method == 'qlearn':
            print('Wrong function, please run with qlearn() !')
        for step in range(self.steps):
            # get next information
            self.TransMat()
            self.GetAction()
            self.record[step, :] = self.state, self.action, self.reward[self.next_state], self.next_state, self.next_action
            # update action value
            self.ValueUpdate()
            # update for next state
            self.state = self.next_state
            self.action = self.next_action
            if self.state > 36:
                break
        # episode reward
        epi_reward = np.sum(self.record[:,2])
        return self.action_value, epi_reward