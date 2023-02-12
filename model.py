import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu=nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3=nn.Linear(hidden_size, output_size)
        #self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x=self.linear1(x)
        x=self.relu(x)
        x = self.linear2(x)
        x=self.relu(x)
        x=self.linear3(x)
        #x=self.sigmoid(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        #self.criterion = nn.BCELoss()
        

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.bool) #long
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)

        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )
        output=self.model(state)
        target=output.clone()
        #target=reward[:-1]+self.gamma*self.model(next_state).max()
        
        #for idx in range(len(done)):
        #    Q_new = reward[idx]
        #    if not done[idx]:
        #        Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
        #    target[idx] = Q_new
  

        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            target[idx] = Q_new

        # 1: predicted Q values with current state
        self.optimizer.zero_grad()
        loss = self.criterion(target, output)

        #loss = self.criterion(target, output)
        loss.backward()
        self.optimizer.step()



