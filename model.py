import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        #self.linear1 = nn.Linear(input_size, hidden_size)
        #self.relu=nn.ReLU()
        #self.linear2 = nn.Linear(hidden_size, hidden_size)
        #self.linear3=nn.Linear(hidden_size, output_size)

        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        #self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #x=self.linear1(x)
        #x=self.relu(x)
        #x = self.linear2(x)
        #x=self.relu(x)
        #x=self.linear3(x)
        #x=self.sigmoid(x)

        x = F.relu(self.linear1(x))
        x = self.linear2(x)
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
        

    def train_step(self, state, action, reward, next_state, game_over):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long) #long
        reward = torch.tensor(reward, dtype=torch.float)
        #game_over=torch.tensor(game_over, dtype=torch.bool)
        
        #if state.ndim == 1:
        #    state=state.unsqueeze(0)
        #if next_state.ndim == 1:
        #    next_state=next_state.unsqueeze(0)
            
        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            game_over = (game_over, )
        
        output=self.model(state)
        target=output.clone()
        #print("target", target)


        #Q_new=reward+self.gamma*torch.max(self.model(next_state), dim=1)[0]* ~game_over
        #action = action.unsqueeze(1)
        #if action.shape != target.shape:
        #    action=action.reshape(target.shape)
            


        #print("action", action)
        #print("target", target)
        #print("tamaño", len(target))

        #target=target.unsqueeze(0)
        #print("target squeeze", target)
        #target.scatter_(1, action, Q_new)
        

        for idx in range(len(game_over)):
            Q_new = reward[idx]
            if not game_over[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            target[idx] = Q_new

        # 1: predicted Q values with current state
        self.optimizer.zero_grad()
        loss = self.criterion(target, output)
        loss.backward()
        self.optimizer.step()



