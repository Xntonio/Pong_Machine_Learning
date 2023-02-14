import torch
import torch.nn as nn
import torch.optim as optim
import os

# Definir la arquitectura de la red neuronal
class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size,output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) # 3 entradas (posición x e y de la bola, posición y de la barra)
        self.fc2 = nn.Linear(hidden_size, output_size) # 2 salidas (probabilidad de mover la barra hacia arriba o hacia abajo)
        self.softmax = nn.Softmax(dim=0)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))
        return x


    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)
        
model = Linear_QNet

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        # Definir la función de pérdida y el optimizador
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()      

# Generar algunos datos de entrenamiento de ejemplo
# x_train = torch.randn(100, 3) # 100 ejemplos con 3 características
# y_train = torch.randint(2, (100, )) # 100 etiquetas binarias (0 o 1)
    def train_step(self, state, action, reward, next_state, done):
        # Entrenar el modelo
        state = torch.tensor(state, dtype=torch.float) #[x,y,y]
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.bool) #long [0] o [1]
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)
        
        self.optimizer.zero_grad()
        print("state",state)
        pred = self.model(state)
        print("pred",pred)
        target = pred.clone()
        
        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )
            
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            target[idx] = Q_new
        
        
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()
    
    

    
    
    
    
  