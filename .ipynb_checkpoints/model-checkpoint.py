import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from helper import file_hash

MODEL_DIR = os.path.join(
    os.path.realpath(os.path.dirname(__file__)),
    'models')
MODEL_FILE = os.path.join(MODEL_DIR, 'model.pt')

class LinearQNet(nn.Module):
    
    def __init__(self, n_in, n_hidden, n_out):
        super().__init__()
        self.linear1 = nn.Linear(n_in, n_hidden)
        self.linear2 = nn.Linear(n_hidden, n_out)
        self.pretrained = False
        if os.path.exists(MODEL_FILE):
            self.load()
        return
    
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
            
    def load(self):
        self.load_state_dict(torch.load(MODEL_FILE))
        model_hash = file_hash(MODEL_FILE)
        print(f'Loaded trained model [{model_hash[:16]}]')
        self.pretrained = True
        return
    
    def save(self):
        os.makedirs(MODEL_DIR, exist_ok=True)
        torch.save(self.state_dict(), MODEL_FILE)
        return
    
class QTrainer():
    
    def __init__(self, model, lr, gamma):
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.loss = nn.SmoothL1Loss()
        return
    
    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']
    
    def train_step(self, state_old, action, reward, state_new, done):
        state_old = torch.tensor(state_old, dtype=torch.float)
        state_new = torch.tensor(state_new, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        
        if len(state_old.shape) == 1:
            state_old = state_old.unsqueeze(0)
            state_new = state_new.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            done = (done,)
            
        # Get predicted Q values with current state
        Q_old = self.model(state_old)
        
        # reward + y * max(next predicted Q value)
        target = Q_old.clone().detach()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new += self.gamma * torch.max(self.model(state_new[idx]))
            target[idx][action.argmax().item()] = Q_new
        
        self.optimizer.zero_grad()
        loss = self.loss(target, Q_old)
        loss.backward()
        self.optimizer.step()
        return
    
    def update_lr(self, new_lr):
        for g in self.optimizer.param_groups:
            g['lr'] = new_lr
        return
