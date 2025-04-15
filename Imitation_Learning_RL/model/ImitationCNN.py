import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class ImitationCNN(nn.Module):
    def __init__(self):
        super(ImitationCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(5, 32, kernel_size=5, stride=2, padding=2)  
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1) 
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)

        self.fc = nn.Linear(256 * 23 * 40, 512) 
        
        # Output: steer, throttle, brake
        self.steer_head = nn.Linear(512, 1)
        self.throttle_head = nn.Linear(512, 1)  
        self.brake_head = nn.Linear(512, 1)    

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        
        x = x.view(x.size(0), -1)  
        x = F.relu(self.fc(x))
        
        steer = self.steer_head(x)
        throttle = self.throttle_head(x)
        brake = self.brake_head(x)
        
        return torch.cat([steer, throttle, brake], dim=1)
