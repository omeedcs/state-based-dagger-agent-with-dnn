import torch

class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__() 
        self.linOne = torch.nn.Linear(11, 16)
        self.linTwo = torch.nn.Linear(16, 32)
        self.linThree = torch.nn.Linear(32, 64)
        self.linFour = torch.nn.Linear(64, 128)
        self.linFive = torch.nn.Linear(128, 256)
        self.linSix = torch.nn.Linear(256, 3)

        self.net = torch.nn.Sequential(
          self.linOne,
          torch.nn.ReLU(),
          self.linTwo,
          torch.nn.ReLU(),
          self.linThree,
          torch.nn.Dropout(.25),
          torch.nn.ReLU(),
          self.linFour,
          torch.nn.Dropout(.25),
          torch.nn.ReLU(),
          self.linFive,
          torch.nn.Dropout(.25),
          torch.nn.ReLU(),
          self.linSix)

        self.skip = torch.nn.Linear(11, 3)

        self.init_weight()
        
    def init_weight(self):
        for name, param in self.named_parameters():
            if 'bias' in name:
             torch.nn.init.constant_(param, 0.0)
            elif 'weight' in name:
             torch.nn.init.xavier_normal_(param)

    def forward(self, x):
      return self.net(x) + self.skip(x)



# 1. The initial architecture with 4 linear layers. -> failed.
# 2. We added dropout. -> failed. 
# 3. Attempted batch normalization. -> failed. 
# 4. Residual Layer