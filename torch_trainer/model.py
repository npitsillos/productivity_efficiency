import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import torchvision
from train import Trainer

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.fc1 = nn.Linear(784, 512)
    self.fc2 = nn.Linear(512, 256)
    self.fc3 = nn.Linear(256, 50)
    self.fc4 = nn.Linear(50, 10)

  def forward(self, x):
    x = F.sigmoid(self.fc1(x))
    x = F.sigmoid(self.fc2(x))
    x = F.sigmoid(self.fc3(x))
    x = self.fc4(x)
    return F.log_softmax(x)
  
if __name__ == "__main__":
  random_seed = 1
  torch.manual_seed(random_seed)
  n_epochs = 3
  batch_size_train = 64
  batch_size_test = 1000
  learning_rate = 0.01
  momentum = 0.5
  log_interval = 10
  train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./files/', train=True, download=True,
                              transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                  (0.1307,), (0.3081,))
                              ])), batch_size=batch_size_train, shuffle=True)

  test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./files/', train=False, download=True,
                              transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                  (0.1307,), (0.3081,))
                              ])), batch_size=batch_size_test, shuffle=True)
                              
  net = Net()
  optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
  lr_scheduler = lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
  trainer = Trainer(net, 10, train_loader, torch.device('cuda'), nn.NLLLoss(),optimizer,lr_scheduler,1)
  trainer.start()