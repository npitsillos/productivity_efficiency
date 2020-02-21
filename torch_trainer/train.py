import sys
import time
import torch

class Trainer():

    def __init__(self, model, num_epochs, data_loader,
                device, loss_criterion, optimizer, lr_scheduler,
                print_freq):
        self.model = model
        self.num_epochs = num_epochs
        self.data_loader = data_loader
        self.device = device
        self.print_freq = print_freq
        self.loss_criterion = loss_criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.epoch = 0
    
    def start(self):
        self.model.train()

        while self.epoch < self.num_epochs:
            for inputs, targets in self.get_inputs_targets():
                
                self.optimizer.zero_grad()
                inputs = torch.reshape(inputs, (64,1,784))
                
                outputs = self.model(inputs)

                loss = self.loss_criterion(outputs, targets)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.epoch += 1
    
    def get_inputs_targets(self):
        for obj in self.data_loader:
            yield obj