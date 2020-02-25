import sys
import time
import torch

from utils import Logger

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
        self.logger = Logger()
    
    def start(self):
        self.model.train()
        self.model.to(self.device)
        while self.epoch < self.num_epochs:
            for inputs, targets in self.logger.log_epoch(self.data_loader, self.print_freq, self.epoch):
                
                self.optimizer.zero_grad()
                
                targets = targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.loss_criterion(outputs, targets)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.lr_scheduler.step()
            self.epoch += 1