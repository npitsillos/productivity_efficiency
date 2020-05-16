import sys
import time
import torch

from utils import Logger
from evaluator import Evaluator

class Trainer():

    def __init__(self, model, num_epochs, train_loader, val_loader,
                device, loss_criterion, optimizer, print_freq):
        self.model = model
        self.num_epochs = num_epochs
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.print_freq = print_freq
        self.loss_criterion = loss_criterion
        self.optimizer = optimizer
        self.epoch = 0
        self.logger = Logger()
    
    def train_model(self):
        self.model.to(self.device)
        while self.epoch < self.num_epochs:
            self.model.train()

            for inputs, targets in self.logger.log(self.train_loader, self.print_freq, "Epoch: [{}]".format(self.epoch)):
                
                self.optimizer.zero_grad()
                
                targets = targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.loss_criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                self.logger.update(loss=loss.cpu().detach().item()/len(inputs))
                self.logger.update(lr=self.optimizer.param_groups[0]["lr"])
            self.eval_model()
            self.epoch += 1
        
    def eval_model(self):
        self.model.eval()
        # Evaluate model
        self.evaluator = Evaluator(self.loss_criterion)
        with torch.no_grad():
            for inputs, targets in self.logger.log(self.val_loader, self.print_freq, "Test:"):

                model_time = time.time()
                outputs = self.model(inputs)

                model_time = time.time() - model_time

                evaluator_time = time.time()
                self.evaluator.update(targets, outputs)
                evaluator_time = time.time() - evaluator_time

                self.logger.update(model_time=model_time, evaluator_time=evaluator_time)
        
        self.evaluator.log()