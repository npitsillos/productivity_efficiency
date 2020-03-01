
class Evaluator(object):
    
    def __init__(self, eval_method):
        self.eval_method = eval_method

        self.total_loss = 0.0

    def update(self, targets, predictions):
        
        self.total_loss += self.eval_method(predictions, targets)
    
    def log(self):
        print("Total loss: {}".format(self.total_loss))