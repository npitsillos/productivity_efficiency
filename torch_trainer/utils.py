from collections import defaultdict

class Logger(object):
    ''' Implements a Logger for logging the progress of an epoch
    '''

    def log_epoch(self, iterable_loader, print_freq, epoch):
        
        i = 0
        for obj in iterable_loader:
            
            yield obj

            if i % print_freq == 0:
                header = "Epoch: [{}] [{}/{}]".format(epoch, i, len(iterable_loader))
                print(header)
            i += 1