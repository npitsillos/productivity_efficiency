import torch
import time
import datetime
from collections import defaultdict, deque

class Metric(object):
    
    def __init__(self, max_len=20, fmt=None):
        ''' Instantiates a Metric

            max_len: length of queue
            fmt: format of this Metric
        '''
        if fmt is None:
            fmt = "{median:.4f} ({global_average:.4f})"

        self.fmt = fmt
        
        self.deque = deque(maxlen=max_len)
        self.total = 0.0
        self.count = 0

    def update(self, value, n=1):
        ''' Updates a Metric

            value: value to append to queue
            n: number of items added
        '''
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def avg(self):
        tensor_type = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        tensor = torch.Tensor(list(self.deque)).type(tensor_type)
        return tensor.mean().item()
        
    @property
    def value(self):
        return self.deque[-1]
    
    @property
    def max(self):
        return max(self.deque)

    @property
    def median(self):
        tensor = torch.Tensor(list(self.deque))
        return tensor.median().item()
    
    @property    
    def global_average(self):
        return self.total / self.count

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_average=self.global_average,
            max=self.max,
            value=self.value)

class Logger(object):
    ''' Implements a Logger for logging the progress of an epoch
    '''

    def __init__(self):
        self.metrics = defaultdict(Metric)
        self.delimiter = "  "
        
    def update(self, **kwargs):
        for k,v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            self.metrics[k].update(v)

    def add_metric(self, name, metric):
        self.metrics[name] = metric

    def __str__(self):
        loss_str = []
        for name, meter in self.metrics.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def log(self, iterable_loader, print_freq, header=None):
        
        i = 0
        if header is None:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = Metric(fmt="{avg:.4f}")
        data_time = Metric(fmt="{avg:.4f}")
        space_fmt = ':' + str(len(str(len(iterable_loader)))) + 'd'

        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max_mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])

        MB = 1024.0 * 1024.0
        for obj in iterable_loader:
            data_time.update(time.time() - end)
            yield obj

            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable_loader) - 1:
                eta_seconds = iter_time.global_average * (len(iterable_loader) - 1)
                eta_string = str(datetime.timedelta(seconds=(int(eta_seconds))))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable_loader), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable_loader), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable_loader)))
