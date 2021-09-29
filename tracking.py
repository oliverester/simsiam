from collections import deque, defaultdict
import torch.distributed as dist
import torch
import time, datetime


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    @ heavily adapted from up-detr  
    """

    def __init__(self, window_size=20, to_tensorboard=True, type='avg'):
       
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.type = type
        self.to_tensorboard = to_tensorboard

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        Warning OE: can only be called once. After that global avg might be compromised due to multiple "sum-reduces"
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]
    
    @property
    def log(self):
         return round(getattr(self, self.type),2)
    
    def __str__(self):
        return f"{self.type}: {round(getattr(self, self.type),2)}"
        # return self.fmt.format(
        #     median=self.median,
        #     avg=self.avg,
        #     global_avg=self.global_avg,
        #     max=self.max,
        #     value=self.value)


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True



class MetricLogger(object):
    """
    @ heavily adapted from up-detr  
    """
    def __init__(self, delimiter="\t", tensorboard_writer=None):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.tensorboard_writer = tensorboard_writer

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int, tuple))
            # if tuple - 2nd elment is count value
            if isinstance(v, tuple):
                assert(len(v) == 2)
                if isinstance(v[0], torch.Tensor):
                    v = list(v)
                    v[0] = v[0].item()
                self.meters[k].update(v[0], v[1])
            else:    
                self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        max_metrics = 3
        counter = 0
        for name, meter in self.meters.items():
            if counter == max_metrics:
                break
            counter += 1
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter
        
    def send_meters_to_tensorboard(self, step):
        if self.tensorboard_writer is None:
            Warning("No tensorboard writer attached to MetricLogger")
            return
        for name, meter in self.meters.items():
            self.tensorboard_writer.add_scalar(tag=name, 
                                    scalar_value=meter.log, 
                                    global_step=step)

        
    def log_every(self, iterable, print_freq, epoch=None, header=None):
        
        self.add_meter('total_time', SmoothedValue(window_size=1, type='avg'))

        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(type='avg')
        data_time = SmoothedValue(type='avg')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
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
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
                    
                # send meters to tensorboard
                self.update(total_time=time.time() - start_time)
                self.send_meters_to_tensorboard(step=epoch+i/len(iterable))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))
        self.update(total_time=total_time)
         # send meters to tensorboard
        self.send_meters_to_tensorboard(step=epoch)

##
class Visualizer():
    """ Visualizer wrapper based on Tensorboard.

    Returns:
        Visualizer: Class file.
    """
    def __init__(self, writer):
         self.writer = writer

    def write_images(self,
                     image_tensor, 
                     label_tensor=None,
                     tag=None,
                     sample_size=None,
                     epoch=None):
        """Writes an image tensor to the current tensorboard run. Select sample size to draw samples from the tensor.
        Args:
            image_tensor ([type]): [description]
            label_tensor ([type], optional): [description]. Defaults to None.
            tag ([type], optional): [description]. Defaults to None.
            sample ([type], optional): [description]. Defaults to None.
        """
        
        if self.writer is None:
            return
        
        batch_size = image_tensor.size(0)
        
        if sample_size is not None:
            if sample_size > batch_size:
                sample_size = batch_size
            
            perm = torch.randperm(batch_size)   
            idx = perm[:sample_size]
            image_samples = image_tensor[idx]
        else:
            image_samples = image_tensor
        
        self.writer.add_images(tag, image_samples, epoch)
        
        
    def compare_images(self,
                    image_tensor1,
                    image_tensor2, 
                    label_tensor=None,
                    tag=None,
                    sample_size=None,
                    epoch=None):
        """Compares two image tensors by writing to the current tensorboard run. Select sample size to draw samples from the tensor.
        Args:
            image_tensor ([type]): [description]
            label_tensor ([type], optional): [description]. Defaults to None.
            tag ([type], optional): [description]. Defaults to None.
            sample ([type], optional): [description]. Defaults to None.
        """
        
        if self.writer is None:
            return
        
        batch_size = image_tensor1.size(0)
        
        if sample_size is not None:
            if sample_size > batch_size:
                sample_size = batch_size
            
            perm = torch.randperm(batch_size)   
            idx = perm[:sample_size]
            image_samples1 = image_tensor1[idx]
            image_samples2 = image_tensor2[idx]
        else:
            image_samples1 = image_tensor1
            image_samples2 = image_tensor2
        
        self.writer.add_images(tag + " Tensor 1", image_samples1, epoch)
        self.writer.add_images(tag + " Tensor 2", image_samples2, epoch)


def log_config(logpath, config_path):
    from shutil import copy
    copy(config_path, logpath)