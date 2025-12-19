# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
"""
Various utilities related to track and report metrics
"""
import datetime
import time
from collections import defaultdict, deque

import torch
import torch.distributed as dist

from util.dist import is_dist_avail_and_initialized

## 跟踪一系列值并提供对窗口内平滑值或全局系列平均值的访问。
class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        # 字符串格式
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        #deque是双端列表，两端都可实现append和pop,还可设置最大长度
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt
    ##根据value更新 deque 、count和total
    def update(self, value, num=1):
        self.deque.append(value)
        self.count += num
        self.total += value * num

    ## 将count 和 total 值在不同gpu上进行同步
    def synchronize_between_processes(self):
        """
        Distributed synchronization of the metric
        Warning: does not synchronize the deque!
        """
        #如果不并行话训练，则直接返回
        if not is_dist_avail_and_initialized():
            return
        #将count 和 total 合并成一个tensor，才能在进行分布式训练时进行分发和并行计算
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        #设置并行化堵塞
        dist.barrier()
        # 对t进行并行求和，即将所有节点上的t值相加
        dist.all_reduce(t)
        #返回同步后的t值
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property #获取所有value中间值
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property #获取当前节点上value的平均值
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property  #获取所有节点上value的平均值
    def global_avg(self):
        return self.total / self.count

    @property #获取最大的一个value值
    def max(self):
        return max(self.deque)

    @property #获取最后一个value值
    def value(self):
        return self.deque[-1]

    ## 字符串内置函数，对类对象使用print 或者str就可获得 __str__中的数据
    def __str__(self):
        return self.fmt.format(
            median=self.median, avg=self.avg, global_avg=self.global_avg, max=self.max, value=self.value
        )


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        #设置值的默认值为 一个SmoothedValue对象 的字典
        self.meters = defaultdict(SmoothedValue)
        #设置分隔符参数
        self.delimiter = delimiter

    ## 将 输入的值更新到meters中
    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def update_v2(self, key, value, num):
        self.meters[key].update(value, num)
    ## 当调用了类中不存在的属性时，还自动访问__getattr__（）函数
    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, attr))

    ##在调用类的str时，将meters中的所有值加载到一个str中
    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    #将meter中的各个值并行化
    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    #往meter字典中添加属性和值
    def add_meter(self, name, meter):
        self.meters[name] = meter

    ## 记录iterable中的每个元素以及处理的时间等等信息，并且以print_freq的概率输出
    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        ## 记录时间
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"

        ## 设置log格式
        if torch.cuda.is_available():
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                    "max mem: {memory:.0f}",
                ]
            )
        else:
            log_msg = self.delimiter.join(
                [header, "[{0" + space_fmt + "}/{1}]", "eta: {eta}", "{meters}", "time: {time}", "data: {data}"]
            )

        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            #max_memory_allocated 可以查看设备上tensor最大占用的GPU显存，以字节bytes为单位
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    print(
                        log_msg.format(
                            i, len(iterable), eta=eta_string, meters=str(self), time=str(iter_time), data=str(data_time)
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("{} Total time: {} ({:.4f} tokenzier / it)".format(header, total_time_str, total_time / len(iterable)))


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)
    # 从输出中选取最大的前k个值，并按顺序排列
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    #判断预测正确的值
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        #计算总值
        correct_k = correct[:k].view(-1).float().sum(0)
        #求平均值并百分化
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
