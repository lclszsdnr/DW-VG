# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Utilities related to distributed mode.

By default, the reduce of metrics and such are done on GPU, since it'tokenzier more straightforward (we reuse the NCCL backend)
If you want to reduce on CPU instead (required for big datasets like GQA), use the env variable MDETR_CPU_REDUCE=1
"""
import functools
import io
import os

import torch
import torch.distributed as dist

_LOCAL_PROCESS_GROUP = None


@functools.lru_cache()
def _get_global_gloo_group():
    """
    Return a process group based on gloo backend, containing all the ranks
    The result is cached.
    """

    if dist.get_backend() == "nccl":
        return dist.new_group(backend="gloo")

    return dist.group.WORLD


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """

    world_size = get_world_size()
    if world_size == 1:
        return [data]

    cpu_group = None
    if os.getenv("MDETR_CPU_REDUCE") == "1":
        cpu_group = _get_global_gloo_group()

    buffer = io.BytesIO()
    torch.save(data, buffer)
    data_view = buffer.getbuffer()
    device = "cuda" if cpu_group is None else "cpu"
    tensor = torch.ByteTensor(data_view).to(device)

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device=device, dtype=torch.long)
    size_list = [torch.tensor([0], device=device, dtype=torch.long) for _ in range(world_size)]
    if cpu_group is None:
        dist.all_gather(size_list, local_size)
    else:
        print("gathering on cpu")
        dist.all_gather(size_list, local_size, group=cpu_group)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)
    assert isinstance(local_size.item(), int)
    local_size = int(local_size.item())

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device=device))
    if local_size != max_size:
        padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device=device)
        tensor = torch.cat((tensor, padding), dim=0)
    if cpu_group is None:
        dist.all_gather(tensor_list, tensor)
    else:
        dist.all_gather(tensor_list, tensor, group=cpu_group)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        tensor = torch.split(tensor, [size, max_size - size], dim=0)[0]
        buffer = io.BytesIO(tensor.cpu().numpy())
        obj = torch.load(buffer)
        data_list.append(obj)

    return data_list

## 在并行化时reduce 输入字典中的值
def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict

## 该函数就是一个比较高级实现的，进行对只有在节点为master时才进行打印的设置
def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    '''
    当使用内建模块中函数或其它功能时，可以直接使用，不用添加内建模块的名字；但是，如果想要向内建模块中添加一些功能，
    以便在任何函数中都能直接使用而不用再进行 import，这时，就要导入内建模块 builtins（ python2 导入 __builtin__）
    ，在内建模块的命名空间(即 __dict__ 字典属性)中添加该功能。就要导入 模块。
    如下面的例子，将print_hello 加入到内建模块中，并取名为hello,此时，print_hello 和 hello 两个函数名几乎是一样，
    但是有一点区别，print_hello 只能在该模块中使用，而 hello 可以在本程序中的其它任何一个模块中使用，因为 hello 已经放到内建模块中了。
    
    import builtins
    def print_hello():
        print("hello, world")
    builtins.__dict__['hello'] = print_hello
    print_hello()		# 将打印"hello, world"
    hello()				# 将打印"hello, world"

    
    '''
    import builtins as __builtin__
    #获取Python中内建的print函数
    builtin_print = __builtin__.print
    ## 重写print函数，令该函数只有在is_master 或 force 为True时，才真的运行print
    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)
    #将重写的print 替换掉内建的print
    __builtin__.print = print

## 检测是否进行了分布式训练，返回true or false
def is_dist_avail_and_initialized():
    """
    Returns:
        True if distributed training is enabled
    """
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    """
    Returns:
        The number of processes in the process group
    """
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    """
    Returns:
        The rank of the current process within the global process group.
    """
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def get_local_rank() -> int:
    """
    Returns:
        The rank of the current process within the local (per-machine) process group.
    """
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    assert _LOCAL_PROCESS_GROUP is not None
    return dist.get_rank(group=_LOCAL_PROCESS_GROUP)


def get_local_size() -> int:
    """
    Returns:
        The size of the per-machine process group,
        i.e. the number of processes per machine.
    """
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size(group=_LOCAL_PROCESS_GROUP)


def is_main_process():
    """Return true if the current process is the main one"""
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    """Utility function to save only from the main process"""
    if is_main_process():
        torch.save(*args, **kwargs)

## 设置并行化
def init_distributed_mode(args):
    """Initialize distributed training, if appropriate"""
    ## 从环境变量中获取并行化参数
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    ## 使用SLURM时并行化参数的设置
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    # 否则就是限制错误
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True
    #设置模型运行所在显卡
    torch.cuda.set_device(args.gpu)
    #设置并行化通信方式
    args.dist_backend = "nccl"
    print("| distributed init (rank {}): {}".format(args.rank, args.dist_url), flush=True)

    # 启动torch的并行化集群
    dist.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
    )
    #设置同步屏障,在这里等待所有进程到此后才能进行接着训练
    dist.barrier()
    setup_for_distributed(args.rank == 0)
