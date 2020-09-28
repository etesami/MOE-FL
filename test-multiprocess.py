
from multiprocessing import Process, Array, Value
from multiprocessing import Queue as mq
from federated_learning.FLNet import FLNet
import time
import random
import syft as sy
import torch

def func_parallel(msg, num, server, A):
    number = random.random()
    for i in range(num):
        print("[{}] ({}): {}".format(number, i, msg))
        time.sleep(1 + number)
    for a in A:
        a.send(server)
    return "[{}]: Sucess".format(number)


if __name__ == "__main__":
    # pool = Pool()
    queue = mq()
    hook = sy.TorchHook(torch)
    device = torch.device("cpu")
    server = sy.VirtualWorker(hook, id="server")
    model1 = FLNet().to(device)
    model2 = FLNet().to(device)
    A = Array(FLNet, [model1, model2])    
    # result = pool.apply_async(func_parallel, ("salam1",5,))
    p1 = Process(target = func_parallel, args=('salam-1', 5, server, A,))
    # p2 = Process(target = func_parallel, args=('salam-2', 5,))
    p1.start()
    # p2.start()
    
    # for i in range(2):
    #     func_parallel("salam-main", 1)
    
    p1.join()
    print(model1.location)
    # p2.join()
