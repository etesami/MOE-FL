# from multiprocessing import Pool
from multiprocessing import Process
import time
import random

def func_parallel(msg, num):
    number = random.random()
    for i in range(num):
        print("[{}] ({}): {}".format(number, i, msg))
        time.sleep(1 + number)
    return "[{}]: Sucess".format(number)


if __name__ == "__main__":
    # pool = Pool()
    # result = pool.apply_async(func_parallel, ("salam1",5,))
    p1 = Process(target = func_parallel, args=('salam-1', 5,))
    p2 = Process(target = func_parallel, args=('salam-2', 5,))
    p1.start()
    p2.start()
    
    for i in range(2):
        func_parallel("salam-main", 1)
    
    p1.join()
    p2.join()
