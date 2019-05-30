import time


def add(a, b):
    return a + b


def outfunc(func):
    stime = time.clock()
    
    def infunc(*args, **kwargs):
        res = func(*args, **kwargs)
        etime = time.clock()
        print("use time: {:.7f}".format(etime - stime))
        return res
    
    return infunc


@outfunc
def mul(a, b):
    return a * b


@outfunc
def loop(num):
    n = 1
    for i in range(num):
        n *= n**i
    return n


if __name__ == '__main__':
    loop(10000000)