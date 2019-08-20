from time import time


def timer(f):
    def g(*args, **kwargs):
        t0 = time()
        a = f(*args, **kwargs)
        print(f"Runtime: {round(time() - t0,2)} seconds")
        return a
    return g
