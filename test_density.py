import argparse
import time

import ray
import nums.numpy as nps
from nums.core import settings

def measure(x1, x2):
    start = time.time()

    result = x1 @ x2
    print(result.touch())

    end = time.time()

    return end - start

def run(construct):
    size = 5000
    density = 0.01

    print("Density", density)

    print("Total entries", size ** 2)

    print("Constructing")

    x1, x2 = construct(size, density)

    print("Running")

    secs = measure(x1, x2)

    print("--- %s seconds ---" % secs)

def construct_dense(size, density):
    print("Dense")
    x1 = nps.random.rand(size, size)
    x2 = nps.random.rand(size, size)

    return x1, x2

def construct_sparse(size, density):
    x1 = nps.random.randn_sparse(size, size, density, 'csr')
    x2 = nps.random.randn_sparse(size, size, density, 'csc')

    return x1, x2

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--address', default="")
    args = parser.parse_args()

    settings.use_head = True
    settings.cluster_shape = (1, 1)
    print("connecting to head node", args.address)
    ray.init(**{
        "address": args.address
    })

    run(construct_sparse)


