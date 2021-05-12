import argparse
import time

import ray
import nums.numpy as nps
from nums.core import settings

def routine():
    size = 100
    density = 0.20

    print("Total entries", size ** 2)

    print("Constructing")

    x1 = nps.random.randn_sparse(size, size, density, 'csr')
    x2 = nps.random.randn_sparse(size, size, density, 'csc')

    print("Running")

    start = time.time()

    result = x1 @ x2
    print(result.touch())

    end = time.time()

    print("--- %s seconds ---" % (end - start))


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

    routine()


