import sys
import numpy as np
from matplotlib import pyplot as plt
from time import sleep
import random

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

if len(sys.argv) < 2:
    print("Error: please give a file to import")
    exit()
    
dict = unpickle(sys.argv[1])

for key, value in dict.items():
    print(key)
    
print("---")
data = dict[b'data']


data = [d.reshape(3,1024).T.reshape(32,32,3) for d in data]

for d in data:
    plt.imshow(d)
    plt.show()
    plt.close()