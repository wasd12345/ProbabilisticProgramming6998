import numpy as np
from urllib import request
import gzip
import pickle
from ncp import tools

filename = [
["training_images","train-images-idx3-ubyte.gz"],
["test_images","t10k-images-idx3-ubyte.gz"],
["training_labels","train-labels-idx1-ubyte.gz"],
["test_labels","t10k-labels-idx1-ubyte.gz"]
]

def download_mnist():
    base_url = "http://yann.lecun.com/exdb/mnist/"
    for name in filename:
        print("Downloading "+name[1]+"...")
        request.urlretrieve(base_url+name[1], name[1])
    print("Download complete.")

def save_mnist():
    mnist = {}
    for name in filename[:2]:
        with gzip.open(name[1], 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1,28*28)
    for name in filename[-2:]:
        with gzip.open(name[1], 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)
    with open("mnist.pkl", 'wb') as f:
        pickle.dump(mnist,f)
    print("Save complete.")

def init():
    download_mnist()
    save_mnist()

def load_mnist():
    download_mnist()
    save_mnist()
    with open("./ncp/datasets/mnist.pkl",'rb') as f:
        mnist = pickle.load(f)
    train = tools.AttrDict(inputs = mnist["training_images"], targets = np.expand_dims(mnist["training_labels"], 1))
    test = tools.AttrDict(inputs = mnist["test_images"], targets = np.expand_dims(mnist["test_labels"], 1))
    #domain variable needed for plotting? (just taking value from toy.py) 
    domain = np.linspace(-1.2, 1.2, 1000)
    return tools.AttrDict(domain = domain, train = train, test = test, target_scale = 1)

if __name__ == '__main__':
    init()
