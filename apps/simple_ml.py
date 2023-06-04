import struct
import gzip
import numpy as np

import sys
sys.path.append('python/')
import needle as ndl


def unzip(file_path: str) -> bytes:
    with gzip.open(file_path, "rb") as f_in:
        # Read the uncompressed data
        uncompressed_data = f_in.read()
        return uncompressed_data


def parse_images(file_path: str) -> np.ndarray[np.float32]:
    data: bytes = unzip(file_path)
    image_num: int = int.from_bytes(data[4:8], byteorder='big')
    row_num: int = int.from_bytes(data[8:12], byteorder='big')
    col_num: int = int.from_bytes(data[12:16], byteorder='big')
    data = data[16:]
    assert(row_num == 28 and col_num == 28 and len(data) % image_num == 0)
    image_size = int(len(data)/image_num)
    images = np.ndarray((image_num, image_size), dtype=np.float32)
    for i in range(image_num):
        tup = struct.unpack_from("B"*image_size, data, i*image_size)
        images[i] = np.array(tup).astype(np.float32)
    images_normed = (images - np.min(images)) / (np.max(images) - np.min(images))
    return images_normed


def parse_labels(file_path: str) -> np.ndarray[np.uint8]:
    data: bytes = unzip(file_path)
    label_num: int = int.from_bytes(data[4:8], byteorder='big')
    data = data[8:]
    labels = np.ndarray((label_num, ), dtype=np.uint8)
    for i in range(label_num):
        labels[i] = struct.unpack_from("B", data, i)[0]
    return labels


def parse_mnist(image_filename, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0. The normalization should be applied uniformly
                across the whole dataset, _not_ individual images.

            y (numpy.ndarray[dtype=np.uint8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.uint8 and
                for MNIST will contain the values 0-9.
    """
    images = parse_images(image_filename)
    labels = parse_labels(label_filename)
    return images, labels


def softmax_loss(Z: ndl.Tensor, y_one_hot: ndl.Tensor):
    """ Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y_one_hot (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    loss = ndl.log(ndl.summation(ndl.exp(Z), axes=1)) - ndl.summation(Z*y_one_hot, axes=1)
    return ndl.summation(loss) / loss.shape[0]


def nn_epoch(X, y, W1, W2, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W1
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """
    iterations = int(X.shape[0] / batch)
    for iteration in range(iterations):
        X_batch = X[iteration * batch:(iteration + 1) * batch]
        y_batch = y[iteration * batch:(iteration + 1) * batch]
        m, n = X_batch.shape
        d, k = W2.shape
        Z = ndl.Tensor(X_batch) @ W1
        Z = ndl.relu(Z)
        Z = Z @ W2
        Iy = np.zeros((m, k))
        Iy[np.arange(len(y_batch)), y_batch] = 1
        loss = softmax_loss(Z, ndl.Tensor(Iy))
        loss.backward()
        W1, W2 = W1 - W1.grad * lr, W2 - W2.grad * lr
        W1 = W1.detach()
        W2 = W2.detach()
    return W1, W2


### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT

def loss_err(h,y):
    """ Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h,y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
