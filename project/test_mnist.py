from mnist import MNIST
import mytorch
import numpy as np
import argparse
import os
from tqdm import tqdm
from model import TwoLayerNetwork

mndata = MNIST("data/")
mndata.gz = True
images, labels = mndata.load_training()


BACKEND = mytorch.make_tensor_backend(mytorch.CudaOps)

# Number of classes (10 digits)
C = 10

# Size of images (height and width)
H, W = 28, 28


def make_mnist(start, stop):
    ys = []
    X = []
    for i in range(start, stop):
        y = labels[i]
        vals = [0.0] * 10
        vals[y] = 1.0
        ys.append(vals)
        X.append([[images[i][h * W + w] for w in range(W)] for h in range(H)])
    return X, ys


def default_log_fn(epoch, total_loss, acc):
    print("Epoch: ", epoch, " loss: ", total_loss, "acc: %.2f"%(acc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', default=16, metavar='N', type=int)
    parser.add_argument('--model', default='two_layer_net', type=str, metavar='name')                  
    parser.add_argument('--checkpoint', type=str, metavar='checkpoint')
    args = parser.parse_args()

    assert args.checkpoint is not None, "Please provide  checkpoint path!"
    checkpoint_path = args.checkpoint

    data_val = make_mnist(10000, 10500)
    (X_val, y_val) = data_val

    bs = args.bs
    
    model = TwoLayerNetwork()
    state_dict = mytorch.load(checkpoint_path)
    mytorch.load_state_dict(model, state_dict)
    model.eval()
    # mytorch.save(model, os.path.join(checkpoint_path, 'two_layer_net.pkl'))
    # state_dict = mytorch.load(os.path.join(checkpoint_path, 'two_layer_net.pkl'))
    # mytorch.load_state_dict(model, state_dict)
    n_test_samples = len(X_val)

    best_acc = 0.0
    best_model_name = ""
    best_saved = False
   
    correct = 0
    for val_example_num in tqdm(range(0, n_test_samples, bs)):
        if n_test_samples - val_example_num <= bs:
            continue
        y_ = np.array(y_val[val_example_num : val_example_num + bs])
        x_ = (np.array(X_val[val_example_num : val_example_num + bs]) / 255.0 * 0.99) + 0.01
        y = mytorch.tensor(y_, backend=BACKEND)
        x = mytorch.tensor(x_, backend=BACKEND)
        out = mytorch.logsoftmax(out, dim=1)
        for i in range(bs):
            m = -1000
            ind = -1
            for j in range(C):
                if out[i, j] > m:
                    ind = j
                    m = out[i, j]
            if y[i, ind] == 1.0:
                correct += 1
    total_test = n_test_samples // bs * bs
    test_acc = correct / total_test
    print("=====Acc on MNIST: %.2f====="%(test_acc))