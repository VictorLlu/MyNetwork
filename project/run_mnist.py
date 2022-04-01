from mnist import MNIST
import mytorch
import numpy as np


mndata = MNIST("data/")
mndata.gz = True
images, labels = mndata.load_training()


BACKEND = mytorch.make_tensor_backend(mytorch.CudaOps)
BATCH = 16

# Number of classes (10 digits)
C = 10

# Size of images (height and width)
H, W = 28, 28


def RParam(*shape):
    r = 0.1 * (mytorch.rand(shape, backend=BACKEND) - 0.5)
    return mytorch.Parameter(r)


class Linear(mytorch.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.weights = RParam(in_size, out_size)
        self.bias = RParam(out_size)
        self.out_size = out_size

    def forward(self, x):
        batch, in_size = x.shape
        return (
            x.view(batch, in_size) @ self.weights.value.view(in_size, self.out_size)
        ).view(batch, self.out_size) + self.bias.value


class Network(mytorch.Module):
    """
    Implement a CNN for MNist classification based on LeNet.

    This model should implement the following procedure:

    1. Apply a convolution with 4 output channels and a 3x3 kernel followed by a ReLU (save to self.mid)
    2. Apply a convolution with 8 output channels and a 3x3 kernel followed by a ReLU (save to self.out)
    3. Apply 2D pooling (either Avg or Max) with 4x4 kernel.
    4. Flatten channels, height, and width. (Should be size BATCHx392)
    5. Apply a Linear to size 64 followed by a ReLU and Dropout with rate 25%
    6. Apply a Linear to size C (number of classes).
    7. Apply a logsoftmax over the class dimension.
    """

    def __init__(self):
        super().__init__()

        # For vis
        self.mid = None
        self.out = None

        self.linear1 = Linear(784, 128)
        self.linear2 = Linear(128, C)

    def forward(self, x):
        # ASSIGN4.5
        x = self.linear1(x).relu()
        x = self.linear2(x)
        x = mytorch.logsoftmax(x, dim=1)
        return x


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


class ImageTrain:
    def __init__(self):
        self.model = Network()

    def run_one(self, x):
        return self.model.forward(mytorch.tensor([x], backend=BACKEND))

    def train(
        self, data_train, data_val, learning_rate, max_epochs=500, log_fn=default_log_fn
    ):
        (X_train, y_train) = data_train
        (X_val, y_val) = data_val
        # self.model = Network()
        model = self.model
        n_training_samples = len(X_train)
        n_step_per_epoch = n_training_samples // BATCH
        n_test_samples = len(X_val)
        optim = mytorch.SGD(self.model.parameters(), learning_rate)
        losses = []
        for epoch in range(1, max_epochs + 1):
            total_loss = 0.0

            model.train()
            for batch_num, example_num in enumerate(
                range(0, n_training_samples, BATCH)
            ):

                if n_training_samples - example_num <= BATCH:
                    continue
                y_ = np.array(y_train[example_num : example_num + BATCH])
                x_ = (np.array(X_train[example_num : example_num + BATCH]) / 255.0 * 0.99) + 0.01
                y = mytorch.tensor(y_, backend=BACKEND)
                x = mytorch.tensor(x_, backend=BACKEND)
                # import pdb;pdb.set_trace()
                x.requires_grad_(True)
                y.requires_grad_(True)
                # Forward
                out = model.forward(x.view(BATCH, H*W)).view(BATCH, C)
                prob = (out * y).sum(1)
                loss = -(prob / y.shape[0]).sum()

                assert loss.backend == BACKEND
                loss.view(1).backward()
                print("{}/{}\t loss: {}".format(batch_num, n_step_per_epoch, loss[0]))
                # import pdb;pdb.set_trace()
                total_loss += loss[0]
                losses.append(total_loss)

                # Update
                optim.step()

                if batch_num % 5 == 0 and batch_num !=0 :
                    model.eval()
                    # Evaluate on 5 held-out batches

                    correct = 0
                    for val_example_num in range(0, n_test_samples, BATCH):
                        if n_test_samples - val_example_num <= BATCH:
                            continue
                        y_ = np.array(y_val[val_example_num : val_example_num + BATCH])
                        x_ = (np.array(X_val[val_example_num : val_example_num + BATCH]) / 255.0 * 0.99) + 0.01
                        y = mytorch.tensor(y_, backend=BACKEND)
                        x = mytorch.tensor(x_, backend=BACKEND)
                        out = model.forward(x.view(BATCH, H*W)).view(BATCH, C)
                        for i in range(BATCH):
                            m = -1000
                            ind = -1
                            for j in range(C):
                                if out[i, j] > m:
                                    ind = j
                                    m = out[i, j]
                            if y[i, ind] == 1.0:
                                correct += 1
                    total_test = n_test_samples // BATCH * BATCH
                    log_fn(epoch, total_loss, correct / total_test)

                    total_loss = 0.0
                    model.train()


if __name__ == "__main__":
    data_train, data_val = (make_mnist(0, 5000), make_mnist(10000, 10500))
    ImageTrain().train(data_train, data_val, learning_rate=0.05)
