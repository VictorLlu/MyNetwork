
import mytorch
BACKEND = mytorch.make_tensor_backend(mytorch.CudaOps)

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


class TwoLayerNetwork(mytorch.Module):
    """
    Implement a CNN for MNist classification based on LeNet.

    This model should implement the following procedure:

    1. Apply a convolution with 4 output channels and a 3x3 kernel followed by a ReLU (save to self.mid)
    2. Apply a convolution with 8 output channels and a 3x3 kernel followed by a ReLU (save to self.out)
    3. Apply 2D pooling (either Avg or Max) with 4x4 kernel.
    4. Flatten channels, height, and width. (Should be size bsx392)
    5. Apply a Linear to size 64 followed by a ReLU and Dropout with rate 25%
    6. Apply a Linear to size C (number of classes).
    """

    def __init__(self, hidden_plane=128):
        super().__init__()

        self.linear1 = Linear(784, hidden_plane)
        self.linear2 = Linear(hidden_plane, C)

    def forward(self, x):
        x = self.linear1(x).relu()
        x = self.linear2(x)
        return x
