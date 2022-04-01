from mnist import MNIST
import mytorch
import numpy as np
import logging
import argparse
import os
import datetime
import time
from model import TwoLayerNetwork
from numba import cuda
import csv

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=5, metavar='epoch', type=int)
    parser.add_argument('--gpu', default=0, metavar='gpu', type=int)
    parser.add_argument('--bs', default=16, metavar='N', type=int)
    parser.add_argument('--lr', type=float, default=0.05, metavar='LR',
                        help='learning rate (default: 0.05)')
    parser.add_argument('--wd', type=float, default=0.0001, metavar='WeightDecay',
                        help='weight decay (default: 0.0001)')  
    parser.add_argument('--hidden_plane', default=128, metavar='hidden', type=int)
    parser.add_argument('--model', default='two_layer_net', type=str, metavar='name')                  
    parser.add_argument('--checkpoint', default='output', type=str, metavar='checkpoint')
    parser.add_argument('--grid_search', type=str, metavar='checkpoint')
    args = parser.parse_args()
    
    cuda.select_device(args.gpu)

    model_name = args.model + "-%d-%d-%f-%.6f"%(args.hidden_plane, args.bs, args.lr, args.wd)
    checkpoint_path = os.path.join(args.checkpoint, model_name)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
        # print("Checkpoint created! Checkpoint will be saved at %s" % checkpoint_path)
    
    logging.basicConfig(filename="%s/training_log.log"%(checkpoint_path), format='%(asctime)s %(message)s', filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.CRITICAL)

    if args.grid_search is not None:
        # csv_name = "grid_search_result-%s.csv"%datetime.now().strftime("%Y-%m-%d-%H.%M.%S")
        csv_name = os.path.join(args.checkpoint, "grid_search_result-%s.csv"%args.grid_search)
        if not os.path.exists(csv_name):
            with open(csv_name, "w+", newline='') as file: 
                csv_file = csv.writer(file)
                head = ["name", "epoch", "bs", "hidden layer", "learning rate", "weight decay", "acc"]
                csv_file.writerow(head)

    info = "gpu: %d, bs: %d, lr: %f, wd: %f, hidden_plane: %d, checkpoint: %s"%\
    (args.gpu, args.bs, args.lr, args.wd, args.hidden_plane, checkpoint_path)
    print(info)
    logger.critical(info)

    max_epochs = args.epochs
    bs = args.bs
    lr = args.lr
    weight_decay = args.wd

    data_train, data_val = (make_mnist(0, 10000), make_mnist(10000, 10500))
    (X_train, y_train) = data_train
    (X_val, y_val) = data_val
    
    model = TwoLayerNetwork(hidden_plane=args.hidden_plane)
    # mytorch.save(model, os.path.join(checkpoint_path, 'two_layer_net.pkl'))
    # state_dict = mytorch.load(os.path.join(checkpoint_path, 'two_layer_net.pkl'))
    # mytorch.load_state_dict(model, state_dict)

    n_training_samples = len(X_train)
    n_step_per_epoch = n_training_samples // bs
    n_test_samples = len(X_val)
    optim = mytorch.SGD(model.parameters(), lr)
    losses = []
    best_acc = 0.0
    best_model_name = ""
    best_saved = False
    for epoch in range(1, max_epochs + 1):
        model.train()
        for batch_num, example_num in enumerate(
            range(0, n_training_samples, bs)
        ):  
            begin_time = time.time()
            iteraion = batch_num + (epoch - 1) * n_step_per_epoch
            if n_training_samples - example_num <= bs:
                continue
            y_ = np.array(y_train[example_num : example_num + bs])
            x_ = (np.array(X_train[example_num : example_num + bs]) / 255.0 * 0.99) + 0.01
            y = mytorch.tensor(y_, backend=BACKEND)
            x = mytorch.tensor(x_, backend=BACKEND)
            x.requires_grad_(True)
            y.requires_grad_(True)
            optim.zero_grad()
            # Forward
            out = model.forward(x.view(bs, H*W))
            loss = mytorch.CrossEntropyLoss(out, y)
            # prob = (out * y).sum(1)
            # loss = -(prob / y.shape[0]).sum()

            reg_loss = 0
            for param in model.parameters():
                reg_loss = reg_loss + (param.value * param.value).sum()

            loss = loss + reg_loss * weight_decay

            assert loss.backend == BACKEND
            loss.view(1).backward()
            losses.append(loss[0])
            # Update
            optim.step()
            end_time = time.time()
            iter_time = end_time - begin_time
            remain_time = (max_epochs - epoch) * iter_time * n_step_per_epoch\
                            + iter_time * (n_step_per_epoch - 1 - batch_num)
            remain_hour = remain_time // 3600
            remain_min = remain_time // 60
            info = 'Epoch [%d][%d/%d], etc: %dh:%dmin, lr: %f, loss: %f'%\
                            (epoch, batch_num, n_step_per_epoch, remain_hour, remain_min, lr, loss[0])
            # print(info)
            logger.critical(info)

            if batch_num % 10 == 0 and batch_num !=0 :
                model.eval()
                # Evaluate on 5 held-out batches

                correct = 0
                for val_example_num in range(0, n_test_samples, bs):
                    if n_test_samples - val_example_num <= bs:
                        continue
                    y_ = np.array(y_val[val_example_num : val_example_num + bs])
                    x_ = (np.array(X_val[val_example_num : val_example_num + bs]) / 255.0 * 0.99) + 0.01
                    y = mytorch.tensor(y_, backend=BACKEND)
                    x = mytorch.tensor(x_, backend=BACKEND)
                    out = model.forward(x.view(bs, H*W))
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
                info = '==== Epoch %d, Iteration %d, Test acc %.4f ====' % (epoch, iteraion, test_acc)
                # print(info)
                logger.critical(info)
                if test_acc > best_acc:
                    best_acc = test_acc
                    if best_saved:
                        os.remove(best_model_name)
                    else:
                        best_saved = True
                    best_model_name = os.path.join(checkpoint_path, 'best-%d.pkl'%(iteraion))
                    mytorch.save(model, best_model_name)
                    info = 'best model saved at iteration: %d' % (iteraion)
                    # print(info)
                    logger.critical(info)
                model.train()
    mytorch.save(model, os.path.join(checkpoint_path, 'last.pkl'))
    info = 'Best acc: %.4f'%best_acc
    # print(info)
    logger.critical(info)
    if args.grid_search is not None:
        with open(csv_name, "a+", newline='') as file: 
            csv_file = csv.writer(file)
            data = [model_name, max_epochs, bs, args.hidden_plane, lr, weight_decay, best_acc]
            csv_file.writerow(data)
