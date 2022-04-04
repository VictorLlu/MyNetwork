import os
import matplotlib.pyplot as plt
import numpy as np

filename = "output/two_layer_net-1024-16-0.500000-0.000010-0.9/training_log.log"
loss_list = []
with open(filename) as f:
    lines = f.readlines()
for line in lines:
    line_list = line[:-1].split(' ')
    if line_list[2] == 'Epoch':
        loss_list.append(float(line_list[9]))
iter = np.arange(len(loss_list))
interval = 50
plt.plot(iter[::interval], loss_list[::interval])
plt.grid()
plt.xlabel("Iterations")
plt.ylabel("Cross entropy loss")
plt.savefig("/Users/victorlu/Desktop/two-layer-net/loss_plot.pdf")