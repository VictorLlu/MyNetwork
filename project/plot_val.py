import os
import matplotlib.pyplot as plt
import numpy as np

filename = "output/two_layer_net-1024-16-0.500000-0.000010-0.9/training_log.log"
val_list = []
with open(filename) as f:
    lines = f.readlines()
for line in lines:
    line_list = line[:-1].split(' ')
    if line_list[2] == '====':
        val_list.append(float(line_list[9]))
iter = np.arange(len(val_list))
iter = iter * 10
plt.plot(iter, val_list)
plt.grid()
plt.xlabel("Iterations")
plt.ylabel("Top1 accuracy")
# plt.show()
plt.savefig("/Users/victorlu/Desktop/two-layer-net/val_plot.pdf")