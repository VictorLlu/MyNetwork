import os
import subprocess
from datetime import datetime

lr_list = [0.05, 0.08, 0.1, 0.2, 0.5, 1, 2]
wd_list = [0, 1e-5, 1e-4, 1e-3]
hidden_list = [64, 128, 256, 512]
working_gpu = list(range(8))
thread_working = []
grid_search_aux = datetime.now().strftime("%Y-%m-%d-%H-%M")

for lr in lr_list:
    for wd in wd_list:
        for hidden in hidden_list:
            if len(working_gpu) > 0:
                thread_working.append((subprocess.Popen("python train_mnist.py --gpu %d --hidden_plane %d --lr %f --wd %f --grid_search %s"%(working_gpu[0], hidden, lr, wd, grid_search_aux), shell=True), working_gpu[0]))
                working_gpu.pop(0)
            else:
                wait_flag = True
                while wait_flag:
                    for p_id, p_working in enumerate(thread_working):
                        if p_working[0].poll() is not None:
                            thread_working.pop(p_id)
                            working_gpu.append(p_working[1])
                            thread_working.append((subprocess.Popen("python train_mnist.py --gpu %d --hidden_plane %d --lr %f --wd %f --grid_search %s"%(working_gpu[0], hidden, lr, wd, grid_search_aux), shell=True), working_gpu[0]))
                            working_gpu.pop(0)
                            wait_flag = False
                            break
            # os.system()