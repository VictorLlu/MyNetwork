# Two-layer Nerual Network 
## Introduction
This repo is the first homework of course DATA620004 Nerual Network and Deep Learning. 
This project is required to construct a two-layer nerual network and be trained as a qualified classifier.
To meet the requirement, I build a light deep learning architecture called `mytorch` based on `numpy` and its cuda counterpart: `numba`.
`mytorch` inherits base architecture from [minitorch](https://minitorch.github.io/), an diy teaching library for machine learning engineers developed by Cornell Tech.
However, it doesn't include detailed implementaion codes, which leave students to complete auto grad, tensor, operators and so on.
The schematic of my torch is shown below (only major classes or functions are shown).
```
mytorch----autodiff.py
        |       |--Variable
        |       |--BaseFunction
        |
        |--operators.py
        |       |--[BaseOperators]: add, mul, ...
        |
        |--tensor_data.py
        |       |--TensorData
        |
        |--tensor_functions.py
        |       |--TensorFunction(BaseFunction, [BaseOperators])
        |
        |--tensor.py
        |       |--Tensor(Variable)
        |
        |--module.py
        |       |--Parameter
        |       |--Module
        |
        |--cuda_ops.py
        |       |--map
        |       |-zip
        |       |-reduce
        |       |-matrix_multiply
        |
        |--nn.py
        |       |--argmax
        |       |--max
        |       |--softmax
        |       |--logsoftmax
        |
        |--optim.py
        |       |--Optimizer
        |               |--SGD
        |
        |--loss.py
        |       |--CrossEntropyLoss
        |
        |--learning_rate_scheduler.py
        |       |--LearningRateSchedular
        |               |--StepLR
```
Details of implementation is illustrated throughly in report.
With the help of `mytorch`, the two-layer network achieves 97% accuracy on MNIST validation set with high efficiency.

## Preparation
First, install python library as is given in requirment.txt by 
```
pip install -r requirement.txt
```
Next, install mytorch by
```
git clone https://github.com/VictorLlu/MyNetwork.git
cd  MyNetwork
pip install -e .
```
Then, please download 4 .gz files from [mnist](http://yann.lecun.com/exdb/mnist/) dataset and place them at `MyNetwork/project/data/`.

## Training
To train a two-layer network, you can run
```
cd project
python train_mnist.py --gpu ${GPU INDEX} --epochs ${EPOCHS} --bs ${BATCH SIZE} --lr ${LEARNING RATE} --lr_drop ${LEARNING RATE DECAY RATE} --wd ${WEIGHT DECAY} --hidden-plane ${HIDDEN LAYERS} --checkpoint ${OUPUT PATH}
```
## Test
To test a pretrained two-layer network, you can run
```
cd project
python test_mnist.py --gpu ${GPU INDEX} --bs ${BATCH SIZE} --checkpoint ${CHECKPOINT PATH}
```
## Grid Search
To search the best hyperparameters, please run
```
cd project
python grid_search_mnist.py
```
The search result will be shown in a `csv` file in the output path.