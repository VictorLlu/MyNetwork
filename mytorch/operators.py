import math

def mul(x, y):
    ":math:`f(x, y) = x * y`"
    return x * y


def id(x):
    ":math:`f(x) = x`"
    return x


def add(x, y):
    ":math:`f(x, y) = x + y`"
    return x + y


def neg(x):
    ":math:`f(x) = -x`"
    return -x

def lt(x, y):
    ":math:`f(x) =` 1.0 if x is less than y else 0.0"
    if x < y:
        return 1
    else:
        return 0

def eq(x, y):
    ":math:`f(x) =` 1.0 if x is equal to y else 0.0"
    if x == y:
        return 1.0
    else:
        return 0.0

def max(x, y):
    ":math:`f(x) =` x if x is greater than y else y"
    if x > y:
        return x
    else:
        return y

def is_close(x, y):
    ":math:`f(x) = |x - y| < 1e-2` "
    return abs(x-y) < 1e-2


def sigmoid(x):
    """
    :math:`f(x) =  \frac{1.0}{(1.0 + e^{-x})}` if x >=0 else :math:`\frac{e^x}{(1.0 + e^{x})}`
    for stability.

    """  
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))


def relu(x):
    """
    :math:`f(x) =` x if x is greater than 0, else 0
    """
    if x > 0:
        return x
    else:
        return 0.0

def relu_back(x, y):
    ":math:`f(x) =` y if x is greater than 0 else 0"
    if x > 0:
        return y
    else:
        return 0

EPS = 1e-6

def log(x):
    ":math:`f(x) = log(x)`"
    return math.log(x + EPS)


def exp(x):
    ":math:`f(x) = e^{x}`"
    return math.exp(x)


def log_back(a, b):
    return b / (a + EPS)


def inv(x):
    ":math:`f(x) = 1/x`"
    return 1.0 / x


def inv_back(a, b):
    return -(1.0 / a ** 2) * b

def map(fn):
    """
    Higher-order map.
    Args:
        fn (one-arg function): process one value

    Returns:
        function : a function that takes a list and applies `fn` to each element
    """

    def apply(my_list):
        new_list = []
        for counter, value in enumerate(my_list):
            new_list.append(fn(value))
        return new_list

    return apply


def negList(ls):
    "Use :func:`map` and :func:`neg` to negate each element in `ls`"
    return map(neg)(ls)


def zipWith(fn):
    """
    Higher-order zipwith (or map2).
    Args:
        fn (two-arg function): combine two values

    Returns:
        function : takes two equally sized lists `ls1` and `ls2`, produce a new list by
        applying fn(x, y) one each pair of elements.

    """
    def apply(my_list1, my_list2):
        new_list = []
        for counter, value in enumerate(my_list1):
            new_list.append(fn(my_list1[counter], my_list2[counter]))
        return new_list
    return apply


def addLists(ls1, ls2):
    "Add the elements of `ls1` and `ls2` using :func:`zipWith` and :func:`add`"
    return zipWith(add)(ls1, ls2)


def reduce(fn, start):
    r"""
    Higher-order reduce.
    Args:
        fn (two-arg function): combine two values
        start (float): start value :math:`x_0`
    Returns:
        function : function that takes a list `ls` of elements
        :math:`x_1 \ldots x_n` and computes the reduction :math:`fn(x_3, fn(x_2,
        fn(x_1, x_0)))`
    """
    def apply(input_list):
        my_list = list(input_list).copy()

        if len(my_list) == 0:
            return start
        current_value = my_list.pop()
        return fn(current_value, apply(my_list))
    return apply


def sum(ls):
    """
    Sum up a list using :func:`reduce` and :func:`add`.
    """
    my_fn = reduce(add, 0)
    return my_fn(ls)


def prod(ls):
    """
    Product of a list using :func:`reduce` and :func:`mul`.
    """
    my_fn = reduce(mul, 1)
    return my_fn(ls)

