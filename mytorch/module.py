## Task 0.4
## Modules
import os
import dill as pkl
from .tensor import Tensor

class Module:
    """
    Attributes:
        _modules (dict of name x :class:`Module`): Storage of the child modules
        _parameters (dict of name x :class:`Parameter`): Storage of the module's parameters
        mode (string): Mode of operation, can be {"train", "eval"}.

    """

    def __init__(self):
        self._modules = {}
        self._parameters = {}
        self.training = True

    def modules(self):
        "Return the child modules of this module."
        return self.__dict__["_modules"].values()

    def train(self):
        "Set the mode of this module and all descendent modules to `train`."
        self.training = True
        if len(self._modules) != 0:
            for key in self._modules:
                self._modules[key].train()

            # if(self._modules[key]    )

        # self.__dict__["_modules"].values().mode = "train"

        # for m in self._modules:
        #     m.mode = "train"

        # TODO: Implement for Task 0.4.
        # raise NotImplementedError("Need to implement for Task 0.4")

    def eval(self):
        "Set the mode of this module and all descendent modules to `eval`."
        # TODO: Implement for Task 0.4.
        self.training = False
        if len(self._modules) != 0:
            for key in self._modules:
                self._modules[key].eval()
        # self.__dict__["_modules"].values().mode = "eval"

        # for m in self._modules:
        #     m.mode = "eval"
        # raise NotImplementedError("Need to implement for Task 0.4")

    def named_parameters(self):
        """
        Collect all the parameters of this module and its descendents.


        Returns:
            dict: Each name (key) and :class:`Parameter` (value) under this module.
        """
        if len(self._modules) == 0:
            new_dict = {}
            for p in self._parameters:
                new_dict[p] = self._parameters[p]
            return new_dict
        else:
            this_para = {}
            for key in self._parameters:
                this_para[key] = self._parameters[key]
            for key in self._modules:
                child_para = self._modules[key].named_parameters()
                for name in child_para:
                    this_para[key + "." + name] = child_para[name]
            return this_para

        # TODO: Implement for Task 0.4.
        # raise NotImplementedError("Need to implement for Task 0.4")

    def named_variables(self):
        """
        Collect all the parameters of this module and its descendents.


        Returns:
            dict: Each name (key) and :class:`Parameter` (value) under this module.
        """
        if len(self._modules) == 0:
            new_dict = {}
            for p in self._parameters:
                new_dict[p] = self._parameters[p].value._tensor
            return new_dict
        else:
            this_para = {}
            for key in self._parameters:
                this_para[key] = self._parameters[key].value._tensor
            for key in self._modules:
                child_para = self._modules[key].named_parameters()
                for name in child_para:
                    this_para[key + "." + name] = child_para[name].value._tensor
            return this_para

    def parameters(self):
        return self.named_parameters().values()
    
    @property
    def state_dict(self):
        return self.named_parameters()
    
    # def load(self, state_dict):
    #     for key in state_dict.keys():


    def add_parameter(self, k, v):
        """
        Manually add a parameter. Useful helper for scalar parameters.

        Args:
            k (str): Local name of the parameter.
            v (value): Value for the parameter.

        Returns:
            Parameter: Newly created parameter.
        """
        val = Parameter(v)
        self.__dict__["_parameters"][k] = val
        return val

    def __setattr__(self, key, val):
        if isinstance(val, Parameter):
            self.__dict__["_parameters"][key] = val
        elif isinstance(val, Module):
            self.__dict__["_modules"][key] = val
        else:
            super().__setattr__(key, val)

    def __getattr__(self, key):
        if key in self.__dict__["_parameters"]:
            return self.__dict__["_parameters"][key]

        if key in self.__dict__["_modules"]:
            return self.__dict__["_modules"][key]

        return self.__getattribute__(key)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self):
        assert False, "Not Implemented"

    def __repr__(self):
        def _addindent(s_, numSpaces):
            s = s_.split("\n")
            if len(s) == 1:
                return s_
            first = s.pop(0)
            s = [(numSpaces * " ") + line for line in s]
            s = "\n".join(s)
            s = first + "\n" + s
            return s

        child_lines = []

        for key, module in self._modules.items():
            mod_str = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append("(" + key + "): " + mod_str)
        lines = child_lines

        main_str = self.__class__.__name__ + "("
        if lines:
            # simple one-liner info, which most builtin Modules will use
            main_str += "\n  " + "\n  ".join(lines) + "\n"

        main_str += ")"
        return main_str


class Parameter:
    """
    A Parameter is a special container stored in a :class:`Module`.

    It is designed to hold a :class:`Variable`, but we allow it to hold
    any value for testing.
    """

    def __init__(self, x=None):
        self.value = x
        if hasattr(x, "requires_grad_"):
            self.value.requires_grad_(True)

    def update(self, x):
        "Update the parameter value."
        self.value = x
        if hasattr(x, "requires_grad_"):
            self.value.requires_grad_(True)

    def __repr__(self):
        return repr(self.value)

def load_state_dict(model, src_state_dict):
    missing_keys = []
    unexpected_keys = list(src_state_dict.keys())
    tgt_state_dict = model.named_parameters()
    for key in tgt_state_dict.keys():
        if key in unexpected_keys:
            tgt_state_dict[key].value._tensor = src_state_dict[key]
            unexpected_keys.remove(key)
        else:
            missing_keys.append(key)
    if len(missing_keys) != 0:
        print("missing keys: ", missing_keys)
    if len(unexpected_keys) != 0:
        print("unexpected keys", unexpected_keys)

def save(model, path):
    state_dict = model.named_variables()
    with open(path, 'wb') as f:
        pkl.dump(state_dict, f)
    
def load(path):
    with open(path, 'rb') as f:
        state_dict = pkl.load(f)
    return state_dict