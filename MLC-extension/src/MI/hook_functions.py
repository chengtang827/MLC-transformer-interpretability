import torch
import matplotlib.pyplot as plt
import math
import numpy as np
import itertools
import re
import einops
def inverse_softmax(probabilities, axis=0):
    """
    Calculate the original logits from softmax probabilities.
    
    Args:
        probabilities (np.array): Array of probabilities from softmax
        
    Returns:
        np.array: Original logits (up to an additive constant)
    """
    # Take the natural log of the probabilities
    logits = np.log(probabilities)
    
    # Note: The original logits are only recoverable up to an additive constant
    # We can choose to make the mean zero as one possible solution
    return logits - np.mean(logits, axis=axis, keepdims=True)

class ActivationCache:
    def __init__(self):
        self.cache = {}

    def store(self, activation, module_name:dict):
        self.cache[str(module_name)] = activation.detach().cpu().numpy()

    def get(self):
        return self.cache

    def clear(self):
        self.cache = {}


                    
def build_hook_func(hook_name:dict, activation_cache=None, mode='cache', patch_activation=None):
    """
    Build a hook function that saves the activation of a module to a cache.

    Args:
        module_name (str): Name of the module
        activation_cache (ActivationCache): Cache to save the activation to 

    Returns:
        function: Hook function
    """
    assert mode in ['cache', 'patch', 'swap']

    head = hook_name.get('head', None)

    # if patch_activation is numpy array, convert to torch tensor
    if isinstance(patch_activation, np.ndarray):
        patch_activation = torch.from_numpy(patch_activation)

    if mode == 'cache':
        def hook_func(activation, hook):
            
            if head is not None:
                activation_slice = torch.index_select(activation, dim=1, index=torch.tensor([head]))
            else:
                activation_slice = activation

            activation_cache.store(activation=activation_slice, module_name=hook_name)
            return activation
        
        return hook_func
    
    if mode == 'patch':
        def hook_func(activation, hook, patch_activation=patch_activation):
             
            if head is not None: # single head patching
            
                # patch could be [1, d_head] from null_dataset
                # or [batch, n_head, seq, d_head] from cache
                
                if activation.ndim != patch_activation.ndim:
                    patch_activation = einops.repeat(patch_activation, '1 c -> b 1 w c', 
                                                    b=activation.shape[0], 
                                                    w=activation.shape[2])

                activation[:,head,:,:] = patch_activation.squeeze(1)[:, :activation.shape[2], :]

            else:
                if activation.ndim != patch_activation.ndim:
                    # TODO need a more elegant solution
                    if patch_activation.ndim == 1:
                        patch_activation = einops.repeat(patch_activation, 'd_head -> batch seq d_head', 
                                                        batch=activation.shape[0],
                                                        seq=activation.shape[1])
                    else:
                        patch_activation = einops.repeat(patch_activation, 'n_head d_head  -> batch n_head seq d_head', 
                                                        batch=activation.shape[0],
                                                        seq=activation.shape[2])
                # in the seq dimension, reduce to the seq length of the activation
                activation_slice = torch.index_select(patch_activation, dim=-2, index=torch.arange(activation.shape[-2]))

                
                activation = activation*0+activation_slice
            
            return activation

        return hook_func
    


# def regex_match(a:str, b:str):
# # Convert wildcard pattern `b` into regex
#     a, b = str(a), str(b)
#     regex = "^" + re.escape(b).replace(r'\*', '.*') + "$"
#     return re.fullmatch(regex, a) is not None

def regex_match(a: str, b: str) -> bool:
    """
    Match two strings, supporting wildcards (*) and OR expressions in parentheses (a|b).
    
    Args:
        a: The string to match against
        b: The pattern, which can contain wildcards (*) and OR expressions (x|y)
    
    Examples:
        >>> regex_match("1", "(1|2)")  # True
        >>> regex_match("2", "(1|2)")  # True
        >>> regex_match("3", "(1|2)")  # False
        >>> regex_match("test1", "test(1|2)")  # True
        >>> regex_match("foo", "f(o|a)*")  # True
    """
    a, b = str(a), str(b)
    
    # Helper function to process the pattern
    def process_pattern(pattern: str) -> str:
        # First escape all special regex characters
        escaped = re.escape(pattern)
        
        # Unescape the special characters we want to support
        escaped = escaped.replace(r'\(', '(')  # Allow parentheses
        escaped = escaped.replace(r'\)', ')')
        escaped = escaped.replace(r'\|', '|')  # Allow OR operator
        escaped = escaped.replace(r'\*', '.*')  # Allow wildcards
        
        return f"^{escaped}$"
    
    # Create the regex pattern
    regex = process_pattern(b)
    
    try:
        return re.fullmatch(regex, a) is not None
    except re.error:
        # Handle invalid regex patterns gracefully
        return False
def add_hooks(net, mode='cache', hook_names:list[dict]=None, patch_activation:list[np.ndarray]|None=None):
    activation_cache = ActivationCache()
    hooked_modules = []
    net_modules = dict(net.named_modules())
    
    # To keep the same shape as hook_names
    if patch_activation is None:
        patch_activation = [None]*len(hook_names)

    for i_hook, hook_name in enumerate(hook_names):
        module_name = hook_name.get('module', None)
        module = net_modules[module_name]

        # print(f'Adding hook to {module_name}')
        module.add_hook(hook=build_hook_func(hook_name=hook_name, 
                                             activation_cache=activation_cache, 
                                             mode=mode, 
                                             patch_activation=patch_activation[i_hook]))
        hooked_modules.append(module)

    return activation_cache, hooked_modules




