import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import random

DELTA = 1e-8


def set_global_seed(seed):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def print_to_output(message=None, title=None, overoneline=True, verbose=True):
    """
    This function is used to simplify the printing task. You can
    print a message with a title, over one line or not for lists.
    """
    if verbose:
        # We print the title between '----'
        if title is not None:
            print('\n' + title.center(70, '-') + '\n')
        # We print the message
        if message is not None:
            # Lists have particular treatment
            if isinstance(message, list):
                # Either printed over one line
                if overoneline:
                    to_print = ''
                    for i in message:
                        to_print += '%s | ' % str(i)
                    print(to_print[:-3])
                # Or not
                else:
                    for i in message:
                        print(i)
            else:
                print(message)


def center(x):
    return x - x.mean()


def reduce_(x):
    return x / (x.std() + DELTA)


def normalize(x):
    return reduce_(center(x))
