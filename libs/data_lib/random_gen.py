"""

@authors: Raffaele
"""

import tensorflow as tf
import numpy as np
import random


def set_seed(seed):
    seed = 1234
    tf.random.set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
