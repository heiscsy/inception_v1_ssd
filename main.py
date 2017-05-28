from __future__ import print_function
from __future__ import division
from __future__ import absolute_import


from ssd_training import SSDTrain
import tensorflow as tf
import numpy as np
import struct

if __name__ == '__main__':
    train_ob = SSDTrain()
    train_ob.train()
