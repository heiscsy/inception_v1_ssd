from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import ssd_training

if __name__ == '__main__':
    train_ob = ssd_training.SSDTrain()
    train_ob.train()
