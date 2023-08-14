from SELFRec import SELFRec
from util.conf import ModelConf
import tensorflow as tf
import torch
import numpy as np
import random
import os

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    tf.set_random_seed(seed)
    torch.manual_seed(seed)

    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True


if __name__ == '__main__':

    torch.cuda.set_device(0)
    set_seed(2023)


    model = 'IDVT'
    import time

    s = time.time()
    conf = ModelConf('./conf/' + model + '.conf')
    rec = SELFRec(conf)
    rec.execute()
    e = time.time()
    print("Running time: %f s" % (e - s))
