# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from yacs.config import CfgNode as CN

_C = CN()

_C.ROOT = '/data/public/datasets/tianzhi_kemu2/tianzhi/aug_train/color'
_C.EXPNAME = '' #需要在yaml文件中声明
_C.USESYNCBN = True
_C.train = False


# Cudnn related params
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

# common params for NETWORK
_C.MODEL = CN()
_C.MODEL.NAME = 'seg_hrnet'
_C.MODEL.PRETRAINED = ''
_C.MODEL.EXTRA = CN(new_allowed=True)

_C.LOSS = CN()
_C.LOSS.USE_OHEM = False
_C.LOSS.OHEMTHRES = 0.9
# _C.LOSS.OHEMKEEP = 100000
#_C.LOSS.CLASS_BALANCE = False

# DATASET related params
_C.DATASET = CN()
# _C.DATASET.ROOT = ''
# _C.DATASET.DATASET = 'cityscapes'
_C.DATASET.NUM_CLASSES = 6
# _C.DATASET.TRAIN_SET = 'list/cityscapes/train.lst'
# _C.DATASET.EXTRA_TRAIN_SET = ''
# _C.DATASET.TEST_SET = 'list/cityscapes/val.lst'

# training
_C.TRAIN = CN()

_C.TRAIN.BATCHSIZE_PER_CARD = 4
_C.TRAIN.CARD_NUM = 4
_C.TRAIN.TRAINCLASS = -1
_C.TRAIN.LR = 2e-4
_C.TRAIN.TOTALEPOCH = 300
_C.TRAIN.IMAGE_SIZE = [1024, 512]  # width * height
_C.TRAIN.SHUFFLE = True
_C.TRAIN.NUM_WORKERS = 8
_C.TRAIN.LOSS = 'CrossEntropy'
_C.TRAIN.RESUME = True
_C.TRAIN.RESUME_START = 29
_C.TRAIN.OPTIMIZER = 'sgd'
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.WD = 0.0001



# _C.TRAIN.LR_FACTOR = 0.1
# _C.TRAIN.LR_STEP = [90, 110]
# _C.TRAIN.LR = 0.01
# _C.TRAIN.EXTRA_LR = 0.001


# _C.TRAIN.NESTEROV = False
# _C.TRAIN.IGNORE_LABEL = -1
#
# _C.TRAIN.RESUME = False
#
# _C.TRAIN.BATCH_SIZE_PER_GPU = 32
#
# # only using some training samples
# _C.TRAIN.NUM_SAMPLES = 0

# testing
_C.TEST = CN()
_C.TEST.LABEL_LIST = []
_C.TEST.ROOT = ''
_C.TEST.WEIGTH = ''
_C.TEST.IMAGE_SIZE = 1024
_C.TEST.BASE_SIZE = 2048

_C.TEST.BATCH_SIZE_PER_GPU = 32
# only testing some samples
_C.TEST.NUM_SAMPLES = 0

_C.TEST.MODEL_FILE = ''
_C.TEST.FLIP_TEST = False
_C.TEST.MULTI_SCALE = False
_C.TEST.SCALE_LIST = [1]

# debug
_C.DEBUG = CN()
_C.DEBUG.DEBUG = False
_C.DEBUG.SAVE_BATCH_IMAGES_GT = False
_C.DEBUG.SAVE_BATCH_IMAGES_PRED = False
_C.DEBUG.SAVE_HEATMAPS_GT = False
_C.DEBUG.SAVE_HEATMAPS_PRED = False


def update_config(cfg, args):
    cfg.defrost()

    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    cfg.freeze()


if __name__ == '__main__':
    import sys

    # with open(sys.argv[1], 'w') as f:
    #     print(_C, file=f)
    print(_C)

