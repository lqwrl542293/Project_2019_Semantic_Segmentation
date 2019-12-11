import argparse

from config import config
from config import update_config

#更新参数代码
def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args


args = parse_args()

# print(config)
extra = config.MODEL.EXTRA
# print(extra)
print(extra.STAGE1)