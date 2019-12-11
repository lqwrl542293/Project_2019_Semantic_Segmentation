import torch

path1='/data/users/mzy/fromMac/tianzhibei/weights/UNet34_seg_water_with_focalloss_newdata__2.th'
maxnum=5
init = 0
# original saved file with DataParallel
state_dict = torch.load(path1)
# create new OrderedDict that does not contain `module.`

for k, v in state_dict.items():
    print(k)
    print(v)
    init = init + 1
    if init == maxnum:
        break
