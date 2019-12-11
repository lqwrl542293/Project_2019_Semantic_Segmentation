import matplotlib.pyplot as plt
import math
import re
import pylab
from pylab import figure, show, legend

# read the log file
fp = open('/data/users/mzy/zyw/code/hrnet/log/HRNet_v2_noSyncBN_noPretrain_0.log', 'r')
train_iterations = []
train_loss = []
for ln in fp:
    if'epoch:' in ln and  'time:' in ln:
        arr = re.findall(r'epoch:\b\d+\b', ln)
        