# use tracklet data from SUMMET and cluster
# measure performance using cluster purity
# i.e., what % of the tracklets in a cluster are of the same class

import torch

from src import *
from src.util import *
from src.timing import *

from data_sources.video_separation import SUMMET_Loader


summet_loader = SUMMET_Loader()
tracklets = summet_loader.tracklets
labels = summet_loader.labels
tracklets_flat = tracklets.reshape(tracklets.shape[0] * tracklets.shape[1], -1).T

print('tracklets: ', tracklets.shape)
print('flat: ', tracklets_flat.shape)
print('labels: ', labels.shape)

