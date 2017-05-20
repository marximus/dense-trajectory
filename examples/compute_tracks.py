import sys
from os.path import join, dirname, abspath

import numpy as np

# add path so that densetrack can be imported
sys.path.append(dirname(dirname(abspath(__file__))))
import densetrack

video = np.load(join(dirname(__file__), "sample_data", "7069-6_roi_0.npy"))
tracks = densetrack.densetrack(video)
print('Computed {} tracks'.format(len(tracks)))
