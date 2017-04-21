from os.path import join, dirname
import numpy as np

import densetrack

video = np.load(join(dirname(__file__), "sample_data", "7069-6_roi_0.npy"))
print(video.shape)
tracks = densetrack.densetrack(video)
print(len(tracks))
