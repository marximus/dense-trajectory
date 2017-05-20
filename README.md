# Dense Trajectory
Python wrapper for dense trajectory code downloaded from [this link](http://lear.inrialpes.fr/people/wang/download/dense_trajectory_release_v1.2.tar.gz).

Project Homepage: http://lear.inrialpes.fr/people/wang/dense_trajectories

The directory [dense_trajectory_release_v1.2](dense_trajectory_release_v1.2) contains the dense trajectory code with the following modifications:
  - [DenseTrack.cpp](dense_trajectory_release_v1.2/DenseTrack.cpp): Changed main function to accept options as parameters to function, rather than reading them from
          the command line. It was also modified to return a python object with the results, rather than printing results
          to standard output.
  - [Makefile](dense_trajectory_release_v1.2/Makefile): Updated to compile shared library rather than executable.
  - Updated to work with Python 3.
  - Updated to work with OpenCV 3.
  
## Installation
First update [Makefile](dense_trajectory_release_v1.2/Makefile) with user dependent variables. Then run
```
cd dense_trajectory_release_v1.2
make
```
This will create the shared library *DenseTrack.so* in the directory 'dense_trajectory_release_v1.2/release/'. *densetrack.py* 
uses ctypes to load *DenseTrack.so* and call the function in the shared library to compute the dense trajectories.

## Usage
```
import densetrack
tracks = densetrack.densetrack(video)
```
For a full example see [examples/compute_tracks.py](examples/compute_tracks.py).
