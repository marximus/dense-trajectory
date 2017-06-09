"""
See Also
--------
http://starship.python.net/crew/theller/ctypes/tutorial.html
https://docs.python.org/3/library/ctypes.html
"""
import ctypes
import os

import numpy as np
import pandas as pd
from numpy.ctypeslib import ndpointer

# for read_video
os.environ['IMAGEIO_FFMPEG_EXE'] = '/home/mam588/miniconda3/envs/cilia/bin/ffmpeg'  # for imageio
import imageio
import cv2

PARENTDIR = os.path.dirname(__file__)
LIBPATH = os.path.join(PARENTDIR, "dense_trajectory_release_v1.2", "release", "DenseTrack.so")

_lib = ctypes.CDLL(LIBPATH)

_densetrack = _lib.densetrack
_densetrack.argtypes = [
    ndpointer(dtype=np.uint8, ndim=3, flags="C_CONTIGUOUS"), # frames
    ctypes.c_size_t,     # len
    ctypes.c_size_t,     # rows
    ctypes.c_size_t,     # cols
    ctypes.c_int,        # track_length
    ctypes.c_int,        # min_distance
    ctypes.c_int,        # patch_size
    ctypes.c_int,        # nxy_cell
    ctypes.c_int,        # nt_cell
    ctypes.c_int,        # scale_num
    ctypes.c_int,        # init_gap
    ctypes.c_int,        # poly_n
    ctypes.c_double      # poly_sigma
]
_densetrack.restype = ctypes.py_object


def densetrack(
        video,
        track_length=15,
        min_distance=5,
        patch_size=32,
        nxy_cell=2,
        nt_cell=3,
        scale_num=8,
        init_gap=1,
        poly_n=7,
        poly_sigma=1.5,  # used in Farneback method in C++ code
):
    """Compute dense trajectories for video.

    Parameters
    ----------
    video : array of uint8, shape (frames, height, width)
        Greyscale video from which to compute trajectories.
    track_length : int
        Length of the trajectories.
    min_distance : int
        Sampling stride. The stride for dense sampling feature points.
    patch_size : int
        The neighborhood size for computing the descriptor.
    nxy_cell : int
        Spatial cells. The number of cells in the nxy axis.
    nt_cell : int
        Temporal cells. The number of cells in the nt axis.
    scale_num : int
        The number of maximal spatial scales.
    init_gap : int
        The gap for re-sampling feature points.
    poly_n : int
    poly_sigma : float

    Returns
    -------
    tracks : type specified by ret_type
        Dense trajectories. Type of returned value depends on `ret_type`.

    The returned trajectories will have the following attributes:
        frame_num
            The trajectory ends on which frame.
        mean_x
            The mean value of the x coordinates of the trajectory.
        mean_y
            The mean value of the y coordinates of the trajectory.
        var_x
            The variance of the x coordinates of the trajectory.
        var_y
            The variance of the y coordinates of the trajectory.
        length
            The length of the trajectory.
        scale
            The trajectory is computed on which scale.
        x_pos
            The normalized x position w.r.t. the video (0~0.999), for spatio-temporal pyramid.
        y_pos
            The normalized y position w.r.t. the video (0~0.999), for spatio-temporal pyramid.
        t_pos
            The normalized t position w.r.t. the video (0~0.999), for spatio-temporal pyramid.
        trajectory: shape (2x[trajectory length]) (default 30 dimension)
        HOG: ndarray, shape (8x[spatial cells]x[spatial cells]x[temporal cells] (default 96 dimension)
        HOF: 9x[spatial cells]x[spatial cells]x[temporal cells] (default 108 dimension)
        MBHx: 8x[spatial cells]x[spatial cells]x[temporal cells] (default 96 dimension)
        MBHy: 8x[spatial cells]x[spatial cells]x[temporal cells] (default 96 dimension)
    """
    ret_fields = ['frame_num', 'mean_x', 'mean_y', 'var_x', 'var_y', 'length',
                  'scale', 'x_pos', 'y_pos', 't_pos', 'coords', 'trajectory',
                  'hog', 'hof', 'mbh_x', 'mbh_y']

    if not video.flags['C_CONTIGUOUS']:
        video = np.ascontiguousarray(video)

    data = _densetrack(
        video, video.shape[0], video.shape[1], video.shape[2],
        track_length, min_distance, patch_size, nxy_cell, nt_cell, scale_num, init_gap,
        poly_n, poly_sigma
    )

    # tracks = [dict(zip(ret_fields, track)) for track in data]  # uncomment to return list
    # tracks = pd.DataFrame(data, columns=ret_fields)  # uncomment to return dataframe

    types = (np.int, np.float, np.float, np.float, np.float, np.float, np.float,
             np.float, np.float, np.float, np.float, np.float, np.float, np.float,
             np.float, np.float)
    shapes = (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, (track_length+1,2), (track_length,2),
              8*nxy_cell*nxy_cell*nt_cell, 9*nxy_cell*nxy_cell*nt_cell, 8*nxy_cell*nxy_cell*nt_cell,
              8*nxy_cell*nxy_cell*nt_cell)
    tracks = np.array(data, dtype=list(zip(ret_fields, types, shapes)))

    return tracks


def densetrack_fileio(infile, outfile, max_frames=None, **kwargs):
    """
    Load video from infile, compute trajectories and save trajectories to outfile. 
    
    Parameters
    ----------
    infile : str
        Video file. Must be one of: .npy or .mov, .avi, .mpg, .mpeg, .mp4, .mkv, .wmv.
    outfile : str
        Path where trajectories will be saved as numpy array. Must have extension .npy.
    max_frames : int or None
        Maximum number of frames to compute trajectories for. If None, they will be 
        computed for all frames.
    **kwargs : keyword arguments
        Keyword arguments are passed to densetrack function.
    
    Returns
    -------
    None
    """
    infile_ext = os.path.splitext(infile)[1]
    if infile_ext == '.npy':
        video = np.load(infile)
        video = video[:max_frames]
    elif infile_ext in ('.mov', '.avi', '.mpg', '.mpeg', '.mp4', '.mkv', '.wmv'):
        video = read_video(infile, max_frames)
    else:
        raise ValueError('infile has unsupported extension')

    outfile_ext = os.path.splitext(outfile)[1]
    if outfile_ext != '.npy':
        raise ValueError('outfile must be a .npy file')

    tracks = densetrack(video, **kwargs)
    np.save(outfile, tracks)


def read_video(filename, max_frames=None):
    reader = imageio.get_reader(filename)

    nframes = reader.get_meta_data()['nframes']
    width, height = reader.get_meta_data()['size']

    nframes = min(nframes, np.inf if max_frames is None else max_frames)

    video = np.zeros((nframes, height, width), dtype=np.uint8)
    for i, im in enumerate(reader):
        if i == nframes:
            break
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        video[i] = gray

    return video
