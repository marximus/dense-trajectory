"""
See Also
--------
http://starship.python.net/crew/theller/ctypes/tutorial.html
https://docs.python.org/3/library/ctypes.html
"""

import ctypes
import os
import multiprocessing

import numpy as np
import pandas as pd
from numpy.ctypeslib import ndpointer

from extras import cache
from extras.arrfuncs import stack_structured_arrays
from extras.progress import progress_bar


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


def densetrack(video, track_length=15, min_distance=5, patch_size=32, nxy_cell=2, nt_cell=3,
               scale_num=8, init_gap=1, 
               poly_n=7, poly_sigma=1.5,  # used in Farneback method in C++ code
               ret_type='array'):
    """
    Compute dense trajectories for video.

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
    ret_type : str
        Type of returned trajectories. Must be 'list', 'dataframe', or 'array'.

    Returns
    -------
    tracks : dict, ndarray, or dataframe
        Dense trajectories. Type of returned value depends on `ret_type`.
    """
    ret_fields = ('frame_num', 'mean_x', 'mean_y', 'var_x', 'var_y', 'length',
                  'scale', 'x_pos', 'y_pos', 't_pos', 'coords', 'trajectory',
                  'hog', 'hof', 'mbh_x', 'mbh_y')

    if not video.flags['C_CONTIGUOUS']:
        video = np.ascontiguousarray(video)

    # all Python types except integers, strings, and bytes objects have to be wrapped in their
    # corresponding ctypes type, so that they can be converted to the required C data type.
    # https://docs.python.org/3.6/library/ctypes.html#calling-functions-continued
    #poly_sigma = ctypes.c_double(poly_sigma)
    #print('Python - poly_n: {}'.format(poly_n))
    #print('Python - poly_sigma: {}'.format(poly_sigma))
    #print()

    data = _densetrack(
        video, video.shape[0], video.shape[1], video.shape[2],
        track_length, min_distance, patch_size, nxy_cell, nt_cell, scale_num, init_gap,
        poly_n, poly_sigma
    )

    if ret_type == 'list':
        tracks = [dict(zip(ret_fields, track)) for track in data]
    elif ret_type == 'dataframe':
        tracks = pd.DataFrame(data, columns=ret_fields)
    elif ret_type == 'array':
        types = (np.int, np.float, np.float, np.float, np.float, np.float, np.float,
                 np.float, np.float, np.float, np.float, np.float, np.float, np.float,
                 np.float, np.float)
        shapes = (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, (track_length+1,2), (track_length,2),
                  8*nxy_cell*nxy_cell*nt_cell, 9*nxy_cell*nxy_cell*nt_cell, 8*nxy_cell*nxy_cell*nt_cell,
                  8*nxy_cell*nxy_cell*nt_cell)
        assert len(types) == len(ret_fields) == len(shapes)
        tracks = np.array(data, dtype=list(zip(ret_fields, types, shapes)))
    else:
        raise ValueError("ret_type must be 'list' or 'dict'")

    return tracks


def _compute_or_load_tracks(args):
    """
    ** This should NOT be called directly. It is used in compute function **

    Compute dense trajectories for a video.

    If `cache_dir` is specified, the trajectories for a video will only be computed if no
    entry exists in the cache for the given `vid_path`.

    `func_kwargs` will be passed to function computing trajectories.

    Parameters
    ----------
    vid_path : str
        Path to .npy file of video.
    cache_dir : str or None
        Path to cache directory, or None if no cache should be used.
    func_kwargs : dict
        Keyword arguments passed to function that computes trajectories.

    Returns
    -------
    vid_path : str
        Same as input.
    vid_tracks : structured array
        Dense trajectories for video.
    """
    vid_path, cache_dir, func_kwargs = args

    if cache_dir:
        cache_key = vid_path.replace('/', '_').replace('.', '_') + '.npy'
        cache_path = os.path.join(cache_dir, cache_key)

        if os.path.exists(cache_path):  # Load trajectories from cache
            vid_tracks = np.load(cache_path)
        else:  # Compute trajectories and save to cache
            frames = np.load(vid_path)
            vid_tracks = densetrack(frames, **func_kwargs)
            np.save(cache_path, vid_tracks)
    else:  # Compute trajectories
        frames = np.load(vid_path)
        vid_tracks = densetrack(frames, **func_kwargs)

    return vid_path, vid_tracks


def compute(rois, cache_root=None, inherit_fields=None, show_progress=False, **kwargs):
    """
    Compute dense trajectories for ROIs.

    `rois` is a pandas.DataFrame that must have a column named 'video', which contains the paths to the
    .npy video files of the ROIs.

    If `inherit_fields` is not None, it should contain the attributes (column names of pandas.DataFrame) 
    of the ROIs which should be inherited by the dense trajectories. The returned dense trajectories will 
    have the fields specified in `inherit_fields`.

    Any extra keyword arguments will be passed to the _compute function.

    If `cache_root` is not None, the ROIs will be cached (using their .npy video paths as keys) for each unique
    set of `kwargs`. For each unique set of `kwargs`, a directory will be created in `cache_root` that contains
    the cached dense trajectories of the ROIs.

    Parameters
    ----------
    rois : pandas.DataFrame
        ROIs for which to compute dense trajectories. The DataFrame must have the column 'video', which
        contains the paths to the .npy video files of the ROIs.
    cache_root : str
        Directory in which to cache the results based on .npy file path and kwargs.
    inherit_fields : list of str or None
        If not None, the attributes of the ROIs that will be inherited by the dense trajectories. They must
        be column names of the ROIs in `rois`.
    show_progress : bool
        If True, show progress bar while computing dense trajectories.
    **kwargs
        Arguments passed to the _compute function. See the _compute function for details.

    Returns
    -------
    tracks : structured ndarray
        Dense trajectories as a structured ndarray, where the fields are attributes of the dense 
        trajectories. Any columns in `rois` will be inherited by the dense trajectories.
    """
    # Update default parameters with user supplied kwargs. The parameters will be used to create a separate
    # cache for each different combination of parameters.
    params = dict(
        track_length=15, min_distance=5, patch_size=32, nxy_cell=2, nt_cell=3, 
        scale_num=8, init_gap=1, poly_n=5, poly_sigma=1.1
    )

    for arg in kwargs.keys():
        if arg not in params:
            raise ValueError('Invalid keyword argument: {}'.format(arg))
    params.update(kwargs)

    # Create the directory in which to cache the dense trajectories for this specific set of parameters.
    cache_dir = cache.get_cache_dir(cache_root, params) if cache_root else None

    # Compute the dense trajectories
    tracks = dict()
    with multiprocessing.Pool() as pool:
        fargs = list(zip(rois['video'], [cache_dir]*len(rois['video']), [params]*len(rois['video'])))
        for i, (vid_path, vid_tracks) in enumerate(pool.imap_unordered(_compute_or_load_tracks, fargs)):
            tracks[vid_path] = vid_tracks
            if show_progress:
                progress_bar(i+1, len(fargs))
    tracks = [tracks[vid_path] for vid_path in rois['video']]

    # Stack the dense trajectories into a single numpy structured array. Any attributes
    # of the ROIs will be inherited by their corresponding dense trajectories.
    fields = [(name, rois[name].values) for name in inherit_fields] if inherit_fields else []
    tracks = stack_structured_arrays(tracks, append_fields=fields)

    # Remove any trajectories with nans.
    nan_indices = []
    for field in tracks.dtype.names:
        if tracks[field].dtype.descr[0][1] != '|O':  # skip numpy arrays containing objects
            nan_ind = np.where(np.isnan(tracks[field]))[0]
            nan_indices.extend(nan_ind)
    if len(nan_indices) > 0:
        nan_indices = np.unique(nan_indices)
        print('Removing {} trajectories with nan'.format(len(nan_indices)))
        tracks = np.delete(tracks, nan_indices, axis=0)

    return tracks

