#!/usr/bin/env python

import sys
import os
import inspect
import argparse
from os.path import splitext, basename, dirname, abspath, join, exists

import numpy as np
from joblib import Parallel, delayed

sys.path.append(dirname(dirname(abspath(__file__))))
import densetrack


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # required arguments
    parser.add_argument('video', type=str,
                        help='path to video or directory of videos')
    parser.add_argument('outdir', type=str, nargs='?', default=os.getcwd(),
                        help='path to directory where output will be saved')

    # optional arguments
    parser.add_argument('--overwrite', dest='overwrite', action='store_true',
                        help='overwrite existing files')
    parser.add_argument('--max-frames', type=int,
                        help='maximum number of frames to use')
    parser.add_argument('--n-jobs', type=int, default=-1,
                        help='number of concurrently running jobs')

    # dense trajectory arguments
    group = parser.add_argument_group('trajectory parameters')
    group.add_argument('--track-length', type=int, default=10,
                        help='length of trajectories')
    group.add_argument('--min-distance', type=int, default=5,
                        help='sampling stride for dense sampling feature points')
    group.add_argument('--patch-size', type=int, default=32,
                        help='neighborhood size for computing the descriptor')
    group.add_argument('--nxy-cell', type=int, default=2,
                        help='number of cells in the nxy axis')
    group.add_argument('--nt-cell', type=int, default=3,
                        help='number of cells in the nt axis')
    group.add_argument('--scale-num', type=int, default=8,
                        help='number of maximal spatial scales')
    group.add_argument('--init-gap', type=int, default=1,
                        help='gap for resampling feature points')
    group.add_argument('--poly-n', type=int, default=7, help='TODO')
    group.add_argument('--poly-sigma', type=float, default=1.5, help='TODO')

    args = parser.parse_args()

    # output directory name based on trajectory parameters
    group_track = {g.title: g for g in parser._action_groups}['trajectory parameters']
    args_track = {a.dest: getattr(args, a.dest, None) for a in group_track._group_actions}

    outdir = join(args.outdir, '_'.join(['{}-{}'.format(k, args_track[k]) for k in sorted(args_track.keys())]))

    # create list of input files
    if os.path.isdir(args.video):
        infile = [os.path.join(args.video, f) for f in os.listdir(args.video)]
    else:
        infile = [args.video]

    # only keep files with appropriate file extensions
    exts = ('.npy', '.mov', '.avi', '.mpg', '.mpeg', '.mp4', '.mkv', '.wmv')
    infile = np.array([f for f in infile if splitext(f)[1] in exts])

    # set output file names
    outfile = np.array([join(outdir, splitext(basename(f))[0]+'.npy') for f in infile])

    # ignore existing videos if overwrite is False
    if not args.overwrite:
        exists = np.array(list(map(exists, outfile)))
        infile = infile[~exists]
        outfile = outfile[~exists]

    # create output directory
    if not os.path.exists(outdir):
        print('Creating {}'.format(outdir))
        os.makedirs(outdir)

    Parallel(n_jobs=args.n_jobs, verbose=30)(delayed(densetrack.densetrack_fileio)(
        ifile, ofile, args.max_frames, **args_track
    ) for ifile, ofile in zip(infile, outfile))
