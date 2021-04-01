from collections import deque
import sys
import numpy as np
from class_cluster import ClusterStructure, NoneType
#from intra_blob import accum_blob_Dert
from frame_blobs import CBlob, flood_fill, assign_adjacents
from comp_slice_ import*

flip_ave = 10
ave_dir_val = 1
ave_M = 20

def segment_by_direction(blob, verbose=False):  # draft

    dert__ = list(blob.dert__)
    mask__ = blob.mask__
    merged_blob_ = []
    weak_dir_blob_ = []
    dy__ = dert__[1]; dx__ = dert__[2]
    # segment blob into primarily vertical and horizontal sub blobs according to the direction of kernel-level gradient:

    dir_blob_, idmap, adj_pairs = flood_fill(dert__, dy__>dx__, verbose=False, mask__=mask__, blob_cls=CBlob, accum_func=accum_dir_blob_Dert)
    assign_adjacents(adj_pairs, CBlob)

    for dir_blob in dir_blob_:
        if abs(dir_blob.G * ( dir_blob.Dy / (dir_blob.Dx+.001))) > ave_dir_val:  # direction strength eval
            if dir_blob.M > ave_M:
                dir_blob.fsliced = 1
                merged_blob_.append( slice_blob(dir_blob) )  # slice across directional sub-blob
            else:
                merged_blob_.append( dir_blob)  # returned blob is not sliced
        else:
            weak_dir_blob_.append(dir_blob)  # weak-direction sub-blob, buffer and merge if connected by assign_adjacents

    merged_blob_.append([ merge_blobs_recursive( weak_dir_blob_) ])

    return merged_blob_  # merged blobs may or may not be sliced


def merge_blobs_recursive(weak_dir_blob_):  # draft

    merged_blob_ = []
    merged_weak_dir_blob_ = []

    for blob in weak_dir_blob_:
        merge_adjacents_recursive(blob, blob.adj_blobs)  # merge dert__ and accumulate params in blob

        if abs(blob.G * ( blob.Dy / (blob.Dx+.001))) > ave_dir_val:  # direction strength eval
            if blob.M > ave_M:
                blob.fsliced = 1
                merged_blob_.append( slice_blob(blob))  # slice across directional sub-blob
            else:
                merged_blob_.append(blob)  # returned blob is not sliced
        else:
            merged_weak_dir_blob_.append(blob)  # weak-direction sub-blob, buffer and merge if connected by assign_adjacents

    if merged_weak_dir_blob_:
        merged_blob_.append([ merge_blobs_recursive( merged_weak_dir_blob_) ])

    return merged_blob_  # merged blobs may or may not be sliced


def merge_adjacents_recursive(blob, adj_blobs):

    for adj_blob, pose in blob.adj_blobs[0]:  # sub_blob.adj_blobs = [ [[adj_blob1, pose1],[adj_blob2, pose2]], A, G, M, Ga]
            if not adj_blob.fmerged:  # potential merging blob
                if abs(adj_blob.G * ( adj_blob.Dy / (adj_blob.Dx+.001))) <= ave_dir_val:

                    if blob not in adj_blob.merged_blob_:  # and adj_blob not in merged_blob_: it can't be, one pass?
                        adj_blob.fmerged = 1
                        blob = merge_blobs(blob, adj_blob)  # merge dert__ and accumulate params
                        merge_adjacents_recursive(blob, adj_blob.adj_blobs)


def merge_blobs(blob, adj_blob):
    # merge adj_blob into blob
    return blob


def accum_dir_blob_Dert(blob, dert__, y, x):
    blob.I += dert__[0][y, x]
    blob.Dy += dert__[1][y, x]
    blob.Dx += dert__[2][y, x]
    blob.G += dert__[3][y, x]
    blob.M += dert__[4][y, x]

    if len(dert__) > 5:  # past comp_a fork

        blob.Dyy += dert__[5][y, x]
        blob.Dyx += dert__[6][y, x]
        blob.Dxy += dert__[7][y, x]
        blob.Dxx += dert__[8][y, x]
        blob.Ga += dert__[9][y, x]
        blob.Ma += dert__[10][y, x]
        blob.Dir += dert__[11][y,x]