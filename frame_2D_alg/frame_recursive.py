from class_cluster import ClusterStructure, NoneType, comp_param, Cdert
from frame_blobs import *
from frame_bblobs import *


def frame_recursive(image, intra, render, verbose):

    frame = frame_blobs_root(image, intra, render, verbose)
    frame = frame_bblobs_root(frame, intra, render, verbose)

    types_ = [[0],[1],[2],[3]]  # each I, G, M, A, only params here?

    return  frame_level_root(frame, types_)


def frame_level_root(frame, types_):

    sublayer0 = frame.sublayers[-1]
    new_sublayer0 = []
    frame.sublayers = [new_sublayer0]  # or add to existing sublayers?

    nextended = 0  # number of extended-depth P_s
    new_types_ = []
    new_M = 0

    for pblob_, types in zip(sublayer0, types_):

        if len(pblob_) > 2 and sum([pblob.M for pblob_ in sublayer0 for pblob in pblob_]) > ave_M:  # 2: min aveN, will be higher
            nextended += 1  # nesting depth of this P_ will be extended

            derp_t = cross_comp(pblob_)

            # no rdn?
            # not using param_name now
            for param, derp_ in enumerate(derp_t):  # derp_ -> Pps:

                new_types = types.copy()
                new_types.insert(0, param)  # add param index
                new_types_.append(new_types)

                pblob_ = form_bblob_(derp_)
                new_sublayer0 += [pblob_]

                # intra section here

                new_M += sum([pblob.M for pblob in pblob_])
        else:
            new_types_ += [[] for _ in range(4)]  # align indexing with sublayer, replace with count of missing prior pblob_s

    if len(sublayer0) / max(nextended,1) < 4 and new_M > ave_M * 4:  # ave_extend_ratio and added M, will be default if pipelined
        # add levels for frame too?
        if len(sublayer0) / max(nextended,1) < 8 and new_M > ave_M * 8:
            frame_level_root(frame, new_types_)  # try to add new level

    norm_feedback(frame)



def norm_feedback(frame):
    pass