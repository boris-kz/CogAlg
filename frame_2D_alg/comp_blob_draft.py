'''
Cross-compare blobs with incrementally intermediate adjacency, within a frame
'''

from class_cluster import ClusterStructure, NoneType
import numpy as np
import cv2

class CderBlob(ClusterStructure):

    blob = object
    _blob = object
    mB = int
    dB = int
    dI = int
    mI = int
    dA = int
    mA = int
    dG = int
    mG = int
    dM = int
    mM = int

class CBblob(ClusterStructure):

    # derBlob params
    mB = int
    dB = int
    dI = int
    mI = int
    dA = int
    mA = int
    dG = int
    mG = int
    dM = int
    mM = int

    blob_ = list

ave_mB = 0  # ave can't be negative
ave_rM = .7  # average relative match at rL=1: rate of ave_mB decay with relative distance, due to correlation between proximity and similarity

def cross_comp_blobs(frame):
    '''
    root function of comp_blob: cross compare blobs with their adjacent blobs in frame.blob_, including sub_layers
    '''
    blob_ = frame.blob_

    for blob in blob_:  # each blob forms derBlob per compared adj_blob and accumulates adj_blobs'derBlobs:
        if not isinstance(blob.DerBlob, CderBlob):  # blob was not compared as adj_blob, forming derBlob
            blob.DerBlob = CderBlob()

        comp_blob_recursive(blob, blob.adj_blobs[0], derBlob_=[])
        # derBlob_ and derBlob_id_ are local and frame-wide

    bblob_ = form_bblob_(blob_)  # form blobs of blobs, connected by mutual match

    visualize_cluster_(bblob_, blob_, frame)

    return bblob_


def comp_blob_recursive(blob, adj_blob_, derBlob_):
    '''
    called by cross_comp_blob to recursively compare blob to adj_blobs in incremental layers of adjacency
    '''
    derBlob_pair_ = [ [derBlob.blob, derBlob._blob]  for derBlob in derBlob_]  # blob, adj_blob pair

    for adj_blob in adj_blob_:
        if [blob, adj_blob] in derBlob_pair_:  # blob was compared in prior function call
            break
        elif [adj_blob, blob] in derBlob_pair_:  # derBlob.blob=adj_blob, derBlob._blob=blob
            derBlob = derBlob_[derBlob_pair_.index([adj_blob,blob])]
            accum_derBlob(blob, derBlob)
            # also adj_blob.rdn += 1?
        else:  # form new derBlob
            derBlob = comp_blob(blob, adj_blob)  # compare blob and adjacent blob
            accum_derBlob(blob, derBlob)         # from all compared blobs, regardless of mB sign
            derBlob_.append(derBlob)             # also frame-wide

        if derBlob.mB > 0:  # replace blob with adj_blob for continued adjacency search:
            if not isinstance(adj_blob.DerBlob, CderBlob):  # else DerBlob was formed in previous call
                adj_blob.DerBlob = CderBlob()
            comp_blob_recursive(adj_blob, adj_blob.adj_blobs[0], derBlob_)  # search depth could be different, compare anyway
            break
        elif blob.M + blob.neg_mB + derBlob.mB > ave_mB:  # neg mB but positive comb M,
            # extend blob comparison to adjacents of adjacent, depth-first
            blob.neg_mB += derBlob.mB  # mB and distance are accumulated over comparison scope
            blob.distance += np.sqrt(adj_blob.A)
            comp_blob_recursive(blob, adj_blob.adj_blobs[0], derBlob_)


def comp_blob(blob, _blob):
    '''
    cross compare _blob and blob
    '''
    _I, _G, _M, _A = _blob.I, _blob.G, _blob.M, _blob.A
    I, G, M,  A = blob.I, blob.G, blob.M, blob.A

    dI = _I - I  # d is always signed
    mI = min(_I, I)
    dA = _A - A
    mA = min(_A, A)
    dG = _G - G
    mG = min(_G, G)
    dM = _M - M
    mM = min(_M, M)

    mB = mI + mA + mG + mM - ave_mB * (ave_rM ** ((1+blob.distance) / np.sqrt(A)))
    # deviation from average blob match at current distance
    dB = dI + dA + dG + dM

    derBlob  = CderBlob(blob=blob, _blob=_blob, mB=mB, dB=dB)  # blob is core node, _blob is adjacent blob

    if _blob.fsliced and blob.fsliced:
        pass

    return derBlob


def form_bblob_(blob_):
    '''
    bblob is a cluster (graph) of blobs with positive mB to any other member blob, formed by comparing adj_blobs
    two kinds of bblobs: merged_blob and blob_graph
    '''
    bblob_ = []
    for blob in blob_:
        if blob.DerBlob.mB > 0 and not isinstance(blob.bblob, CBblob):  # init bblob with current blob
            bblob = CBblob()
            merged_ids = [bblob.id]
            accum_bblob(bblob_, bblob, blob, merged_ids)  # accum blob into bblob
            form_bblob_recursive(bblob_, bblob, bblob.blob_, merged_ids)

    # test code to see duplicated blobs in bblob, not needed in actual code
    for bblob in bblob_:
        bblob_blob_id_ = [ blob.id for blob in bblob.blob_]
        if len(bblob_blob_id_) != len(np.unique(bblob_blob_id_)):
            raise ValueError("Duplicated blobs")

    return bblob_


def form_bblob_recursive(bblob_, bblob, blob_, merged_ids):

    blobs2check = []  # list of blobs to check for inclusion in bblob

    for blob in blob_:  # search new added blobs to get potential border clustering blob
        if (blob.DerBlob.mB > 0):  # positive summed mBs

            if isinstance(blob.bblob, CBblob) and blob.bblob.id not in merged_ids:  # merge existing bblobs
                if blob.bblob in bblob_:
                    bblob_.remove(blob.bblob)  # remove the merging bblob from bblob_
                merge_bblob(bblob_, bblob, blob.bblob, merged_ids)
            else:
                for derBlob in blob.derBlob_:
                    # blob is in bblob.blob_, but derBlob._blob is not in bblob_blob_ and (DerBlob.mB > 0 and blob.mB > 0):
                    if (derBlob._blob not in bblob.blob_) and (derBlob._blob.DerBlob.mB + blob.DerBlob.mB > 0):
                        accum_bblob(bblob_, bblob, derBlob._blob, merged_ids)  # pack derBlob._blob in bblob
                        blobs2check.append(derBlob._blob)
                    elif (derBlob.blob not in bblob.blob_) and (derBlob.blob.DerBlob.mB + blob.DerBlob.mB > 0):
                        accum_bblob(bblob_, bblob, derBlob.blob, merged_ids)
                        blobs2check.append(derBlob.blob)

    if blobs2check:
        form_bblob_recursive(bblob_, bblob, blobs2check, merged_ids)

    bblob_.append(bblob)  # pack bblob after scanning all accessible derBlobs


def merge_bblob(bblob_, _bblob, bblob, merged_ids):

    merged_ids.append(bblob.id)

    for blob in bblob.blob_: # check and merge blobs of bblob
        if blob not in _bblob.blob_:
            _bblob.blob_.append(blob) # add blob to bblob
            blob.bblob = _bblob       # update bblob reference of blob

            for derBlob in blob.derBlob_: # pack adjacent blobs of blob into _bblob
                if derBlob._blob not in _bblob.blob_:
                    if isinstance(derBlob._blob.bblob, CBblob):  # derBlob._blob is having bblob, merge them
                        if derBlob._blob.bblob.id not in merged_ids:
                            if derBlob._blob.bblob in bblob_:
                                bblob_.remove(derBlob._blob.bblob) # remove the merging bblob from bblob_
                            merge_bblob(bblob_, _bblob, derBlob._blob.bblob, merged_ids)

                    else:
                        # accumulate derBlob only if _blob (adjacent) not in _bblob
                        _bblob.DerBlob.accumulate(**{param:getattr(derBlob, param) for param in _bblob.DerBlob.numeric_params})
                        _bblob.blob_.append(derBlob._blob)
                        derBlob._blob.bblob = _bblob

def accum_derBlob(blob, derBlob):

    blob.accum_from(derBlob)
    blob.derBlob_.append(derBlob)

def accum_bblob(bblob_, bblob, blob, merged_ids):

    bblob.blob_.append(blob) # add blob to bblob
    blob.bblob = bblob       # update bblob reference of blob

    for derBlob in blob.derBlob_: # pack adjacent blobs of blob into bblob
        if derBlob._blob not in bblob.blob_:
            if isinstance(derBlob._blob.bblob, CBblob): # derBlob._blob is having bblob, merge them
                if derBlob._blob.bblob.id not in merged_ids:
                    if derBlob._blob.bblob in bblob_:
                        bblob_.remove(derBlob._blob.bblob) # remove the merging bblob from bblob_
                    merge_bblob(bblob_, bblob,derBlob._blob.bblob, merged_ids)
            else:
                # accumulate derBlob only if _blob (adjacent) not in bblob
                bblob.accum_from(derBlob)
                bblob.blob_.append(derBlob._blob)
                derBlob._blob.bblob = bblob

'''
    cross-comp among sub_blobs in nested sub_layers:
    _sub_layer = bblob.sub_layer[0]
    for sub_layer in bblob.sub_layer[1:]:
        for _sub_blob in _sub_layer:
            for sub_blob in sub_layer:
                comp_blob(_sub_blob, sub_blob)
        merge(_sub_layer, sub_layer)  # only for sub-blobs not combined into new bblobs by cross-comp
'''


def visualize_cluster_(bblob_, blob_, frame):

    colour_list = []  # list of colours:
    colour_list.append([200, 130, 1])  # blue
    colour_list.append([75, 25, 230])  # red
    colour_list.append([25, 255, 255])  # yellow
    colour_list.append([75, 180, 60])  # green
    colour_list.append([212, 190, 250])  # pink
    colour_list.append([240, 250, 70])  # cyan
    colour_list.append([48, 130, 245])  # orange
    colour_list.append([180, 30, 145])  # purple
    colour_list.append([40, 110, 175])  # brown
    colour_list.append([192, 192, 192])  # silver
    colour_list.append([255, 255, 0])  # blue2
    colour_list.append([34, 34, 178])  # red2
    colour_list.append([0, 215, 255])  # yellow2
    colour_list.append([50, 205, 50])  # green2
    colour_list.append([114, 128, 250])  # pink2
    colour_list.append([255, 255, 224])  # cyan2
    colour_list.append([0, 140, 255])  # orange2
    colour_list.append([204, 50, 153])  # purple2
    colour_list.append([63, 133, 205])  # brown2
    colour_list.append([128, 128, 128])  # silver2

    # initialization
    ysize, xsize = frame.dert__[0].shape
    blob_img = np.zeros((ysize, xsize,3)).astype('uint8')
    cluster_img = np.zeros((ysize, xsize,3)).astype('uint8')
    img_separator = np.zeros((ysize,3,3)).astype('uint8')

    # create mask
    blob_mask = np.zeros_like(frame.dert__[0]).astype('uint8')
    cluster_mask = np.zeros_like(frame.dert__[0]).astype('uint8')

    blob_colour_index = 1
    cluster_colour_index = 1

    cv2.namedWindow('blobs & clusters', cv2.WINDOW_NORMAL)

    # draw blobs
    for blob in blob_:
        cy0, cyn, cx0, cxn = blob.box
        blob_mask[cy0:cyn,cx0:cxn] += (~blob.mask__ * blob_colour_index).astype('uint8')
        blob_colour_index += 1

    # draw blob cluster of bblob
    for bblob in bblob_:
        for blob in bblob.blob_:
            cy0, cyn, cx0, cxn = blob.box
            blob_colour_index += 1
            cluster_mask[cy0:cyn,cx0:cxn] += (~blob.mask__ *cluster_colour_index).astype('uint8')

        # increase colour index
        cluster_colour_index += 1

        # insert colour of blobs
        for i in range(1,blob_colour_index):
            blob_img[np.where(blob_mask == i)] = colour_list[i % 20]

        # insert colour of clusters
        for i in range(1,cluster_colour_index):
            cluster_img[np.where(cluster_mask == i)] = colour_list[i % 20]

        # combine images for visualization
        img_concat = np.concatenate((blob_img, img_separator,
                                    cluster_img, img_separator), axis=1)

        # plot cluster of blob
        cv2.imshow('blobs & clusters',img_concat)
        cv2.resizeWindow('blobs & clusters', 1920, 720)
        cv2.waitKey(10)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def generate_unique_id(id1, id2):
    '''
    generate unique id based on id1 and id2, different order of id1 and id2 yields unique id in different sign
    '''
    # generate unique id with sign
    '''
    get sign based on order of id1 and id2, output would be +1 or -1
    id_sign = ((0.5*(id1+id2)*(id1+id2+1) + id1) - (0.5*(id2+id1)*(id2+id1+1) + id2)) / abs(id1-id2)
    modified pairing function, so that different order of a and b will generate same value
    unique_id = (0.5*(id1+id2)*(id1+id2+1) + (id1*id2)) * id_sign
    '''
    # generate unique id without sign
    unique_id = (0.5 * (id1 + id2) * (id1 + id2 + 1) + (id1 * id2))

    return unique_id