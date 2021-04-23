'''
Cross-compare blobs with incremental adjacency, within a frame
'''

from class_cluster import ClusterStructure, NoneType
import numpy as np
import cv2

class CderBlob(ClusterStructure):

    _blob = object # adj blobs of head node
    dI = int
    mI = int
    dA = int
    mA = int
    dG = int
    mG = int
    dM = int
    mM = int
    mB = int
    dB = int

class CBblob(ClusterStructure):

    derBlob = object
    derBlob_ = list

ave_mB = 200
ave_rM = .5  # average relative match per input magnitude, at rl=1 or .5?


def cross_comp_blobs(frame):
    '''
    root function of comp_blob: cross compare blobs with their adjacent blobs in frame.blob_, including sub_layers
    '''
    blob_ = frame.blob_; checked_ids_ = []
    compared_blobs = [[],[]]  # blob, compared_blob_

    for blob in blob_:
        compared_blob_ = []  # compared to current blob, reinitialized for each new blob
        if blob.id not in checked_ids_:
            checked_ids_.append(blob.id); net_M = 0  # checked ids per blob: may be checked from multiple root blobs
            comp_blob_recursive(blob, blob.adj_blobs[0], checked_ids_, net_M, compared_blob_)  # comp between blob, blob.adj_blobs

        compared_blobs[0].append(blob)
        compared_blobs[1].append(compared_blob_)  # nested list

    bblob_ = form_bblob_(blob_, compared_blobs)  # form blobs of blobs

    visualize_cluster_(blob_,frame)

    return bblob_


def comp_blob_recursive(blob, adj_blob_, checked_ids_, neg_mB, compared_blob_):
    '''
    called by cross_comp_blob to recursively compare blob to adj_blobs in incremental layers of adjacency
    '''
    for adj_blob in adj_blob_:
        if adj_blob.id not in checked_ids_ and not adj_blob.derBlob_: # adj blob not checked and is not the head node
            checked_ids_.append(adj_blob.id)

            distance = np.sqrt(blob.A) / 2 + np.sqrt(adj_blob.A)  # approximated euclidean distance to adj_adj_blobs
            # this is not correct, distance is computed over negative mB adj_blobs only, not sure yet how to pass it

            derBlob = comp_blob(blob, adj_blob, distance)  # cross compare blob and adjacent blob
            neg_mB += derBlob.mB  # mB accumulated over comparison scope
            blob.derBlob_.append(derBlob)  # blob comparison forms multiple derBlobs, one for each adjacent blob

            if blob.Dert.M + neg_mB > ave_mB:  # extend search to adjacents of adjacent, depth-first
                comp_blob_recursive(blob, adj_blob.adj_blobs[0], checked_ids_, neg_mB, compared_blob_)

            else:  # compared adj blobs are potential bblob elements
                for adj_adj_blob in adj_blob.adj_blobs[0]:
                    if adj_adj_blob not in compared_blob_:
                        compared_blob_.append(adj_adj_blob)  # potential bblob element


def comp_blob(_blob, blob, distance):
    '''
    cross compare _blob and blob
    '''
    (_I, _Dy, _Dx, _G, _M, _Dyy, _Dyx, _Dxy, _Dxx, _Ga, _Ma, _Mdx, _Ddx), _A = _blob.Dert.unpack(), _blob.A
    (I, Dy, Dx, G, M, Dyy, Dyx, Dxy, Dxx, Ga, Ma, Mdx, Ddx), A = blob.Dert.unpack(), blob.A

    dI = _I - I  # d is always signed
    mI = min(_I, I)
    dA = _A - A
    mA = min(_A, A)
    dG = _G - G
    mG = min(_G, G)
    dM = _M - M
    mM = min(_M, M)

    mB = mI + mA + mG + mM \
         - ave_mB * ave_rM ** (1 + distance / np.sqrt(blob.A))  # average blob match projected at current distance
    dB = dI + dA + dG + dM

    # form derBlob regardless
    derBlob  = CderBlob(_blob=blob, dI=dI, mI=mI, dA=dA, mA=mA, dG=dG, mG=mG, dM=dM, mM=mM, mB=mB, dB=dB)

    if _blob.fsliced and blob.fsliced:
        pass

    return derBlob


def form_bblob_(blob_, compared_blobs):
    '''
    form blob of blobs as a cluster of blobs with positive adjacent derBlob_s, formed by comparing adj_blobs
    '''
    checked_ids = []
    bblob_ = []

    for blob, compared_blob_ in zip(blob_, compared_blobs[1]):
        if blob.derBlob_ and blob.id not in checked_ids:  # blob is initial center of adj_blobs cluster
            checked_ids.append(blob.id)

            bblob = CBblob(derBlob=CderBlob())  # init bblob
            for derBlob in blob.derBlob_:
                accum_bblob(bblob,derBlob)  # accum derBlobs into bblob

            for compared_blob in compared_blob_:  # depth first - check potential bblob element from adjacent cluster of blobs
                if compared_blob.derBlob_ and compared_blob.id not in checked_ids:
                    form_bblob_recursive(bblob, blob, compared_blob, checked_ids, compared_blobs)

            bblob_.append(bblob)  # pack bblob after checking through all adjacents

    return(bblob_)


def form_bblob_recursive(bblob, _blob, blob, checked_ids, compared_blobs):
    '''
    As distinct from form_PP_, clustered blobs don't have to be directly adjacent, checking through blob.adj_blobs should be recursive
    '''
    _Adj_mB = sum([derBlob.mB for derBlob in _blob.derBlob_])
    Adj_mB = sum([derBlob.mB for derBlob in blob.derBlob_])

    if (_Adj_mB>0) and (Adj_mB>0):  # both local-center-blob clusters must be positive
        checked_ids.append(blob.id)

        for derBlob in blob.derBlob_:
            accum_bblob(bblob,derBlob)  # accum same sign node blob into bblob

        # sorry, looks like i missed out this section previously, we need retireve compared_blob_ of blob in this section
        # not reviewed yet
        compared_blob_ = compared_blobs[1][compared_blobs[0].index(blob)]
        for compared_blob in compared_blob_: # depth first - check potential blob from adjacent cluster of blobs
            if compared_blob.derBlob_ and compared_blob.id not in checked_ids:  # potential element blob
                form_bblob_recursive(bblob, blob, compared_blob, checked_ids, compared_blobs)


def accum_bblob(bblob, derBlob):

    # for debug purpose, on the overflow issue
    print('bblob id = ' +str(bblob.id))
    print('derBlob id = ' +str(derBlob.id))
    print('bblob params : ')
    print(bblob.derBlob)
    print('derBlob params : ')
    print(derBlob)
    print('------------------------------')
    # accumulate derBlob
    bblob.derBlob.accumulate(**{param:getattr(derBlob, param) for param in bblob.derBlob.numeric_params})
    bblob.derBlob_.append(derBlob)

'''
    cross-comp among sub_blobs in nested sub_layers:
    _sub_layer = bblob.sub_layer[0]
    for sub_layer in bblob.sub_layer[1:]:
        for _sub_blob in _sub_layer:
            for sub_blob in sub_layer:
                comp_blob(_sub_blob, sub_blob)
        merge(_sub_layer, sub_layer)  # only for sub-blobs not combined into new bblobs by cross-comp
'''

# draft, to visualize the cluster of blobs, next is to visualize the merged clusters
def visualize_cluster_(blob_, frame):

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

    # initialization
    ysize, xsize = frame.dert__[0].shape
    blob_img = np.zeros((ysize, xsize,3)).astype('uint8')
    cluster_img = np.zeros((ysize, xsize,3)).astype('uint8')
    img_separator = np.zeros((ysize,3,3)).astype('uint8')

    blob_colour_index = 1
    cluster_colour_index = 1

    cv2.namedWindow('blobs & clusters', cv2.WINDOW_NORMAL)

    for blob in blob_:
        if blob.derBlob_:

            # create mask
            blob_mask = np.zeros_like(frame.dert__[0]).astype('uint8')
            cluster_mask = np.zeros_like(frame.dert__[0]).astype('uint8')

            cy0, cyn, cx0, cxn = blob.box
            # insert index of each blob
            blob_mask[cy0:cyn,cx0:cxn] += (~blob.mask__ * blob_colour_index).astype('uint8')
            blob_colour_index += 1
            # insert index of blob in cluster
            cluster_mask[cy0:cyn,cx0:cxn] += (~blob.mask__ *cluster_colour_index).astype('uint8')

            for derBlob in blob.derBlob_:
                cy0, cyn, cx0, cxn = derBlob._blob.box
                # insert index of each adjacent blob
                blob_mask[cy0:cyn,cx0:cxn] += (~derBlob._blob.mask__ * blob_colour_index).astype('uint8')
                blob_colour_index += 1
                # insert index of each adjacent blob in cluster
                cluster_mask[cy0:cyn,cx0:cxn] += (~derBlob._blob.mask__ * cluster_colour_index).astype('uint8')

            # increase colour index
            cluster_colour_index += 1

            # insert colour of blobs
            for i in range(1,blob_colour_index):
                blob_img[np.where(blob_mask == i)] = colour_list[i % 10]

            # insert colour of clusters
            for i in range(1,cluster_colour_index):
                cluster_img[np.where(cluster_mask == i)] = colour_list[i % 10]

            # combine images for visualization
            img_concat = np.concatenate((blob_img, img_separator,
                                        cluster_img, img_separator), axis=1)

            # plot cluster of blob
            cv2.imshow('blobs & clusters',img_concat)
            cv2.resizeWindow('blobs & clusters', 1920, 720)
            cv2.waitKey(100)

    cv2.waitKey(0)
    cv2.destroyAllWindows()