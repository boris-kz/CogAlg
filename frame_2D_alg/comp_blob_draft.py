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

ave_M = 10
ave_mP = 100
ave_dP = 100


def cross_comp_blobs(frame):
    '''
    root function of comp_blob: cross compare blobs with their adjacent blobs in frame.blob_, including sub_layers
    '''
    blob_ = frame.blob_
    compared_blobs = []
    checked_ids_ = []
    for blob in blob_:
        compared_blob_ = []  # compared to current blob
        if blob.id not in checked_ids_:
            checked_ids_.append(blob.id); net_M = 0  # checked ids per blob: may be checked from multiple root blobs
            comp_blob_recursive(blob, blob.adj_blobs[0], checked_ids_, net_M, compared_blob_)  # comp between blob, blob.adj_blobs
            a = 1
        compared_blobs.append([compared_blob_])

    bblob_ = form_bblob_(blob_, compared_blobs)  # form blobs of blobs

    visualize_cluster_(blob_,frame)

    return bblob_


def comp_blob_recursive(blob, adj_blob_, checked_ids_, net_M, compared_blobs):
    '''
    called by cross_comp_blob to recursively compare blob to adj_blobs in incremental layers of adjacency
    '''
    for adj_blob in adj_blob_:
        if adj_blob.id not in checked_ids_ and not adj_blob.derBlob_:
            checked_ids_.append(adj_blob.id)

            derBlob = comp_blob(blob, adj_blob)  # cross compare blob and adjacent blob
            net_M += adj_blob.Dert.M            # cost of extending the cross comparison
            # blob comparison forms multiple derBlobs: blob.derBlob_, one with each of adjacent blobs
            blob.derBlob_.append(derBlob)

            if blob.Dert.M - net_M > ave_M:  # extend search to adjacents of adjacent, depth-first
                net_M *= 5 # increase cost for depth search
                comp_blob_recursive(blob, adj_blob.adj_blobs[0], checked_ids_, net_M, compared_blobs)

            else:  # compared adj blobs are potential bblob elements
                for adj_adj_blob in adj_blob.adj_blobs[0]:
                    if adj_adj_blob not in compared_blobs:
                        compared_blobs.append(adj_adj_blob)  # potential bblob element


def comp_blob(_blob, blob):
    '''
    cross compare _blob and blob
    '''
    (_I, _Dy, _Dx, _G, _M, _Dyy, _Dyx, _Dxy, _Dxx, _Ga, _Ma, _Mdx, _Ddx), _A = _blob.Dert.unpack(), _blob.A
    (I, Dy, Dx, G, M, Dyy, Dyx, Dxy, Dxx, Ga, Ma, Mdx, Ddx), A = blob.Dert.unpack(), blob.A

    # should we scale down their value? Their values gonna be very large, in millions after the accumulation
    dI = int((_I - I)/255)  # d is always signed
    mI = int(min(_I, I)/255)
    dA = _A - A
    mA = min(_A, A)
    dG = int((_G - G) /255)
    mG = int(min(_G, G)/255)
    dM = int((_M - M)/255)
    mM = int(min(_M, M)/255)
    mB = mI + mA + mG + mM
    dB = dI + dA + dG + dM

    # form derBlob regardless
    derBlob  = CderBlob(_blob=blob, dI=dI, mI=mI, dA=dA, mA=mA, dG=dG, mG=mG, dM=dM, mM=mM, mB=mB, dB=dB)

    if _blob.fsliced and blob.fsliced:
        pass

    return derBlob


def form_bblob_(blob_, compared_blob_):
    '''
    form blob of blobs as a cluster of blobs with positive adjacent derBlob_s, formed by comparing adj_blobs
    '''
    checked_ids = []
    bblob_ = []

    for blob, compared_blobs in zip(blob_, compared_blob_):
        if blob.derBlob_ and blob.id not in checked_ids:  # blob is initial center of adj_blobs cluster
            checked_ids.append(blob.id)

            bblob = CBblob(derBlob=CderBlob())  # init bblob
            for derBlob in blob.derBlob_:
                accum_bblob(bblob,derBlob)  # accum derBlobs into bblob

            for compared_blob in compared_blobs:  # depth first - check potential bblob element from adjacent cluster of blobs
                if compared_blob.derBlob_ and compared_blob.id not in checked_ids:
                    form_bblob_recursive(bblob, blob, compared_blob, checked_ids)

            bblob_.append(bblob)  # pack bblob after checking through all adjacents

    return(bblob_)


def form_bblob_recursive(bblob, _blob, blob, checked_ids):
    '''
    As distinct from form_PP_, clustered blobs don't have to be directly adjacent, checking through blob.adj_blobs should be recursive
    '''
    _Adj_mB = sum([derBlob.mB for derBlob in _blob.derBlob_])
    Adj_mB = sum([derBlob.mB for derBlob in blob.derBlob_])

    if (_Adj_mB>0) and (Adj_mB>0):  # both local-center-blob clusters must be positive
        checked_ids.append(blob.id)

        for derBlob in blob.derBlob_:
            accum_bblob(bblob,derBlob)  # accum same sign node blob into bblob

        for compared_blob in blob.compared_blobs: # depth first - check potential blob from adjacent cluster of blobs
            if compared_blob.derBlob_ and compared_blob.id not in checked_ids:  # potential element blob
                form_bblob_recursive(bblob, blob, compared_blob, checked_ids)


def accum_bblob(bblob, derBlob):

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

# draft, to visualize the cluster of blobs
def visualize_cluster_(blob_, frame):

    colour_list = []  # list of colours:
    colour_list.append([192, 192, 192])  # silver
    colour_list.append([200, 130, 1])  # blue
    colour_list.append([75, 25, 230])  # red
    colour_list.append([25, 255, 255])  # yellow
    colour_list.append([75, 180, 60])  # green
    colour_list.append([212, 190, 250])  # pink
    colour_list.append([240, 250, 70])  # cyan
    colour_list.append([48, 130, 245])  # orange
    colour_list.append([180, 30, 145])  # purple
    colour_list.append([40, 110, 175])  # brown

    ysize, xsize = frame.dert__[0].shape
    cluster_img = np.zeros((ysize, xsize,3)).astype('uint8')
    colour_index = 0

    for blob in blob_:
        if blob.derBlob_:

            cluster_mask = np.ones_like(frame.dert__[0]).astype('bool')

            cy0, cyn, cx0, cxn = blob.box
            cluster_mask[cy0:cyn,cx0:cxn] = np.logical_and(blob.mask__, cluster_mask[cy0:cyn,cx0:cxn])

            for derBlob in blob.derBlob_:
                cy0, cyn, cx0, cxn = derBlob._blob.box
                cluster_mask[cy0:cyn,cx0:cxn] = np.logical_and(derBlob._blob.mask__, cluster_mask[cy0:cyn,cx0:cxn])

            cluster_img[np.where((1-cluster_mask)>0)] = colour_list[colour_index % 10]

            colour_index += 1

            cv2.namedWindow('clusters')
            cv2.imshow('clusters',cluster_img)
            cv2.resizeWindow('clusters', 1920, 720)
            cv2.waitKey(200)

    cv2.waitKey(1000)
    cv2.destroyAllWindows()
