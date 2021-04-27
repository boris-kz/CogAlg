'''
Cross-compare blobs with incremental adjacency, within a frame
'''

from class_cluster import ClusterStructure, NoneType
import numpy as np
import cv2

class CderBlob(ClusterStructure):

    blob = object  # core node
    _blob = object # adj blobs
    neg_mB = int
    distance = int
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

    derBlob = object
    derBlob_ = list
    blob_ = list

ave_mB = 200
ave_rM = .7  # average relative match at rL=1: rate of ave_mB decay with relative distance, due to correlation between proximity and similarity


def cross_comp_blobs(frame):
    '''
    root function of comp_blob: cross compare blobs with their adjacent blobs in frame.blob_, including sub_layers
    '''
    blob_ = frame.blob_

    for blob in blob_:  # each blob forms compared_blob_ from comparisons to blob.adj_blobs:
        comp_blob_recursive(blob, blob.adj_blobs[0], checked_id_=[blob.id],  neg_mB=0, distance=0)

    bblob_ = form_bblob_(blob_)  # form blobs of blobs, connected by mutual match

    visualize_cluster_(bblob_, blob_, frame)

    return bblob_


def comp_blob_recursive(blob, adj_blob_, checked_id_, neg_mB, distance):
    '''
    called by cross_comp_blob to recursively compare blob to adj_blobs in incremental layers of adjacency
    '''
    for adj_blob in adj_blob_:
        if adj_blob.id not in checked_id_:
            checked_id_.append(adj_blob.id)

            adj_blob.derBlob = comp_blob(blob, adj_blob, distance)  # compare blob and adjacent blob
            accum_derBlob(blob.derBlob, adj_blob.derBlob)  # add derBlob into blob.derBlob_

            if adj_blob.derBlob.mB>0:
                # positive mB, replace blob with adj_blob for continuing adjacency search:
                comp_blob_recursive(adj_blob, adj_blob.adj_blobs[0], checked_id_, distance, neg_mB)

            # negative mB, continue searching when > ave_mB
            elif blob.Dert.M + neg_mB > ave_mB:  # extend search to adjacents of adjacent, depth-first

                neg_mB += adj_blob.derBlob.mB  # mB accumulated over comparison scope
                distance += np.sqrt(adj_blob.A)
                comp_blob_recursive(blob, adj_blob.adj_blobs[0], checked_id_, distance, neg_mB)
            '''
            not needed?
            # stop searching when < ave_mB
            else:  # compared adj blobs are potential bblob elements
                for adj_adj_blob in adj_blob.adj_blobs[0]:
                    if adj_adj_blob not in blob.compared_blob_:
                        blob.compared_blob_.append(adj_adj_blob)  # potential bblob element
            '''

def accum_derBlob(_derBlob, derBlob):
    # accumulate derBlob
    _derBlob.accumulate(**{param:getattr(derBlob, param) for param in _derBlob.numeric_params})

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

    mB = mI + mA + mG + mM - ave_mB * (ave_rM ** ((1+distance) / np.sqrt(A)))  # deviation from average blob match projected at current distance
    dB = dI + dA + dG + dM

    derBlob  = CderBlob(blob=_blob, _blob = blob, mB=mB, dB=dB) # blob is core node, _blob is adjacent blob here

    if _blob.fsliced and blob.fsliced:
        pass

    return derBlob

'''
Below is not reviewed: 
'''

def form_bblob_(blob_):
    '''
    form blob of blobs as a cluster of blobs with positive adjacent derBlob_s, formed by comparing adj_blobs
    '''
    checked_ids = []
    bblob_ = []

    for blob in blob_:
        if blob.id not in checked_ids: # not checked and is head node
            neg_mB = 0; checked_ids.append(blob.id)
            bblob = CBblob(derBlob=CderBlob())   # init bblob with previous unconnected blob
            accum_bblob(bblob, blob)
            form_bblob_recursive(bblob, blob, neg_mB, checked_ids)
            bblob_.append(bblob) # pack bblob after search through adjacents

    return(bblob_)


def form_bblob_recursive(bblob, blob, neg_mB, checked_ids):

    _omni_mB = sum([derBlob.mB for derBlob in blob.derBlob_])
    omni_mB = 0

    for compared_blob in blob.compared_blob_: # get sum of omni_mB from all compared_blobs
        if compared_blob.derBlob_ and compared_blob.id not in checked_ids:
            omni_mB += sum([derBlob.mB for derBlob in compared_blob.derBlob_])

    if (_omni_mB>0) and (omni_mB>0):        # check mutual +mB sign
        neg_mB += omni_mB                   # increase neg_mB cost
        for compared_blob in blob.compared_blob_:
            if compared_blob.derBlob_ and compared_blob.id not in checked_ids:
                checked_ids.append(compared_blob.id)
                accum_bblob(bblob, compared_blob)  # accumulate compared blob's cluster
                form_bblob_recursive(bblob, compared_blob, neg_mB, checked_ids) # continue searching

    elif (blob.Dert.M + neg_mB > ave_mB):  # different sign, check ave_mB to search adjacency
        neg_mB += omni_mB                  # increase neg_mB cost
        for compared_blob in blob.compared_blob_:
            if compared_blob.id not in checked_ids:
                checked_ids.append(compared_blob.id)
                form_bblob_recursive(bblob, compared_blob, neg_mB, checked_ids) # continue searching


def accum_bblob(bblob, blob):

    for derBlob in blob.derBlob_:
        # accumulate derBlob
        bblob.derBlob.accumulate(**{param:getattr(derBlob, param) for param in bblob.derBlob.numeric_params})
        bblob.derBlob_.append(derBlob)
        bblob.blob_.append(derBlob._blob) # pack adj of blob into bblob.blob_

    bblob.blob_.append(blob) # pack blob into bblob.blob_

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