'''
Cross-compare blobs with incremental adjacency, within a frame
'''

from class_cluster import ClusterStructure, NoneType
import numpy as np
import cv2

class CderBlob(ClusterStructure):

    blob = object  # not needed, common for all derBlobs in derBlob_?
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
    # DerBlob only, do not accumulate:
    neg_mB = int
    distance = int

class CBblob(ClusterStructure):

    DerBlob = object
    blob_ = list

ave_mB = -20000
ave_rM = .7  # average relative match at rL=1: rate of ave_mB decay with relative distance, due to correlation between proximity and similarity


def cross_comp_blobs(frame):
    '''
    root function of comp_blob: cross compare blobs with their adjacent blobs in frame.blob_, including sub_layers
    '''
    blob_ = frame.blob_

    for blob in blob_:  # each blob forms derBlob per compared adj_blob and accumulates adj_blobs'derBlobs:
        blob.DerBlob = CderBlob()
        comp_blob_recursive(blob, blob.adj_blobs[0], comped_id_=[blob.id])

    bblob_ = form_bblob_(blob_)  # form blobs of blobs, connected by mutual match

    visualize_cluster_(bblob_, blob_, frame)

    return bblob_


def comp_blob_recursive(blob, adj_blob_, comped_id_):
    '''
    called by cross_comp_blob to recursively compare blob to adj_blobs in incremental layers of adjacency
    '''
    for adj_blob in adj_blob_:
        if adj_blob.id in comped_id_:  # this is ids of blobs compared to immediate adjacents?
            # pseudo for assign checked id' derBlob as current derBlob:
            derBlob = @ adj_blob.id
            accum_derBlob(blob, derBlob)
        else:
            derBlob = comp_blob(blob, adj_blob)  # compare blob and adjacent blob
            accum_derBlob(blob, derBlob)  # from all compared blobs, regardless of mB sign
            comped_id_.append(adj_blob.id)

        if derBlob.mB > 0:  # replace blob with adj_blob for continuing adjacency search:
            adj_blob.DerBlob = CderBlob()
            comp_blob_recursive(adj_blob, adj_blob.adj_blobs[0], comped_id_)
        else:
            blob.DerBlob.neg_mB += derBlob.mB  # mB accumulated over comparison scope
            blob.DerBlob.distance += np.sqrt(adj_blob.A)

            if blob.Dert.M + blob.DerBlob.neg_mB > ave_mB: # negative mB, extend blob comparison to adjacents of adjacent, depth-first
                comp_blob_recursive(blob, adj_blob.adj_blobs[0], comped_id_)


def comp_blob(blob, _blob):
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

    mB = mI + mA + mG + mM - ave_mB * (ave_rM ** ((1+blob.DerBlob.distance) / np.sqrt(A)))
    # deviation from average blob match at current distance
    dB = dI + dA + dG + dM

    derBlob  = CderBlob(blob=blob, _blob=_blob, mB=mB, dB=dB)  # blob is core node, _blob is adjacent blob

    if _blob.fsliced and blob.fsliced:
        pass

    return derBlob


def form_bblob_(blob_):
    '''
    form blob of blobs as a cluster of blobs with positive adjacent derBlob_s, formed by comparing adj_blobs
    '''
    checked_ids = []  # checked for clustering, frame-wide vs. per blob
    bblob_ = []

    for blob in blob_:
        if blob.DerBlob.mB > 0 and blob.id not in checked_ids:  # init bblob with current blob
            checked_ids.append(blob.id)

            bblob = CBblob(DerBlob=CderBlob())
            accum_bblob(bblob, blob)
            form_bblob_recursive(bblob_, bblob, checked_ids)

    return bblob_

def form_bblob_recursive(bblob_, bblob, checked_ids):

    fsearch = 0  # there are blobs to check for inclusion in bblob

    for blob in bblob.blob_: # search blob' derBlob's blobs to get potential border clustering blob
        if (blob.DerBlob.mB > 0) and (blob.id not in checked_ids):  # positive mB
            for derBlob in blob.derBlob_:
                # if derBlob.blob is in bblob.blob_, but derBlob._blob is not in bblob_blob_
                # so if sum of derBlob._blob's mB with blob's mB > 0 , pack the derBlob._blob into bblob.blob_
                if (derBlob.blob in bblob.blob_) and (derBlob._blob.DerBlob.mB + blob.DerBlob.mB >0) and (derBlob._blob not in bblob.blob_):
                    merge_blob = derBlob._blob
                    fsearch = 1
                if (derBlob._blob in bblob.blob_) and (derBlob._blob.DerBlob.mB + blob.DerBlob.mB >0) and (derBlob.blob not in bblob.blob_):
                    merge_blob = derBlob.blob
                    fsearch = 1
                if fsearch:
                    accum_bblob(bblob, merge_blob)
                    checked_ids.append(blob.id)
    if fsearch:
        form_bblob_recursive(bblob_, bblob, checked_ids)

    bblob_.append(bblob) # pack bblob after scanning all possible derBlobs


def accum_derBlob(blob, derBlob):

    blob.DerBlob.accumulate(**{param:getattr(derBlob, param) for param in blob.DerBlob.numeric_params})
    blob.derBlob_.append(derBlob)

def accum_bblob(bblob, blob):

    # accumulate derBlob
    bblob.DerBlob.accumulate(**{param:getattr(blob.DerBlob, param) for param in bblob.DerBlob.numeric_params})

    if blob not in bblob.blob_:
    # isn't that checked before calling accum_bblob?
        bblob.blob_.append(blob) # pack blob into bblob.blob_

    for derBlob in blob.derBlob_: # pack adjacent blobs of blob into bblob
        if derBlob._blob not in bblob.blob_:
            bblob.blob_.append(derBlob._blob)

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
        cv2.waitKey(10)

    cv2.waitKey(0)
    cv2.destroyAllWindows()