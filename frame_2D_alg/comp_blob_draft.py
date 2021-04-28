'''
Cross-compare blobs with incremental adjacency, within a frame
'''

from class_cluster import ClusterStructure, NoneType
import numpy as np
import cv2

class CderBlob(ClusterStructure):

    blob = object  # core node
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
    blob_ = list

ave_mB = 200
ave_rM = .7  # average relative match at rL=1: rate of ave_mB decay with relative distance, due to correlation between proximity and similarity


def cross_comp_blobs(frame):
    '''
    root function of comp_blob: cross compare blobs with their adjacent blobs in frame.blob_, including sub_layers
    '''
    blob_ = frame.blob_

    for blob in blob_:  # each blob forms derBlob per compared adj_blob,
        blob.derBlob = CderBlob()  # and accumulates adj_blobs' derBlobs
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
            accum_derBlob(blob, adj_blob)  # add derBlob into blob.derBlob_

            if adj_blob.derBlob.mB>0:
                # positive mB, replace blob with adj_blob for continuing adjacency search:
                comp_blob_recursive(adj_blob, adj_blob.adj_blobs[0], checked_id_, neg_mB, distance)

            elif blob.Dert.M + neg_mB > ave_mB:  # negative mB, extend blob comparison to adjacents of adjacent, depth-first

                neg_mB += adj_blob.derBlob.mB  # mB accumulated over comparison scope
                distance += np.sqrt(adj_blob.A)
                comp_blob_recursive(blob, adj_blob.adj_blobs[0], checked_id_, neg_mB, distance)


def comp_blob(blob, _blob, distance):
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

    mB = mI + mA + mG + mM - ave_mB * (ave_rM ** ((1+distance) / np.sqrt(A)))  # deviation from average blob match at current distance
    dB = dI + dA + dG + dM

    derBlob  = CderBlob(blob=blob, _blob=_blob, mB=mB, dB=dB)  # blob is core node, _blob is adjacent blob

    if _blob.fsliced and blob.fsliced:
        pass

    return derBlob


def form_bblob_(blob_):
    '''
    form blob of blobs as a cluster of blobs with positive adjacent derBlob_s, formed by comparing adj_blobs
    '''
    checked_ids = []  # frame-wide?
    bblob_ = []

    for blob in blob_:
        if blob.id not in checked_ids:  # not checked and is core node
            neg_mB = 0; checked_ids.append(blob.id)
            if blob.derBlob.mB > 0:
                '''
                not sure how exactly to do this:
                
                for bblob in:
                    # we need to add blob.derBlob_, it's not the same as adj_blobs' derBlobs because it may include adj_adj_blob.derBlobs, etc.
                    
                    if (any of mBs in blob.derBlob_) is (any of mBs in bblob_' bblob.derBlob_):
                         accum_bblob(bblob, blob) 
                    else:
                        bblob = CBblob(derBlob=CderBlob())  
                        # init bblob with previously unconnected blob: no +mB common with any of previous bblobs' +mBs
                '''

            form_bblob_recursive(bblob, blob, neg_mB, checked_ids)

            bblob_.append(bblob)  # pack bblob after search through adjacents, all bblobs are positive

    return(bblob_)

'''
Below is not reviewed: 
'''

def form_bblob_recursive(bblob, blob, neg_mB, checked_ids):

    _omni_mB = blob.derBlob.mB
    for adj_blob in blob.adj_blobs[0]:
        if adj_blob.id not in checked_ids:
            omni_mB = adj_blob.derBlob.mB

            if (_omni_mB>0) and (omni_mB>0):    # check mutual +mB sign
                checked_ids.append(adj_blob.id) # add checked id only if same +mutual sign
                neg_mB += omni_mB               # increase neg_mB cost
                accum_bblob(bblob, adj_blob)    # accumulate mutual +mB sign blob
                form_bblob_recursive(bblob, adj_blob, neg_mB, checked_ids) # continue searching with adj_blob

            elif (blob.Dert.M + neg_mB > ave_mB):  # different sign, check ave_mB to continue searching adjacency
                neg_mB += omni_mB                  # increase neg_mB cost
                form_bblob_recursive(bblob, adj_blob, neg_mB, checked_ids) # continue searching


def accum_derBlob(_blob, blob):
    # accumulate derBlob
    _blob.derBlob.accumulate(**{param:getattr(blob.derBlob, param) for param in _blob.derBlob.numeric_params})
    _blob.derBlob_.append(blob.derBlob)

def accum_bblob(bblob, blob):

    # accumulate derBlob
    bblob.derBlob.accumulate(**{param:getattr(blob.derBlob, param) for param in bblob.derBlob.numeric_params})
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
        cv2.waitKey(10)

    cv2.waitKey(0)
    cv2.destroyAllWindows()