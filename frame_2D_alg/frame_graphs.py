'''
Cross-compare blobs with incrementally mediated adjacency, forming blobs of blobs
'''

from class_cluster import ClusterStructure, comp_param
from frame_blobs import CBlob
from frame_2D_alg.vectorize_edge_blob.comp_slice import ave, ave_daangle, ave_dx, ave_Ma, ave_inv # facing error when comp-slice_ import from comp_blob, hence shift it here.
from intra_blob import intra_blob_root
import numpy as np
import cv2

ave_mB = 1   # search termination crit
ave_rM = .7  # average relative match at rL=1: rate of ave_mB decay with relative distance, due to correlation between proximity and similarity
ave_da = 0.7853  # da at 45 degree, = ga at 22.5 degree
ave_ga = .78
ave_ma = 2

ave_I = 10
ave_G = 10
ave_M = 10
ave_A = 10

param_names = ["I", "G", "M", "A"]

class CderBlob(ClusterStructure):  # set of derivatives per blob param

    p = int  # last compared blob param
    s = int  # p summed in rng
    d = int
    m  = int
    dbox = list  # directional distance, but same for all params?
    rdn  = int  # summed param rdn
    subM = int  # match from comp_sublayers, if any
    subD = int  # diff from comp_sublayers, if any
    roots = lambda: [[], []]  # [Ppm,Ppd]: Pps that derp is in, to join rderp_s, or Rderp for rderp
    blob = object
    _blob = object
    ''' old:
    blob = object
    _blob = object
    sub_H = object  # hierarchy of sub_blobs, if any '''
    # in Rderps:
    rderp_ = list  # fixed rng of comparands
    aderp = object  # anchor derp


def frame_bblobs_root(root, intra, render, verbose):
    '''
    root function of comp_blob: cross compare blobs with their adjacent blobs in frame.blob_, including sublayers
    '''
    for ifBa in 0, 1:
        if ifBa: blob_ = root.asublayers[0]
        else:    blob_ = root.rsublayers[0]

        new_root = CBblob(params=[0,0,0,0])
        for fBa in 0, 1:
            derBlob_ = cross_comp(blob_, fBa)
            bblob_ = form_bblob_(derBlob_, fBa)
            # accumulate params
            for bblob in bblob_:
                for i, param in enumerate(bblob.params):
                    new_root.params[i] += param

            if fBa: new_root.asublayers += [bblob_]
            else:   new_root.rsublayers += [bblob_]

            if intra:
                if fBa: new_root.asublayers += intra_blob_root(new_root, render, verbose, fBa=1)
                else:   new_root.rsublayers += intra_blob_root(new_root, render, verbose, fBa=0)

    return new_root


# need to update further evaluation by fBa
def cross_comp(blob_, fBa):

    derBlob_ = []
    blob_pair = []

    for _blob in blob_:
        for blob in _blob.adj_blobs[0]:  # blob_, blob.adj_blobs[1] is pose
            if [_blob, blob] not in blob_pair and [blob, _blob] not in blob_pair:  # this pair of blobs wasn't compared before
                blob_pair.append([_blob, blob])

                # I
                pI = blob.I+_blob.I
                dI = blob.I - _blob.I    # difference
                mI = ave_I - abs(dI)  # indirect match

                # G vector
                _sin, _cos = _blob.dy, _blob.dx
                sin, cos = blob.dy, blob.dx
                # sum (dy,dx)
                sin_sa = (cos * _sin) + (sin * _cos)  # sin(α + β) = sin α cos β + cos α sin β
                cos_sa = (cos * _cos) - (sin * _sin)  # cos(α + β) = cos α cos β - sin α sin β
                # diff (dy,dx)
                sin_da = (cos * _sin) - (sin * _cos)  # sin(α - β) = sin α cos β - cos α sin β
                cos_da= (cos * _cos) + (sin * _sin)   # cos(α - β) = cos α cos β + sin α sin β
                pG = np.arctan2(sin_sa, cos_sa)  # sa: sum of angle
                dG = np.arctan2(sin_da, cos_da)  # da: difference of angle
                mG = ave - abs(dG)  # ma: indirect match of angle

                # M
                pM = blob.M+_blob.M
                dM = blob.M - _blob.M    # difference
                mM = min(blob.M,_blob.M) - abs(dM)/2 - ave_M  # direct match

                # A
                pA = blob.A+_blob.A
                dA = blob.A - _blob.A    # difference
                mA = min(blob.A,_blob.A) - abs(dA)/2 - ave_A  # direct match

                pB = pI + pG + pM + pA
                dB = dI + dG + dM + dA
                mB = mI + mG + mM + mA

                derBlob = CderBlob(p=pB, d=dB, m=mB, blob=blob, _blob=_blob)
                derBlob_.append(derBlob)

    return derBlob_


# very initial draft
# need to add adj_bblob here, not sure how yet
def form_bblob_(derBlob_):

    bblob_ = []
    for derBlob in derBlob_:
        if derBlob.mB>0:  # positve derp only?
            if "bblob" not in locals():
                bblob = CBlob(dert__ = [])
                bblob_.append(bblob)
            accum_bblob(bblob, derBlob)
        else:
            if "pBlob" in locals():
                del pBlob

    return bblob_


def accum_bblob(bblob, derBlob):

    bblob.dert__.append(derBlob)
    bblob.L += 1
    bblob.I += derBlob.I
    bblob.G += derBlob.G
    bblob.M += derBlob.M
    bblob.A += derBlob.A


'''
    bblob is a cluster (graph) of blobs with positive mB to any other member blob, formed by comparing adj_blobs
    two kinds of bblobs: merged_blob and blob_graph
    bblob_ = []
    for blob in blob_:
        MB = sum([derp.mB for derp in blob.derp_]) # blob's mB, sum from blob's derps' mB
        if MB > 0 and not isinstance(blob.bblob, CpBlob):  # init bblob with current blob
            bblob = CpBlob()
            merged_ids = [bblob.id]
            accum_bblob(bblob_, bblob, blob, merged_ids)  # accum blob into bblob
            form_bblob_recursive(bblob_, bblob, bblob.blob_, merged_ids)
    # test code to see duplicated blobs in bblob, not needed in actual code
    for bblob in bblob_:
        bblob_blob_id_ = [ blob.id for blob in bblob.blob_]
        if len(bblob_blob_id_) != len(np.unique(bblob_blob_id_)):
            raise ValueError("Duplicated blobs")
    return bblob_
    '''


def comp_rng_recursive(blob, adj_blob_, _derp, derp_):
    '''
    called by cross_comp_blob to recursively compare Iblob to adj_adj_blobs, then adj_adj_adj_blobs, etc.
    '''
    derp_pair_ = [ [derp.blob, derp._blob]  for derp in derp_]  # blob, adj_blob pair

    for adj_blob in adj_blob_:
        if [adj_blob, blob] in derp_pair_:  # derp.blob=adj_blob, derp._blob=blob
            derp = derp_[derp_pair_.index([adj_blob,blob])]
            # also adj_blob.rdn += 1?
        elif [blob, adj_blob] not in derp_pair_:  # form new derp if blob pair wasn't compared in prior function call
            derp = comp_blob(blob, adj_blob, _derp)  # compare blob and adjacent blob
            derp_.append(derp)             # also frame-wide

        if "derp" in locals(): # derp exists
            # add blob accumulates base param of derp
            blob.derp_.append(derp)  # from all compared blobs, regardless of mB sign

            if derp.mB > 0:  # replace blob with adj_blob for continued adjacency search:
                comp_rng_recursive(adj_blob, adj_blob.adj_blobs[0], [], derp_)  # search depth could be different, compare anyway
                break
            elif blob.M + derp.neg_mB + derp.mB > ave_mB:  # neg mB but positive comb M,
                # extend blob comparison to adjacents of adjacent, depth-first
                derp.neg_mB = derp.mB   # mB and distance are accumulated over comparison scope
                derp.distance = np.sqrt(adj_blob.A)
                if _derp:
                     derp.neg_mB += _derp.neg_mB
                     derp.distance += _derp.distance

                comp_rng_recursive(blob, adj_blob.adj_blobs[0], derp, derp_)


def comp_blob(blob, _blob, _derp):
    '''
    compare _blob and blob
    '''
    derp = Cderp()
    layer1 = dict({'I':.0,'Da':.0, 'M':.0, 'Dady':.0,'Dadx':.0,'Ma':.0,'A':.0,'Mdx':.0, 'Ddx':.0})
    aves = [ave_inv, ave_da, ave_M, ave_daangle, ave_daangle, ave_Ma, ave_A, ave_dx]

    G = np.hypot(blob.Dy, blob.Dx) - ave * blob.A
    _G = np.hypot(_blob.Dy, _blob.Dx) - ave * _blob.A
    absG = max(1, G + (ave * blob.A))
    _absG = max(1, _G + (ave * _blob.A))
    Ga = np.hypot( np.arctan2(blob.Dydy, blob.Dxdy), np.arctan2(blob.Dydx, blob.Dxdx) ) - ave_ga * blob.A
    _Ga = np.hypot( np.arctan2(_blob.Dydy, _blob.Dxdy), np.arctan2(_blob.Dydx, _blob.Dxdx) ) - ave * blob.A
    absGa = max(1, Ga + (ave_da * blob.A))
    _absGa = max(1, _Ga + (ave_da * _blob.A))

    for param_name, param_ave in zip(layer1, aves):
        if param_name == 'Da':
            sin = blob.Dy/absG ; cos = blob.Dx/absG
            _sin = _blob.Dy/_absG; _cos = _blob.Dx/_absG
            param = [sin, cos]
            _param = [_sin, _cos]

        elif param_name == 'Dady':
            sin = blob.Dydy/absGa; cos = blob.Dxdy/absGa
            _sin = _blob.Dydy/_absGa; _cos = _blob.Dxdy/_absGa
            param = [sin, cos]
            _param = [_sin, _cos]

        elif param_name == 'Dadx':
            sin = blob.Dydx/absGa; cos = blob.Dxdx/absGa
            _sin = _blob.Dydx/_absGa; _cos = _blob.Dxdx/_absGa
            param = [sin, cos]
            _param = [_sin, _cos]

        elif param_name not in ['Da', 'Dady', 'Dadx']:
            param = getattr(blob, param_name)
            _param = getattr(_blob, param_name)

        dist_ave = param_ave * (ave_rM ** ((1 + derp.distance) / np.sqrt(blob.A)))  # deviation from average blob match at current distance
        pdert = comp_param(param, _param, param_name, dist_ave)
        layer1[param_name] = pdert
        derp.mB += pdert.m; derp.dB += pdert.d

    derp.layer1 = layer1

    if _derp:
        derp.distance = _derp.distance # accumulate distance
        derp.neg_mB = _derp.neg_mB # accumulate neg_mB

    derp.blob = blob
    derp._blob = _blob

    '''
    In comp_param:
    # G + Ave was wrong because Dy, Dx are summed as signed, resulting G is different from summed abs G 
    G = hypot(blob.Dy, blob.Dx)  
    if G==0: G=1
    _G = hypot(_blob.Dy, _blob.Dx)
    if _G==0: _G=1
    sin = blob.Dy / (G); _sin = _blob.Dy / (_G)   # sine component   = dy/g
    cos = blob.Dx / (G); _cos = _blob.Dx / (_G)   # cosine component = dx/g
    sin_da = (cos * _sin) - (sin * _cos)          # using formula : sin(α − β) = sin α cos β − cos α sin β
    cos_da = (cos * _cos) + (sin * _sin)          # using formula : cos(α − β) = cos α cos β + sin α sin β
    da = np.arctan2( sin_da, cos_da )
    ma = ave_da - abs(da)
    mB = match['I'] + match['A'] + match['G'] + match['M'] + ma \
    - ave_mB * (ave_rM ** ((1+blob.distance) / np.sqrt(blob.A)))  # deviation from average blob match at current distance
    dB = difference['I'] + difference['A'] + difference['G'] + difference['M'] + da
    derp  = Cderp(blob=blob, _blob=_blob, mB=mB, dB=dB)  # blob is core node, _blob is adjacent blob
    if _blob.fsliced and blob.fsliced:
        pass
    '''
    return derp

def form_bblob_recursive(bblob_, bblob, blob_, merged_ids):

    blobs2check = []  # list of blobs to check for inclusion in bblob

    for blob in blob_:  # search new added blobs to get potential border clustering blob

        MB = sum([derp.mB for derp in blob.derp_]) # blob's mB, sum from blob's derps' mB
        if MB > 0:  # positive summed mBs

            if isinstance(blob.bblob, CpBlob) and blob.bblob.id not in merged_ids:  # merge existing bblobs
                if blob.bblob in bblob_:
                    bblob_.remove(blob.bblob)  # remove the merging bblob from bblob_
                merge_bblob(bblob_, bblob, blob.bblob, merged_ids)
            else:
                for derp in blob.derp_:
                    # blob is in bblob.blob_, but derp._blob is not in bblob_blob_ and (derp.mB > 0 and blob.mB > 0):

                    _bblob_mB = sum([blob_derp.mB for blob_derp in derp._blob.derp_]) # blob in bblob, _blob wasn't
                    bblob_mB = sum([blob_derp.mB for blob_derp in derp.blob.derp_])   # _blob in bbob, blob wasn't

                    if (derp._blob not in bblob.blob_) and (_bblob_mB + MB > 0):
                        accum_bblob(bblob_, bblob, derp._blob, merged_ids)  # pack derp._blob in bblob
                        blobs2check.append(derp._blob)
                    elif (derp.blob not in bblob.blob_) and (bblob_mB + MB > 0):
                        accum_bblob(bblob_, bblob, derp.blob, merged_ids)
                        blobs2check.append(derp.blob)

    if blobs2check:
        form_bblob_recursive(bblob_, bblob, blobs2check, merged_ids)

    bblob_.append(bblob)  # pack bblob after scanning all accessible derps


def merge_bblob(bblob_, _bblob, bblob, merged_ids):

    merged_ids.append(bblob.id)

    for blob in bblob.blob_: # check and merge blobs of bblob
        if blob not in _bblob.blob_:
            _bblob.blob_.append(blob) # add blob to bblob
            blob.bblob = _bblob       # update bblob reference of blob

            for derp in blob.derp_: # pack adjacent blobs of blob into _bblob
                if derp.blob not in _bblob.blob_:     # if blob not in _bblob.blob_, _blob in _bblob.blob_
                    merge_blob = derp.blob
                elif derp._blob not in _bblob.blob_:  # if _blob not in _bblob.blob_, blob in _bblob.blob_
                    merge_blob = derp._blob

                if "merge_blob" in locals():
                    if isinstance(merge_blob.bblob, CpBlob):  # derp._blob is having bblob, merge them
                        if merge_blob.bblob.id not in merged_ids:
                            if merge_blob.bblob in bblob_:
                                bblob_.remove(merge_blob.bblob) # remove the merging bblob from bblob_
                            merge_bblob(bblob_, _bblob, merge_blob.bblob, merged_ids)
                    else:
                        # accumulate derp only if either one of _blob or blob (adjacent) not in _bblob
                        _bblob.accum_from(derp)
                        _bblob.accum_from(derp.blob)
                        _bblob.blob_.append(merge_blob)
                        merge_blob.bblob = _bblob


def accum_bblob(bblob_, bblob, blob, merged_ids):

    bblob.blob_.append(blob) # add blob to bblob
    blob.bblob = bblob       # update bblob reference of blob

    for derp in blob.derp_: # pack adjacent blobs of blob into bblob
        if derp.blob not in bblob.blob_:     # if blob not in bblob.blob_, _blob in bblob.blob_
            merge_blob = derp.blob
        elif derp._blob not in bblob.blob_:  # if _blob not in bblob.blob_, blob in bblob.blob_
            merge_blob = derp._blob

        if "merge_blob" in locals():
            if isinstance(merge_blob.bblob, CpBlob): # derp._blob or derp.blob is having bblob, merge them
                if merge_blob.bblob.id not in merged_ids:
                    if merge_blob.bblob in bblob_:
                        bblob_.remove(merge_blob.bblob) # remove the merging bblob from bblob_
                    merge_bblob(bblob_, bblob, merge_blob.bblob, merged_ids)
            else:
                # accumulate derp only if either one of _blob or blob (adjacent) not in bblob
                bblob.accum_from(derp)
                bblob.accum_from(derp.blob)
                bblob.blob_.append(merge_blob)
                merge_blob.bblob = bblob


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