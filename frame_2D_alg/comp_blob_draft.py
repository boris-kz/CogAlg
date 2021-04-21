'''
Cross-compare blobs with incremental adjacency, within a frame
'''

from class_cluster import ClusterStructure, NoneType

class CderBlob(ClusterStructure):

    _blob = object # not necessary if we are gonna use blob.adj_blobs to get adjacent derBlob
    blob = object
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
    sub_derBlob = list  # was for recursion with sub blobs, not needed

class CBblob(ClusterStructure):

    derBlob = object
    derBlob_ = list


ave_M = 100
ave_mP = 100
ave_dP = 100


def cross_comp_blobs(blob_):
    '''
    root function of comp_blob: cross compare blobs with their adjacent blobs in frame.blob_, including sub_layers
    '''
    for blob in blob_:
        checked_ids_ = [blob.id]; net_M = 0  # checked ids per blob: may be checked from multiple root blobs
        comp_blob_recursive(blob, blob.adj_blobs[0], checked_ids_, net_M)  # comp between blob, blob.adj_blobs

    bblob_ = form_bblob_(blob_)

    print(str(len(bblob_)+" cluster of blobs is formed."))
    return bblob_


def form_bblob_(blob_):
    '''
    form blob of blobs as a cluster of blobs positive adjacent derBlob_s, with compared_adj_blobs
    '''
    checked_ids = []
    bblob_ = []

    for blob in blob_:
        if blob.derBlob_ and blob.id not in checked_ids:  # blob is initial center of adj_blobs cluster
            checked_ids.append(blob.id)

            bblob = CBblob(derBlob=CderBlob())  # init bblob
            for derBlob in blob.derBlob_:
                accum_bblob(bblob,derBlob)  # accum derBlobs into bblob

            for compared_blob in blob.compared_blobs:  # depth first - check potential element from adjacent cluster of blobs
                if compared_blob.derBlob_ and compared_blob.id not in checked_ids:
                    form_bblob_recursive(bblob, blob, compared_blob, checked_ids)

            bblob_.append(bblob)  # pack bblob after checking through all adjacents

    '''
    bblob_ = []
    checked_ids = []
    for derBlob in derBlob__[1:]:
        if derBlob.id not in checked_ids: # current derBlob is not checked before
            checked_ids.append(derBlob.id)
            bblob = CBblob(derBlob=CderBlob()) # init new bblob
            accum_bblob(bblob,derBlob)         # accum derBlob into bblob
            for adj_blob in derBlob.blob.adj_blobs[0]:
                if adj_blob.derBlob.id not in checked_ids: # adj blob's derBlob is not checked before
                    form_bblob_recursive(bblob, derBlob, adj_blob.derBlob, checked_ids) # recursively add blob to bblob via the adjacency
            bblob_.append(bblob) # pack bblob after checking through all adjacents
    '''

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


def comp_blob_recursive(blob, adj_blob_, checked_ids_, net_M):
    '''
    called by cross_comp_blob to recursively compare blob to adj_blobs in incremental layers of adjacency
    '''

    for adj_blob in adj_blob_:
        if adj_blob.id not in checked_ids_:
            checked_ids_.append(adj_blob.id)

            derBlob = comp_blob(blob, adj_blob)  # cross compare blob and adjacent blob
            net_M += adj_blob.Dert.M            # cost of extending the cross comparison

            # blob comparison forms multiple derBlobs: blob.derBlob_, one with each of adjacent blobs
            blob.derBlob_.append(derBlob)

            if blob.Dert.M - net_M > ave_M:  # extend search to adjacents of adjacent, depth-first
                comp_blob_recursive(blob, adj_blob.adj_blobs[0], checked_ids_, net_M)

            else:  # compared adj blobs are potential bblob elements
                for adj_adj_blob in adj_blob.adj_blobs[0]:
                    if adj_adj_blob not in blob.compared_blobs:
                        blob.compared_blobs.append(adj_adj_blob)  # potential bblob element



def comp_blob(_blob, blob):
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
    mB = mI + mA + mG + mM
    dB = dI + dA + dG + dM

    # form derBlob regardless
    derBlob  = CderBlob(_blob=_blob, blob=blob, dI=dI, mI=mI, dA=dA, mA=mA, dG=dG, mG=mG, dM=dM, mM=mM, mB=mB, dB=dB)

    if _blob.fsliced and blob.fsliced:
        pass

    return derBlob

'''
    cross-comp among sub_blobs in nested sub_layers:
    _sub_layer = bblob.sub_layer[0]
    for sub_layer in bblob.sub_layer[1:]:
        for _sub_blob in _sub_layer:
            for sub_blob in sub_layer:
                comp_blob(_sub_blob, sub_blob)
        merge(_sub_layer, sub_layer)  # only for sub-blobs not combined into new bblobs by cross-comp
'''