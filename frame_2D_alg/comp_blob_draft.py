'''
Cross-compare blobs with incremental adjacency, within a frame
'''

from class_cluster import ClusterStructure, NoneType

class CderBlob(ClusterStructure):

    _blob = object
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
    sub_derBlob = list  # why sub, this should already be packed in sub_blobs?

class CBblob(ClusterStructure):

    derBlob = object
    derBlob_ = list
    blob_ = list

ave_M = 100
ave_mP = 100
ave_dP = 100


def cross_comp_blobs(blob_):
    '''
    root function of comp_blob: cross compare blobs with their adjacent blobs in frame.blob_, including sub_layers
    '''
    derBlob__ = []
    for blob in blob_:

        checked_ids_ = [blob.id]; net_M = 0  # checked ids per blob: may be checked from multiple root blobs
        derBlob__ += comp_blob_recursive(blob, blob.adj_blobs[0], checked_ids_, net_M)  # comp between blob, blob.adj_blobs

    bblob_ = form_bblob_(derBlob__)

    return bblob_

# draft, similar with derP_2_PP
def form_bblob_(derBlob__):
    '''
    form bblob with same sign derBlob
    '''
    bblob_ = []
    checked_ids = []

    for derBlob in derBlob__[1:]:
        if derBlob.id not in checked_ids: # current derBlob is not checked before
            checked_ids.append(derBlob.id)

            bblob = CBblob(derBlob=CderBlob()) # init new bblob
            accum_bblob(bblob,derBlob)         # accum derBlob into bblob

            if derBlob._blob.derBlob.id not in checked_ids: # current derBlob is not checked before
                form_bblob_recursive(bblob, derBlob, derBlob._blob.derBlob, checked_ids) # recursively add blob to bblob via the adjacency

            bblob_.append(bblob) # pack bblob after checking through all adjacents

    return(bblob_)


def form_bblob_recursive(bblob, _derBlob, derBlob, checked_ids):
    '''
    As distinct from form_PP_, consecutive blobs don't have to be adjacent, that needs to be checked through blob.adj_blobs
    '''

    checked_ids.append(derBlob.id)

    if (_derBlob.mB>0) == (derBlob.mB>0):        # same sign check for adjacent derBlob
        accum_bblob(bblob,derBlob._blob.derBlob) # accum same sign derBlob into bblob

        if derBlob._blob.derBlob.id not in checked_ids: # current derBlob is not checked before
            form_bblob_recursive(bblob, derBlob, derBlob._blob.derBlob, checked_ids)  # recursively add blob to bblob via the adjacency


def accum_bblob(bblob, derBlob):

    # accumulate derBlob
    bblob.derBlob.accumulate(**{param:getattr(derBlob, param) for param in bblob.derBlob.numeric_params})
    # add derBlob and blob to list
    bblob.derBlob_.append(derBlob)
    bblob.blob_.append(derBlob.blob)


def comp_blob_recursive(blob, adj_blob_, checked_ids_, net_M):
    '''
    called by cross_comp_blob to recursively compare blob to adj_blobs in incremental layers of adjacency
    '''
    derBlob_ = []

    for adj_blob in adj_blob_:
        if adj_blob.id not in checked_ids_:

            derBlob = comp_blob(blob,adj_blob)  # cross compare blob and adjacent blob
            net_M += adj_blob.Dert.M          # cost of extending the cross comparison
            checked_ids_.append(adj_blob.id)
            derBlob_.append(derBlob)

            if blob.Dert.M - net_M > ave_M:  # if crit: search adjacents of adjacent, depth-first
                derBlob_ += comp_blob_recursive(blob, adj_blob.adj_blobs[0], checked_ids_, net_M)

            # if blob.fsliced and adj_blob.fsliced: this should be in comp_blob

    return derBlob_


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
    blob.derBlob = derBlob

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