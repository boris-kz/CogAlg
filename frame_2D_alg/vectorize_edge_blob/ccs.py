# generic cycle: xcomp -> cluster -> sub+ eval ) agg+ eval
import numpy as np
from frame_blobs import ave
from utils import kernel_slice_3x3 as ks
from collections import deque, namedtuple
from itertools import product

from utils import imread

class CCBase:
    graphT = namedtuple("graphT", "node_ link_")
    cnodeT = None
    def __init__(self, igraph, sub=False):
        # inputs
        self.igraph = igraph      # input graph
        self.do_sub = sub

        # outputs
        self.cgraph = None

    def evaluate(self):
        # generic cycle: xcomp -> cluster -> sub+ eval ) agg+ eval
        self.xcmp()
        self.cgraph = self.graphT([], [])    # clustered object
        self.cluster()
        if self.do_sub: self.sub()

    def xcmp(self):
        raise NotImplementedError

    def cluster(self):
        raise NotImplementedError

    def sub(self):
        raise NotImplementedError

class RecBase:
    def __init__(self, cc):
        self.cc = cc

    def evaluate(self):
        raise NotImplementedError

# Demonstration example of FrameBlobs and IntraBlob:
class FrameBlobs(CCBase):
    UNFILLED = -1
    EXCLUDED = -2
    cnodeT = blobT = namedtuple("blobT", "id sign dert fopen")
    cnodeT.G = property(lambda self: np.hypot(*self.dert[1:3]))   # G from Dy, Dx

    def xcmp(self):
        self.i__ = self.igraph
        # compute directional derivatives:
        self.dy__ = (
                (self.i__[ks.bl] - self.i__[ks.tr]) * 0.25 +
                (self.i__[ks.bc] - self.i__[ks.tc]) * 0.50 +
                (self.i__[ks.br] - self.i__[ks.tl]) * 0.25)
        self.dx__ = (
                (self.i__[ks.tr] - self.i__[ks.bl]) * 0.25 +
                (self.i__[ks.mr] - self.i__[ks.ml]) * 0.50 +
                (self.i__[ks.br] - self.i__[ks.tl]) * 0.25)
        self.g__ = np.hypot(self.dy__, self.dx__)  # compute gradient magnitude per cell
        self.s__ = ave - self.g__ > 0

    def cluster(self):
        blob_, adjt_ = self.cgraph
        Y, X = self.s__.shape
        idx = 0
        self.idx__ = np.full((Y, X), -1, 'int32')
        for __y, __x in product(range(Y), range(X)):
            if self.idx__[__y, __x] != self.UNFILLED: continue    # ignore filled/clustered derts
            sign = self.s__[__y, __x]
            fopen = False

            # flood fill the blob, start from current position
            fillQ = deque([(__y, __x)])
            while fillQ:
                _y, _x = fillQ.popleft()
                self.idx__[_y, _x] = idx
                # neighbors coordinates, 4 for -, 8 for +
                adj_yx_ = [ (_y-1,_x), (_y,_x+1), (_y+1,_x), (_y,_x-1) ]
                if sign: adj_yx_ += [(_y-1,_x-1), (_y-1,_x+1), (_y+1,_x+1), (_y+1,_x-1)] # include diagonals
                # search neighboring derts:
                for y, x in adj_yx_:
                    if (y, x) in fillQ: continue
                    if not (0<=y<Y and 0<=x<X) or self.idx__[y, x] == self.EXCLUDED: fopen = True    # image boundary is reached
                    elif self.idx__[y, x] == self.UNFILLED:    # pixel is filled
                        if self.s__[y, x] == sign: fillQ += [(y, x)]     # add to queue if same-sign dert
                    elif self.s__[y, x] != sign:            # else check if same-signed
                        adjt = (self.idx__[y, x], idx)
                        if adjt not in adjt_: adjt_ += [adjt]
            # terminate blob
            msk = (self.idx__ == idx)
            blob = self.blobT(
                id=idx,
                sign=sign,
                fopen=fopen,
                dert=np.array([
                    self.i__[ks.mc][msk].sum(),     # I
                    self.dy__[msk].sum(),           # Dy
                    self.dx__[msk].sum()]))         # Dx
            blob_ += [blob]
            idx += 1

    def sub(self):
        self.intra = IntraBlob(self)
        self.intra.evaluate()

class IntraBlob(RecBase):
    pass

if __name__ == "__main__":
    image = imread("images/raccoon_eye.jpeg")
    frame_blobs = FrameBlobs(image)
    frame_blobs.evaluate()
    import matplotlib.pyplot as plt
    img = np.full((frame_blobs.s__.shape), 128, 'uint8')
    for blob in frame_blobs.cgraph.node_:
        msk = frame_blobs.idx__ == blob.id
        img[msk] = 255 * blob.sign
    plt.imshow(img, cmap='gray')
    plt.show()