import numpy as np
from collections import deque

# ******************************************** OBJECT CLASSES ***********************************************************
# Object Classes:
# -frame
# -P
# -segment
# -blob
# ***********************************************************************************************************************

class cl_frame(object):
    ''' frame class hold references and buffers essential for comparison and clustering operations.
        Core objects include:
        - frame.params: to compute averages, redundant for same-scope alt_frames
        - blob_: hold buffers of local or global blobs, depends on the scope of frame
        Others include:
        - copy_dert: if True, buffer a slice of global dert__ into each blob
        - dert__: buffer of derts in 2D-array, provide spatial proximity information for inputs
        - map: boolean map for local frame inside a blob, = True inside the blob, = False outside
        provide ways to manipulate blob's dert.
    '''
    def __init__(self, input, dert_levels=0, map=None, rng=1, ncomp=2, copy_dert=False):
        " constructor function of frame "

        self.dert__ = init_dert__(input, dert_levels)  # init dert__ as a cube: depth is 1 + number of derivatives: p, g, dx, dy
        self.shape =  self.dert__.shape # shape of the frame: self.shape = (Y, X)
        self.map = map
        self.copy_dert = copy_dert
        num_params = self.shape[2] + 3  # 3 params more: xD, abs_xD, Ly
        self.params = [0] * num_params  # 7 params initially: I, G, Dx, Dy, xD, abs_xD, Ly
        self.rng = rng                  # comp range
        self.ncomp = ncomp              # number of compared inputs
        self.blob_ = []                 # buffer for terminated blobs

    def accum_params(self, params1, attr = 'params'):
        ''' accumulate a list attribute with given list.
            the attribute in question is params by default.
            could be used for accumulating any list attribute.
            for example: blob.accum_params(orientation_params, 'orientation_params')
        '''
        self.__dict__[attr] = [p + p1 for p, p1 in zip(self.__dict__[attr], params1)]
        return self

    def terminate(self):
        " frame ends, delete redundant objects "
        del self.dert__
        del self.map
        return self

    # ---------- class frame end ----------------------------------------------------------------------------------------

class cl_P(cl_frame):
    ''' P class holds P: individual 1D patterns that are continuous pixels,
        separated by sign of core parameter. P's variable attributes:
        - sign: determine the sign of P core parameter (which define the initial separate condition).
        All pixels clustered into the same P have core variable the same sign.
        - box = [x_start, x_end, y]: x-bounds and y coordinate.
        - params: initially = [L, I, G, Dx, Dy], with G being the core variable.
        P has frame class method: accum_params()
    '''

    def __init__(self, x0 = 0, num_params = 5, sign=-1):
        " constructor function of P "
        self.sign = sign            # either 0 or 1 normally. -1 for unknown sign
        self.box = [x0] # initialize box with only x_start
        self.params = [0] * num_params  # initialize params with zeroes: [L, I, G, Dx, Dy] for initial comp (default)

    def terminate(self, xn, y):
        " P ends, complete box with x_end and y "
        self.box += [xn, y]  # complete box with x_end and y
        return self

    def localize(self, x0, y0):
        " change global coordinates system to local blob coordinate system (translation) "
        self.box = [b - x0 for b in self.box[:2]] + [b - y0 for b in self.box[2:]]
        return self

    # Accessor methods ----------------------------------------
    def L(self):
        return self.params[0]

    def x0(self):
        return self.box[0]

    def xn(self):
        return self.box[1]

    # ---------- class P end --------------------------------------------------------------------------------------------

class cl_segment(cl_P):
    ''' segments are contiguous same-sign Ps, with only one P per line.
        segment's variable attributes:
        - sign: determine the sign of core parameter
        - box = [x_start, x_end, y_start, y_end]: determines the bounding box of segment
        - params: initially = [L, I, G, Dx, Dy], with G being the core variable.
        - ave_x: half the length of last-line P
        - orientation_params = [xD, abs_xD]: ave_x angle, to evaluate blob for re-orientation
        - Py_: buffer of Ps
        - roots: counter of connected lower-line Ps
        - fork_: buffer of connected higher-line segments
        segment has access to all P class method
    '''

    extend_func = (min, max, min, max)  # used by extend_box() method. shared between all instances of this class and subclasses (blob class)

    def __init__(self, P, fork_ = []):
        " constructor function of segment, initialized with P and fork_"

        self.sign = P.sign                     
        self.box = list(P.box)  # list() to make sure not to copy reference
        self.params = list(P.params)
        self.orientation_params = [0, 0]  # xD and abs_xD
        self.ave_x = (P.x0() + P.xn()) // 2
        self.Py_ = [(P, 0)]                     
        self.roots = 0                          
        self.fork_ = fork_                      

    def accum_P(self, P):
        " merge terminated P into segment "

        new_ave_x = (P.x0() + P.xn()) // 2            # new P's ave_x
        xd = new_ave_x - self.ave_x             # xd = new_ave_x - ave_x
        self.ave_x = new_ave_x                  # replace ave_x with new_ave_x
        self.accum_params([xd, abs(xd)], 'orientation_params')  # xD += xd; abs_xD += abs(xd)
        self.accum_params(P.params).extend_box(P.box[:2])   # accum params and extend x-box
        self.Py_.append((P, xd))                # buffer P into Py_
        self.roots = 0                          # reset roots
        return self

    def extend_box(self, bounds):
        ''' extend box with given bounds
            same functionality with:
            x_start, x_end, y_start, y_end = bound
            self.box[0] = min(self.box[0], x_start)
            self.box[1] = max(self.box[1], x_end)
            self.box[2] = min(self.box[2], y_start)
            self.box[3] = max(self.box[3], y_end)
            but with flexible length of bounds
        '''
        for i in range(len(bounds)):
            self.box[i] = self.extend_func[i](self.box[i], bounds[i])
        return self

    def terminate(self, y_end):
        " segment ends, complete box with y_end "
        self.box.append(y_end)
        return self

    # Accessor methods ----------------------------------------
    def y0(self):
        return self.box[2]

    def yn(self):
        return self.box[3]
    # ---------- class segment end --------------------------------------------------------------------------------------

class cl_blob(cl_segment):
    ''' blobs are 2D patterns of continuous same-sign pixels.
        same areas could be determined with flood fill algorithm,
        but blobs are more composite structure with:
        - sign: determine the sign of core parameter
        - box = [x_start, x_end, y_start, y_end]: determines the bounding box of blob
        - params: initially = [L, I, G, Dx, Dy], with G being the core variable.
        - orientation_params = [xD, abs_xD, L]: used to evaluate blob for re-orientation
        - segment_: buffer of segments
        - open_segments: counter of unfinished segments, for blob termination check
        if potentially evaluated for recursive comp:
        - dert__: a slice of outer frame, buffered for further comp
        - map: to determine which dert in dert__ belongs to current blob
        blob has access to all segment class method
    '''

    def __init__(self, segment):
        " constructor function of blob. initialize blob with initialized segment "

        self.sign = segment.sign                    # initialize blob's sign with segment's sign
        self.box = list(segment.box)  # making sure not to copy reference
        self.params = [0] * len(segment.params)     # params are initialized at 0
        self.orientation_params = [0, 0, 0]         # xD, abs_xD, Ly
        self.segment_ = []                          # buffer for segments
        self.accum_segment(segment)                 # take the first segment in
        self.open_segments = 1                      # 1 incomplete segment

    def accum_segment(self, segment):
        " buffer segment into segment_. Make segment reference to current blob "

        self.segment_.append(segment)
        segment.blob = self
        return self

    def term_segment(self, segment, y):
        " pack terminated segment into current blob "
        self.accum_params(segment.params).extend_box(segment.box[:2])  # accum params and extend x-box
        self.accum_params(segment.orientation_params + [len(segment.Py_)], 'orientation_params')  # orientation params: [xD, abs_xD, Ly]
        segment.terminate(y)    # complete segment.box
        self.open_segments += segment.roots - 1 # update open_segments accordingly
        return self

    def merge(self, fork):
        " merge fork blob into this blob "

        if not fork is self:   # check if fork blob is this blob
            self.accum_params(fork.params).extend_box(fork.box) # accum params and extend box
            self.accum_params(fork.orientation_params, 'orientation_params')   # accum orientation params: [xD, abs_xD, Ly]
            self.open_segments += fork.open_segments   # sum open segments
            for seg in fork.segment_:  # for all segments in fork blobs
                self.accum_segment(seg) # segments of other blob is buffered into this blob
        self.open_segments -= 1 # same or not, open_segments is reduced by 1 (because of fork joining)
        return self

    def localize(self, frame):
        " localize coordinate system. Get a slice of global dert_map if required "
        x0, xn, y0, yn = self.box    # x0, xn, y0, yn
        copy_dert = frame.copy_dert

        if copy_dert:  # receive a slice of global map:
            self.dert__ = frame.dert__[y0:yn, x0:xn]

        if copy_dert:  # localize inner structures:
            map = np.zeros(shape=(yn - y0, xn - x0), dtype=bool)
        for seg in self.segment_:
            seg.localize(x0, y0)
            for P, xd in seg.Py_:
                P.localize(x0, y0)
                if copy_dert:
                    x_1st, x_last, y = P.box
                    map[y, x_1st:x_last] = True   # pixels inside blob box = True

        if copy_dert:
            self.map = map
        return self
    # ---------- class blob end -----------------------------------------------------------------------------------------

# ****************************************** OBJECT CLASSES END *********************************************************

# ******************************************* GENERIC FUNCTIONS *********************************************************
# -init_dert__()
# -form_P()
# -scan_P_()
# -form_segment()
# -form_blob()
# ***********************************************************************************************************************
def init_dert__(input__, dert_levels=0):
    " initialize dert__ from of input__ "
    height, width, depth = input__.shape
    dert__ = np.zeros((height, width, (depth + dert_levels)), dtype=int)
    dert__[:, :, :depth] = input__
    return dert__

def form_P(x, y, s, dert, P, P_):
    " Initialize, accumulate, terminate 1D pattern "
    pri_s = P.sign

    if s != pri_s and pri_s != -1:
        P.terminate(x, y)  # P.box = [x0, xn, y]
        P_.append(P)
        P = cl_P(x0=x, sign=s)  # initialize P with y, x0 = x, sign = s, all params ([L, I, G, Dx, Dy, optional A, sDa]) = 0

    if pri_s == -1: P.sign = s  # new-line P.sign is -1
    P.accum_params([1] + list(dert))  # P.params [L, I, G, Dx, Dy, optional A, sDa] accumulated with [1] + dert [1, p, g, dx, dy, optional a, sda]

    return P  # accumulated within line, P_ is a buffer for conversion to _P_
    # ---------- form_P() end -------------------------------------------------------------------------------------------


def scan_P_(y, P_, seg_, frame):
    " P scans shared-x-coordinate in _P, buffered in seg , combining overlapping Ps into blobs "

    new_P_ = deque()
    if P_ and seg_:
        P = P_.popleft()
        seg = seg_.popleft()
        _P, xd = seg.Py_[-1]    # higher-line P is last element in seg.Py_

        stop = False
        fork_ = []

        while not stop:
            x0, xn = P.box[:2]
            _x0, _xn = _P.box[:2]

            if P.sign == _P.sign and _x0 < xn and x0 < _xn:  # check sign and olp
                seg.roots += 1
                fork_.append(seg)  # P-connected segments buffered into fork_

            if xn < _xn:    # P is left of _P: next P
                new_P_.append((P, fork_))
                fork_ = []
                if P_:      # switch to next P
                    P = P_.popleft()
                else:       # terminate loop
                    if seg.roots != 1:
                        form_blob(y, seg, frame)
                    stop = True
            else:                   # else: next _P
                if seg.roots != 1:
                    form_blob(y, seg, frame)

                if seg_:    # switch to new _P
                    seg = seg_.popleft()
                    _P, xd = seg.Py_[-1]
                else:       # terminate loop
                    new_P_.append((P, fork_))
                    stop = True

    # handle the remainders:
    while P_:
        new_P_.append((P_.popleft(), []))   # no fork
    while seg_:
        form_blob(y, seg_.popleft(), frame)    # seg_.popleft().roots always == 0

    return new_P_

    # ---------- scan_P_() end ------------------------------------------------------------------------------------------


def form_segment(y, P_, frame):
    " Convert ervery Ps into segments or add to higher-line segment, merge blobs "

    seg_ = deque()
    while P_:
        P, fork_ = P_.popleft()  # unpack P and it's fork_

        if not fork_:  # if no higher-line connection:
            seg = cl_segment(P, fork_)  # init segment with P and fork_: sign, box, params, orient [xD, abs_xD], ave_x, Py_, roots, fork_
            blob = cl_blob(seg)  # init blob with segment: sign, box, params, orient [xD, abs_xD, Ly]), segment_, open_segments

        else:
            if len(fork_) == 1 and fork_[0].roots == 1:  # single connection with higher-line segment, which has single lower-line connection: roots == 1
                # hP is merged into higher-line connected segment:
                seg = fork_[0]  # the only fork
                seg.accum_P(P)  # merge P into seg, accumulating params, Py_ and orientation_params

            else:  # initialize new segment with shared higher-line-connected-segment's blob:
                seg = cl_segment(P, fork_)  # seg is initialized
                blob = fork_[0].blob  # load first fork' blob
                blob.accum_segment(seg)  # add seg to fork' blob
                blob.extend_box(P.box[:2])  # extend x-box

                if len(fork_) > 1:  # terminate all fork in fork_, merge blobs joined through current seg
                    if fork_[0].roots == 1:  # segment could be terminated only if roots == 1
                        form_blob(y, fork_[0], frame)

                    for fork in fork_[1:len(fork_)]:
                        if fork.roots == 1:  # segment could be terminated only if roots == 1
                            form_blob(y, fork, frame)
                        blob.merge(fork.blob)  # merge blobs of fork forks into blob of 1st fork
        seg_.append(seg)
    return seg_
    # ---------- form_segment() end -----------------------------------------------------------------------------------------


def form_blob(y, term_seg, frame):
    " Terminate segments and completed blobs "
    blob = term_seg.blob  # blob of terminated segment
    blob.term_segment(term_seg, y)  # segments packed in blob, y_carry: min elevation of term_seg over current hP

    if not blob.open_segments:      # blob is terminated and packed into frame
        blob.terminate(term_seg.yn()).localize(frame)
        blob.rng = frame.rng        # comp range
        blob.ncomp = frame.ncomp    # number of compared inputs
        frame.accum_params(blob.params[1:] + blob.orientation_params)  # frame.params: [I, G, Dx, Dy, xD, abs_xD, Ly], orient: [xD, abs_xD, Ly]
        frame.blob_.append(blob)    # blob is buffered into blob_
    # ---------- form_blob() end ----------------------------------------------------------------------------------------

# ***************************************** GENERIC FUNCTIONS END *******************************************************