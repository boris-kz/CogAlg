import cv2
import math
from collections import deque
# Constrains:
rng = 1                 # number of pixels compared to each pixel in four directions
max_index = rng - 1     # max index of rng_dert1_ and rng_dert2_
min_coord = rng * 2 - 1 # min x and y for form_P input: der2 from comp over rng
ave = 15                 # |g| value that coincides with average match: gP filter
coordsStr = ('min_x', 'max_x', 'min_y', 'max_y')
paramsStr = ('L', 'I', 'G', 'sG', 'Dx', 'Dy', 'A', 'Da', 'sDa')
# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# ***************************************************** THE PATTERN CLASS ***********************************************
# Methods:
# -__init__()
# -getParams()
# -getCoords()
# -accumParams()
# -extendCoords()
# ***********************************************************************************************************************
class pattern(object):
    def __init__(self, s):
        " creates a pattern "
        self.sign = s
        for param in paramsStr[:-3]:    # L, I, G, sG, Dx, Dy, A, Da, sDa; excludes A, Da and sDa for now
            self.__dict__[param] = 0
    # ------------------------------------------------- __init__() end --------------------------------------------------
    def getParams(self):
        return [self.__dict__[param] for param in paramsStr[:-3]]   # L, I, G, sG, Dx, Dy, A, Da, sDa; excludes A, Da and sDa for now
    # ------------------------------------------------- getParams() end -------------------------------------------------
    def getCoords(self):
        return [self.__dict__[coord] for coord in coordsStr if coord in self.__dict__]
    # ------------------------------------------------- getCoords() end -------------------------------------------------
    def accumParams(self, params, i = 0):
        " accumulates params "
        n = len(params)
        while i < n and paramsStr[i] in self.__dict__:    # paramsStr is in Constrains at the top
            self.__dict__[paramsStr[i]] += params[i]
            i += 1
        return self
    # ------------------------------------------------- accumParams() end -----------------------------------------------
    def extendCoords(self, coords):
        " extends min/max x and min/max y "
        i = 0; n = len(coords)
        while i < n and coordsStr[i] in self.__dict__:    # coordsStr is in Constrains at the top
            self.__dict__[coordsStr[i]] = min(coords[i], self.__dict__[coordsStr[i]])   # min coordinate
            if coordsStr[i + 1] in self.__dict__:   # if the max coordinate exists
                self.__dict__[coordsStr[i + 1]] = min(coords[i + 1], self.__dict__[coordsStr[i + 1]])
            i += 2  # 1 dimension each step
        return self
    # ------------------------------------------------- extendCoords() end ----------------------------------------------
# ***************************************************** PATTERN CLASS END ***********************************************
# ***************************************************** THE FRAME CLASS *************************************************
# Methods:
# -__init__()
# -reset()
# -accumBlob()
# -extractBlob()
# -lateralComp()
# -verticalComp()
# -formP()
# -scanP_()
# -formSegment()
# -formBlob()
# ***********************************************************************************************************************
class frame(pattern):   # Inherits methods from pattern class
    ''' Descriptions from frame_blob_grad.py, some outdated:
    frame_blobs_grad() defines blobs by gradient, vs. dx and dy.I did that in frame_old, trying it again due to suggestion by Stephan Verbeeck
    gradient is estimated as hypot(dx, dy) of a quadrant with +dx and +dy, in vertical_comp before form_P call.

    Complemented by intra_blob (recursive search within blobs), it will be 2D version of first-level core algorithm.
    Blob is a contiguous area of positive or negative derivatives from cross-comparison among adjacent pixels within an image.

    Cross-comparison forms match and difference between pixels in horizontal (m, d) and vertical (my, dy) dimensions, and these four
    derivatives define four types of blobs. This version defines d | dy blobs only inside negative m | my blobs,
    while frame_blobs_olp (overlap) defines each blob type over full frame.

    frame_blobs() performs several levels (Le) of encoding, incremental per scan line defined by vertical coordinate y.
    value of y per Le line is shown relative to y of current input line, incremented by top-down scan of input image:

    1Le, line y:    x_comp(p_): lateral pixel comparison -> tuple of derivatives der ) array der_
    2Le, line y-rng to y: y_comp(der_): vertical pixel comp -> 2D tuple der2 ) array der2_
    3Le, line y-rng: form_P(der2) -> 1D pattern P
    4Le, line y-rng-1: scan_P_(P, hP) -> hP, roots: down-connections, fork_: up-connections between Ps
    5Le, line y-rng-2: form_segment(hP, seg) -> seg: merge vertically-connected _Ps in non-forking blob segments
    6Le, line y-rng-2 to y-rng-1: form_blob(seg, blob): merge connected segments in fork_' incomplete blobs, recursively
    if y >= rng * 2 + 3: line y-rng == P_, line y-rng-1 == hP_, line y-rng-2 == seg_, line y-rng-3 == blob_

    Pixel comparison in 2D forms lateral and vertical differences per pixel, combined into gradient.
    They are formed on the same level because average lateral match ~ average vertical match.
    Orientation increases primary dimension of blob to maximize match, and decreases secondary dimension to maximize difference.
    Subsequent union of lateral and vertical blobs is by combined match of their parameters, orthogonal sign is not commeasurable.

    Initial pixel comparison is not novel, I design from the scratch to make it organic part of hierarchical algorithm.
    It would be much faster with matrix computation, but this is minor compared to higher-level processing.
    I implement it sequentially for consistency with accumulation into blobs: irregular and very difficult to map to matrices.

    All 2D functions (y_comp, scan_P_, form_segment, form_blob) input two lines: higher and lower,
    convert elements of lower line into elements of new higher line, then displace elements of old higher line into higher function.
    Higher-line elements include additional variables, derived while they were lower-line elements.

    prefix '_' denotes higher-line variable or pattern, vs. same-type lower-line variable or pattern,
    postfix '_' denotes array name, vs. same-name elements of that array '''
    def __init__(self, path, i=0):
        " read input from file, then compute blobs "
        input = cv2.imread(path, i).astype(int)
        self.reset()
        self.extractBlobs(input)
    # ------------------------------------------------- __init__() end --------------------------------------------------
    def reset(self):
        " reset params "
        self.I = 0
        self.G = 0
        self.sG = 0
        self.Dx = 0
        self.Dy = 0
        self.blob_ = []
        return self
    # ------------------------------------------------- reset() end -----------------------------------------------------
    def accumBlob(self, blob):
        self.blob_.append(blob)
        self.accumParams(blob.getParams(), i=1)    # i = 1: Ignore L
        return self
    # ------------------------------------------------- accumBlob() end -------------------------------------------------
    def extractBlobs(self, image):
        ''' Main body of the operation,
        postfix '_' denotes array vs. element,
        prefix '_' denotes higher-line vs. lower-line variable '''
        # init:
        self.Y, self.X = image.shape
        self.dert__ = [[0] * self.X] * rng   # complete dert are buffered into 2D array for later references
        _P_ = deque()
        rng_dert2__ = []                        # horizontal line of vertical buffers: 2D array of 2D tuples, deque for speed?
        pixel_ = image[0, :]                    # first line of pixels
        dert1_ = self.lateralComp(pixel_)
        for (p, d) in dert1_:
            dert2 = p, d, 0                     # dy is initialized at 0
            rng_dert2_ = deque(maxlen=rng)      # vertical buffer of incomplete derivatives tuples, for fuzzy ycomp
            rng_dert2_.append(dert2)            # only one tuple in first-line rng_der2_
            rng_dert2__.append(rng_dert2_)
        # main:
        for self.y in range(1, self.Y):         # or Y-1: default term_blob in scan_P_ at y = Y?
            pixel_ = image[self.y, :]           # vertical coordinate y is index of new line p_
            dert1_ = self.lateralComp(pixel_)   # lateral pixel comparison
            rng_dert2__, _P_ = self.verticalComp(dert1_, rng_dert2__, _P_)  # vertical pixel comparison
        # frame ends, last vertical rng of incomplete rng_der2__ is discarded,
        # merge segs of last line into their blobs:
        while _P_:  self.formBlob(self.formSegment(_P_.popleft()))
        return self
    # ------------------------------------------------- getBlobs() end --------------------------------------------------
    def lateralComp(self, pixel_):
        ''' Comparison over x coordinate,
         within rng of consecutive pixels on each line '''
        # init:
        dert1_ = []
        rng_dert1_ = deque(maxlen=rng)  # incomplete dert1s, within rng from input pixel: summation range < rng
        rng_dert1_.append((0, 0))
        # main:
        for x, p in enumerate(pixel_):  # pixel p is compared to rng of prior pixels within horizontal line, summing d per prior pixel
            back_d = 0
            for index, (pri_p, d) in enumerate(rng_dert1_):
                id = p - pri_p
                d += id
                back_d += id
                if index < max_index:
                    rng_dert1_[index] = (pri_p, d)
                elif x > min_coord:  # after pri_p comp over rng
                    dert1_.append((pri_p, d))  # completed bilateral tuple is transferred from rng_dert_ to dert_
            rng_dert1_.appendleft((p, back_d))
        # last incomplete rng_dert1_ in line are discarded, vs. dert1_ += reversed(rng_dert1_)
        return dert1_
    # ------------------------------------------------- lateralComp() end -----------------------------------------------
    def verticalComp(self, dert1_, rng_dert2__, _P_):
        ''' Comparison to bilateral rng of vertically consecutive pixels,
        forming der2: pixel + lateral and vertical derivatives '''
        # init:
        P = initP(rng)          # incomplete lateral pattern, dert_ is contained by the whole frame instead of in P
        P_ = deque()            # P buffer for next-line
        buff_ = deque()         # line y - rng * 2 - 1: _Ps buffered by previous run of scan_P_
        new_rng_dert2__ = deque()   # 2D array: line of rng_dert2_s buffered for next-line comp
        dert_ = [0] * rng    # complete dert_
        self.x = rng            # lateral coordinate of pixel in input dert1
        # main:
        for (p, dx), rng_dert2_ in zip(dert1_, rng_dert2__):  # pixel comp to rng _pixels in rng_dert2_, summing dy
            back_dy = 0
            for index, (_p, _dx, _dy) in enumerate(rng_dert2_):  # vertical derivatives are incomplete; prefix '_' denotes higher-line variable
                idy = p - _p
                _dy += idy
                back_dy += idy
                if index < max_index:
                    rng_dert2_[index] = (_p, _dx, _dy)
                elif self.y > min_coord:
                    g = int(math.hypot(_dy, _dx))   # no explicit angle, quadrant is indicated by signs of d and dy
                    sg = g - ave * rng * 2          # match is defined as below-average gradient
                    dert = _p, g, sg, _dx, _dy
                    dert_.append(dert)
                    P = self.formP(dert, P, P_, buff_, _P_)
            rng_dert2_.appendleft((p, dx, back_dy))  # new der2 displaces completed one in vertical rng_der2_ via maxlen
            new_rng_dert2__.append(rng_dert2_)      # 2D array of vertically-incomplete 2D tuples, converted to rng_der2__, for next-line vertical comp
            self.x += 1
        if self.y > min_coord:
            self.dert__.append(dert_)
        return new_rng_dert2__, P_
    # ------------------------------------------------- verticalComp() end ----------------------------------------------
    def formP(self, dert, P, P_, buff_, _P_):
        " Initializes, accumulates and terminates 1D pattern "
        p, g, sg, dx, dy = dert  # 2D tuple of derivatives per pixel, "y" denotes vertical vs. lateral derivatives

        s = 1 if sg > 0 else 0
        pri_s = P.sign
        if s != pri_s and pri_s != -1:  # P is terminated:
            P.max_x = self.x  # P's max_x
            if self.y == min_coord + 1:  # 1st line P_ is converted to init hP_
                P_.append((P, []))  # P, _fork_
            else:
                self.scanP_(P, P_, buff_, _P_)  # P scans hP_
            P = initP(self.x, sign=s)  # new P initialization
        # continued or initialized input and derivatives are accumulated:
        P.accumParams([1, p, g, sg, dx, dy])
        P.sign = s
        if self.x == self.X - rng - 1:  # P is terminated:
            P.max_x = self.X - rng  # P's max_x
            if self.y == min_coord + 1:  # 1st line P_ is converted to init hP_;  scan_P_(), form_segment(), form_blob() use one type of Ps, hPs, buffs
                P_.append((P, []))  # P, _fork_
            else:
                self.scanP_(P, P_, buff_, _P_)  # P scans hP_
            P = initP(rng)  # next line P initialization
        return P  # accumulated within line, P_ is a buffer for conversion to _P_
    # ------------------------------------------------- formP() end -----------------------------------------------------
    def scanP_(self, P, P_, _buff_, hP_):
        " P scans shared-x-coordinate hPs in higher P_, combines overlapping Ps into blobs "
        fork_ = []  # refs to hPs connected to input P
        _min_x = 0  # to start while loop, next ini_x = _x + 1
        min_x, max_x = P.getCoords()
        while _min_x <= max_x:  # while x values overlap between P and _P
            if _buff_:
                hP = _buff_.popleft()  # hP was extended to segment and buffered in prior scan_P_
            elif hP_:
                hP = self.formSegment(hP_.popleft())
            else:
                break  # higher line ends, all hPs are converted to segments
            _P = hP.Py_[-1][0]
            _min_x, _max_x = _P.getCoords()  # hP.Py_[-1][0]

            if P.sign == _P.sign and min_x <= _max_x and _min_x <= max_x:
                hP.roots += 1
                fork_.append(hP)  # P-connected hPs will be converted to segments at each _fork

            if _max_x > max_x:  # x overlap between hP and next P: hP is buffered for next scan_P_, else hP included in a blob segment
                _buff_.append(hP)
            elif hP.roots != 1:
                self.formBlob(hP)  # segment is terminated and packed into its blob
            _min_x = _max_x + 1  # = first x of next _P
        P_.append((P, fork_))  # P with no overlap to next _P is extended to hP and buffered for next-line scan_P_
    # ------------------------------------------------- scanP_() end ----------------------------------------------------
    def formSegment(self, hP):
        " Convert hP into new segment or add it to higher-line segment, merge blobs "
        _P, fork_ = hP
        ave_x = (_P.L - 1) // 2  # extra-x L = L-1 (1x in L)

        if not fork_:  # seg is initialized with initialized blob (params, coordinates, root_, incomplete_segments)
            hP = initSegment(_P, fork_, self.y - rng - 1, ave_x)
            blob = initBlob(hP)
            hP.blob = blob      # every segment has a reference to its blob
        else:
            if len(fork_) == 1 and fork_[0].roots == 1:  # hP has one fork fork_[0] and that fork has one root: hP
                # hP is merged into higher-line blob segment
                fork = fork_[0]
                fork.extendCoords(_P.getCoords()).accumParams(_P.getParams())   # extends coords and accumulates params
                xd = ave_x - fork.ave_x
                fork.xD += xd
                fork.ave_x = ave_x
                fork.Py_.append((_P, xd))  # Py_: vertical buffer of Ps merged into seg
                fork.roots = 0              # reset roots
                hP = fork  # replace segment with including fork's segment
                blob = hP.blob
            else:  # if >1 forks, or 1 fork that has >1 roots:
                hP = initSegment(_P, fork_, self.y - rng - 1, ave_x)  # seg is initialized with fork's blob
                blob = hP.blob = fork_[0].blob
                blob.root_.append(hP)  # segment is buffered into root_

                if len(fork_) > 1:  # merge blobs of all forks
                    if fork_[0].roots == 1:  # if roots == 1
                        self.formBlob(fork_[0], 1)  # merge seg of 1st fork into its blob

                    for fork in fork_[1:len(fork_)]:  # merge blobs of other forks into blob of 1st fork
                        if fork.roots == 1:
                            self.formBlob(fork, 1)

                        if not fork.blob is blob:
                            blobs = fork.blob
                            blob.extendCoords(blobs.getCoords()).accumParams(blobs.getParams())
                            blob.xD += blobs.xD
                            blob.Ly += blobs.Ly
                            blob.incomplete_segments += blobs.incomplete_segments
                            for seg in blobs.root_:
                                if not seg is fork:
                                    seg.blob = blob         # blobs in other forks are references to blob in the first fork
                                    blob.root_.append(seg)  # buffer of merged root segments
                            fork.blob = blob
                            blob.root_.append(fork)
                        blob.incomplete_segments -= 1

            blob.extendCoords(_P.getCoords())
        return hP
    # ------------------------------------------------- formSegment() end -----------------------------------------------
    def formBlob(self, term_seg, y_carry=0):
        " Terminated segment is merged into continued or initialized blob (all connected segments) "
        term_seg.max_y = self.y - rng - y_carry  # y_carry: min elevation of term_seg over current hP
        blob = term_seg.blob
        blob.accumParams(term_seg.getParams())
        blob.xD += term_seg.xD          # ave_x angle, to evaluate blob for re-orientation
        blob.Ly += len(term_seg.Py_)    # Ly = number of slices in segment
        blob.incomplete_segments += term_seg.roots - 1    # reference to term_seg is already in blob[9]
        if blob.incomplete_segments == 0:  # blob is terminated and packed in frame
            blob.max_y = term_seg.max_y
            # frame P are to compute averages, redundant for same-scope alt_frames
            self.accumBlob(blob)
    # ------------------------------------------------- formBlob() end --------------------------------------------------
# ***************************************************** FRAME CLASS END *************************************************
# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# ***************************************************** MODULE FUNCTIONS ************************************************
# Functions:
# -initP()
# ***********************************************************************************************************************
def initP(x, sign=-1):
    " creates a new P, dert_ is contained by the whole frame instead of in P "
    P = pattern(sign)
    P.min_x = x
    return P
def initSegment(P, fork_, y, ave_x):
    " creates a new segment "
    # [P.s, [min_x, max_x, y - rng - 1, -1, 0, 0], [0, 0, 0, 0, 0, 0], [], 1]  # s, coords, params, root_, incomplete_segments
    segment = pattern(P.sign)
    segment.min_x = P.min_x
    segment.max_x = P.max_x
    segment.min_y = y
    segment.ave_x = ave_x
    segment.xD = 0
    segment.fork_ = fork_
    segment.roots = 0
    segment.Py_ = [(P, 0)]
    segment.accumParams(P.getParams())
    return segment
def initBlob(segment):
    " creates a new blob "
    blob = pattern(segment.sign)
    blob.min_x = segment.min_x
    blob.max_x = segment.max_x
    blob.min_y = segment.min_y
    blob.xD = 0
    blob.Ly = 0
    blob.incomplete_segments = 1    # blob is initialized with a new segment
    blob.accumParams(segment.getParams())
    blob.root_ = [segment]
    return blob
# ***************************************************** MODULE FUNCTIONS END ********************************************