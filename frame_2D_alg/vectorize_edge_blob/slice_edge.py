from math import atan2, cos, floor, pi

'''
In natural images, objects look very fuzzy and frequently interrupted, only vaguely suggested by initial blobs and contours.
Potential object is proximate low-gradient (flat) blobs, with rough / thick boundary of adjacent high-gradient (edge) blobs.
These edge blobs can be dimensionality-reduced to their long axis / median line: an effective outline of adjacent flat blob.
-
Median line can be connected points that are most equidistant from other blob points, but we don't need to define it separately.
An edge is meaningful if blob slices orthogonal to median line form some sort of a pattern: match between slices along the line.
These patterns effectively vectorize representation: they represent match and change between slice parameters along the blob.
-
This process is very complex, so it must be selective. Selection should be by combined value of gradient deviation of edge blobs
and inverse gradient deviation of flat blobs. But the latter is implicit here: high-gradient areas are usually quite sparse.
A stable combination of a core flat blob with adjacent edge blobs is a potential object.
'''
octant = 0.3826834323650898  # radians per octant
aveG = 10  # for vectorize
ave_g = 30  # change to Ave from the root intra_blob?
ave_dangle = .2  # vertical difference between angles: -1->1, abs dangle: 0->1, ave_dangle = (min abs(dangle) + max abs(dangle))/2,

def slice_edge_root(frame):

    flat_blob_ = []  # unpacked sub-blobs
    blob_ = frame[-1]  # init frame blob_

    while blob_: flatten_blob_(flat_blob_, blob_)  # get all sub_blobs as a flat list

    return [slice_edge(blob) for blob in flat_blob_]  # form 2D array of Ps: horizontal blob slices in dert__


def flatten_blob_(flat_blob_, blob_):

    root, sign, I, Dy, Dx, G, yx_, dert_, link_, *other_params = blob = blob_.pop(0)
    if sign:  # positive sign
        if other_params:  # blob has rng+ (non-empty other_params), unfold sub-blobs:
            rdn, rng, sub_blob_ = other_params
            blob_ += sub_blob_
    else:  # negative sign, filter
        try: rdn = root[9]  # root is extended blob
        except IndexError: rdn = 1  # root is frame
        if G > aveG * rdn:
            flat_blob_ += [blob]

def slice_edge(edge):   # edge-blob
    root, sign, I, Dy, Dx, G, yx_, dert_, link_ = edge

    P_ = []
    root__ = {}  # map max yx to P, like in frame_blobs
    for yx, axis in select_max(yx_, dert_):  # max = (yx, axis)
        P = CP(edge, yx, axis, root__)
        P_ += [P]

    edge[:] = root, sign, I, Dy, Dx, G, yx_, dert_, link_, P_  # extended with P (no He or node_ yet)

    return edge

def select_max(yx_, dert_):
    max_ = []
    for (y, x), (i, gy, gx, g) in zip(yx_, dert_):
        # sin_angle, cos_angle:
        sa, ca = gy/g, gx/g
        # get neighbor direction
        dy = 1 if sa > octant else -1 if sa < -octant else 0
        dx = 1 if ca > octant else -1 if ca < -octant else 0
        # ?g[y,x] > blob max:
        new_max = True
        for _y, _x in [(y-dy, x-dx), (y+dy, x+dx)]:
            if (_y, _x) not in yx_: continue  # skip if pixel not in edge blob
            _i, _gy, _gx, _g = dert_[yx_.index((_y, _x))]  # get g of neighbor
            if g < _g:
                new_max = False
                break
        if new_max: max_ += [((y, x), (sa, ca))]

    return max_

class CP:
    def __init__(self, edge, yx, axis, root__):  # form_P:

        y, x = yx
        pivot = i, gy, gx, g = interpolate2dert(edge, y, x)  # pivot dert
        ma = ave_dangle  # max value because P direction is the same as dert gradient direction
        m = ave_g - g
        pivot += ma, m   # pack extra ders

        I, G, M, Ma, L, Dy, Dx = i, g, m, ma, 1, gy, gx
        self.axis = ay, ax = axis
        self.yx_, self.dert_, self.link_ = [yx], [pivot], []

        for dy, dx in [(-ay, -ax), (ay, ax)]: # scan in 2 opposite directions to add derts to P
            self.yx_.reverse(); self.dert_.reverse()
            (_y, _x), (_, _gy, _gx, *_) = yx, pivot  # start from pivot
            y, x = _y+dy, _x+dx  # 1st extension
            while True:
                # scan to blob boundary or angle miss:
                try: i, gy, gx, g = interpolate2dert(edge, y, x)
                except TypeError: break  # out of bound (TypeError: cannot unpack None)

                mangle,dangle = comp_angle((_gy,_gx), (gy, gx))
                if mangle < ave_dangle: break  # terminate P if angle miss
                # update P:
                m = ave_g - g
                I += i; Dy += dy; Dx += dx; G += g; Ma += ma; M += m; L += 1
                self.yx_ += [(y, x)]; self.dert_ += [(i, gy, gx, g, ma, m)]
                # for next loop:
                y += dy; x += dx
                _y, _x, _gy, _gx = y, x, gy, gx

        # scan for neighbor Ps, update link_:
        y, x = yx   # get pivot
        for _y, _x in [(y-1,x-1), (y-1,x), (y-1,x+1), (y,x-1), (y,x+1), (y+1,x-1), (y+1,x), (y+1,x+1)]:
            if (_y, _x) in root__:  # neighbor has P
                self.link_ += [root__[_y, _x]]
        root__[y, x] = self    # update root__

        self.yx = self.yx_[L // 2]  # center
        self.latuple = I, G, M, Ma, L, (Dy, Dx)

    def __repr__(self):
        return f"P({', '.join(map(str, self.latuple))})"


def interpolate2dert(edge, y, x):
    root, sign, I, Dy, Dx, G, yx_, dert_, link_ = edge

    if (y, x) in yx_:   # if edge has (y, x) in it
        return dert_[yx_.index((y, x))]

    # get nearby coords:
    y_ = [fy] = [floor(y)]; x_ = [fx] = [floor(x)]
    if y != fy: y_ += [fy+1]    # y is non-integer
    if x != fx: x_ += [fx+1]    # x is non-integer
    n, I, Dy, Dx, G = 0, 0, 0, 0, 0
    for _y in y_:
        for _x in x_:
            if (_y, _x) in yx_:
                _i, _dy, _dx, _g = dert_[yx_.index((_y, _x))]
                I += _i; Dy += _dy; Dx += _dx; G += _g; n += 1

    if n >= 2: return I/n, Dy/n, Dx/n, G/n


def comp_angle(_A, A):  # rn doesn't matter for angles

    _angle, angle = [atan2(Dy, Dx) for Dy, Dx in [_A, A]]

    dangle = _angle - angle  # difference between angles
    if dangle > pi: dangle -= 2*pi  # rotate full-circle clockwise
    elif dangle < -pi: dangle += 2*pi  # rotate full-circle counter-clockwise
    mangle = (cos(dangle)+1)/2  # angle similarity, scale to [0,1]
    dangle /= 2*pi  # scale to the range of mangle, signed: [-.5,.5]

    return [mangle, dangle]


if __name__ == "__main__":
    import sys
    sys.path.append("..")
    from utils import imread
    from frame_blobs import frame_blobs_root
    from intra_blob import intra_blob_root

    image_file = '../images/raccoon_eye.jpeg'
    image = imread(image_file)

    frame = frame_blobs_root(image)
    intra_blob_root(frame)
    edge_ = slice_edge_root(frame)
    # verification:
    for edge in edge_:
        for P in edge[-1]:
            print(P)