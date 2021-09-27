"""
Run as a module to display visualizations of layers.
Save, load mechanics for layers of Ps. Much faster than pickling.
"""

import cv2 as cv
import numpy as np
from struct import pack, unpack
from itertools import zip_longest
from line_patterns import CP
from copy import deepcopy
from collections import deque

def transform_to_global_coords(P__):
    """This change subPs' x0 and L, don't use it in main alg."""
    for P_ in P__:
        q = deque([(0, 1, P_)])        # (x0, rng, P_)
        while q:  # 'recursion'
            x0, rng, P_ = q.popleft()
            L_ratio = 2**(rng-1)
            for P in P_:
                P.x0 = x0
                P.L = P.L*L_ratio
                if P.sublayers:
                    _, _, subrng, subP_, *_ = P.sublayers[0][0]
                    q.append((x0, subrng, subP_))
                x0 += P.L


def Ps_to_layers(P__):
    """
    Return a nested list of layers, which is a nested list of rows,
    which in turn is a list of subsets.
    """

    rows_of_layers = []
    for P_ in P__:

        comb_layers = []
        for P in P_:
            comb_layers = [comb_layer + layer
                           for comb_layer, layer in
                           zip_longest(comb_layers, P.sublayers, fillvalue=[])]

        # add first layer on top
        comb_layers = [[(False, 1, 1, P_, [], [])]] + comb_layers

        rows_of_layers.append(comb_layers)

    # flip/rearrange into layers of rows
    layers_of_rows = [*zip_longest(*rows_of_layers, fillvalue=[])]

    return layers_of_rows


def save_Ps(filename, P__):
    """Pack P layers into binary and save to disk."""
    with open(filename, 'wb') as file:
        X = P__[0][-1].x0 + P__[0][-1].L + 1
        Y = len(P__)
        transform_to_global_coords(P__)
        layers = Ps_to_layers(P__)
        file.write(pack('3L', X, Y, len(layers)))
        for rows in layers:
            file.write(pack('L', len(rows)))
            for subsets in rows:
                file.write(pack('L', len(subsets)))
                for fPd, rdn, rng, P_, *_ in subsets:
                    file.write(pack('?3L', fPd, rdn, rng, len(P_)))
                    for P in P_:
                        file.write(pack('L3dL', P.L, float(P.I), float(P.D), float(P.M), P.x0))


def read_Ps(filename):
    """Read layers from binary file."""
    with open(filename, 'rb') as file:
        X, Y, nlayers = unpack('3L', file.read(12))
        layers = []
        for l in range(nlayers):
            nrows, = unpack('L', file.read(4))
            rows = []
            for r in range(nrows):
                nsubsets, = unpack('L', file.read(4))
                subsets = []
                for s in range(nsubsets):
                    fPd, rdn, rng, nPs = unpack('?3L', file.read(16))
                    P_ = []
                    for n in range(nPs):
                        L, I, D, M, x0 = unpack('L3dL', file.read(36))
                        P_.append(CP(L=L, I=I, D=D, M=M, x0=x0))
                    subsets.append((fPd, rdn, rng, P_, [], []))
                rows.append(subsets)
            layers.append(rows)
    return layers, (Y, X)


def show_layer(layer, shape, resolution=(1024, 512), wname="layer"):
    """Show a single layer in an image"""
    img = np.full(shape, 128, 'uint8')

    scale = min(resolution[1]/shape[0], resolution[0]/shape[1])
    resolution = (int(scale*shape[1]), int(scale*shape[0]))

    for y, subsets in enumerate(layer):
        for fPd, rdn, rng, P_, *_ in subsets:
            for P in P_:
                sign = P.D > 0 if fPd else P.M > 0
                img[y, P.x0 : P.x0+P.L] = sign * 255

    cv.imshow(wname, cv.resize(img, resolution, interpolation=cv.INTER_NEAREST))


if __name__ == "__main__":
    layers, shape = read_Ps("frame_of_patterns.bin")
    ilayer = 0          # begin with layer0
    while True:
        show_layer(layers[ilayer], shape)
        k = cv.waitKey(1)

        if k == 27:             # ESC key
            break
        elif k == ord('w'):      # up
            ilayer = max(ilayer-1, 0)
        elif k == ord('s'):      # down
            ilayer = min(ilayer+1, len(layers)-1)

    cv.destroyAllWindows()