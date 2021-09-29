"""
Run as a module to display visualizations of layers.
"""
import cv2
import cv2 as cv
import numpy as np
import pickle
from struct import pack, unpack
from itertools import zip_longest
from copy import deepcopy
from collections import deque

from line_patterns import CP, Cdert

# Constants
window_name = "layer"

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


def read_Ps(filename):
    """Read layers from binary file."""
    with open(filename, 'rb') as file:
        P__ = pickle.load(file)
        X = P__[0][-1].x0 + P__[0][-1].L + 1
        Y = len(P__)
        transform_to_global_coords(P__)
        layers = Ps_to_layers(P__)
    return layers, (Y, X)


def show_layer(layer, shape, zoom_stack,
               resolution=(1024, 512),
               wname="layer"):
    """Show a single layer in an image"""
    img = np.full(shape, 128, 'uint8')

    scale = min(resolution[1]/shape[0], resolution[0]/shape[1])
    resolution = (int(scale*shape[1]), int(scale*shape[0]))

    for y, subsets in enumerate(layer):
        for fPd, rdn, rng, P_, *_ in subsets:
            for P in P_:
                sign = P.D > 0 if fPd else P.M > 0
                img[y, P.x0 : P.x0+P.L] = sign * 255

    for x, y in zoom_stack:
        h, w = img.shape[0]//3, img.shape[1]//3
        y -= h//2
        x = min(max(x-w//2, 0), img.shape[0]-w)
        y = min(max(y-h//2, 0), img.shape[1]-h)
        img = img[y:y+h, x:x+w]

    cv.imshow(wname, cv.resize(img, resolution, interpolation=cv.INTER_NEAREST))


if __name__ == "__main__":
    # mouse call back variables
    shown_subset_coord = None
    zoom_loc_stack = []
    rerender = True

    def mouse_call_back(event, x, y, flags, param):
        global rerender, shown_subset_coord, zoom_loc_stack
        if event == 0:
            return
        rerender = True
        if event == cv.EVENT_LBUTTONDOWN:
            shown_subset_coord = (x, y)
        elif event == cv.EVENT_LBUTTONUP:
            shown_subset_coord = None
        elif event == cv.EVENT_LBUTTONDBLCLK:   # zoom in
            if len(zoom_loc_stack) < 3:         # max zoom
                zoom_loc_stack.append((x, y))
        elif event == cv.EVENT_RBUTTONDBLCLK:   # zoom out
            if zoom_loc_stack:
                zoom_loc_stack.pop()

    layers, shape = read_Ps("frame_of_patterns_.pkl")
    ilayer = 0          # begin with layer0
    cv.namedWindow(window_name)
    cv.setMouseCallback(window_name, mouse_call_back)

    while True:
        if rerender:
            rerender = False
            try:
                show_layer(layers[ilayer], shape,
                           zoom_loc_stack, wname=window_name)
            except cv.error as e:
                if "error: (-215:Assertion failed) !ssize.empty() in function 'cv::resize'" not in str(e):
                    raise e
                zoom_loc_stack.pop()
                rerender = True

        k = cv.waitKey(1)
        if k == 27:             # ESC key
            break
        elif k == ord('w'):      # up
            ilayer = max(ilayer-1, 0)
            print("Layer", ilayer)
            rerender = True
        elif k == ord('s'):      # down
            ilayer = min(ilayer+1, len(layers)-1)
            print("Layer", ilayer)
            rerender = True

    cv.destroyAllWindows()