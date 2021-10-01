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
zoom_ratio = 0.3

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
               selected_loc=None,
               resolution=(1024, 512),
               wname="layer"):
    """Show a single layer in an image"""
    img = np.full(shape+(3,), 128, 'uint8')
    if selected_loc is not None:
        subset_map = np.zeros(shape+(3,), 'uint8')
    scale = min(resolution[1]/shape[0], resolution[0]/shape[1])
    resolution = (int(scale*shape[1]), int(scale*shape[0]))

    # draw layer pattern map
    subset_id = 1
    tmp_subset_table = [None]
    for y, subsets in enumerate(layer):
        for subset in subsets:
            tmp_subset_table.append(subset)
            fPd, rdn, rng, P_, *_ = subset
            if selected_loc is not None:
                id = (subset_id >> 16, (subset_id >> 8) & 0xFF, subset_id & 0xFF)
                subset_id += 1
            for P in P_:
                sign = P.D > 0 if fPd else P.M > 0
                img[y, P.x0 : P.x0+P.L] = sign * 255
                if selected_loc is not None:
                    subset_map[y, P.x0 : P.x0+P.L] = id

    # zooming handling
    zoomed_img = img[:]
    xy_ratio = 1
    for x, y in zoom_stack:
        x, y = x*xy_ratio, y*xy_ratio
        h, w = zoomed_img.shape[0]*zoom_ratio, zoomed_img.shape[1]*zoom_ratio
        # Compute top-left corner coords
        x = min(max(x, 0), zoomed_img.shape[1]-w)
        y = min(max(y, 0), zoomed_img.shape[0]-h)
        indx = (slice(round(y), round(y+h)), slice(round(x), round(x+w)))
        zoomed_img = zoomed_img[indx]
        xy_ratio *= zoom_ratio
        if selected_loc is not None:
            subset_map = subset_map[indx]

    # if left-clicked
    if selected_loc is not None:
        # extract 3 numbers subset id
        x, y = selected_loc
        resized_subset_map = cv.resize(subset_map, resolution, interpolation=cv.INTER_NEAREST)
        selected_id = resized_subset_map[y, x]
        if (selected_id != (0, 0, 0)).any():     # not blank
            # highlight selected subset (yellow)
            selected = (subset_map == selected_id).all(axis=-1).nonzero()
            selected_b = selected + (0,)
            selected_gr = selected + (slice(1, None),)
            zoomed_img[selected_b] = 0
            zoomed_img[selected_gr] = zoomed_img[selected_gr]//2 + 128

            # display subset parameters
            subset_id = (selected_id[0] << 16) + (selected_id[1] << 8) + selected_id[2]
            fPd, rdn, rng, P_, *_ = tmp_subset_table[subset_id]
            print("Selected fork:", end="")
            print(f" Type = {'Pd' if fPd else 'Pm'}; rdn = {rdn}; rng = {rng}")

    cv.imshow(wname, cv.resize(zoomed_img, resolution, interpolation=cv.INTER_NEAREST))


# draw_PP_ similar process here
def save_Pps(filename, frame_Pp__):
    pass

def read_Pps(filename):
    pass

def ordinal(n):
    if ilayer % 10 == 1: suffix = "st"
    elif ilayer % 10 == 2: suffix = "nd"
    elif ilayer % 10 == 3: suffix = "rd"
    else: suffix = "th"
    return str(n) + suffix

if __name__ == "__main__":
    # mouse call back variables
    selected_loc = None
    zoom_stack = []
    rerender = True

    def mouse_call_back(event, x, y, flags, param):
        global rerender, selected_loc, zoom_stack
        if event == 0:
            return
        rerender = True
        if event == cv.EVENT_LBUTTONUP:         # display pattern
            selected_loc = (x, y)
        elif event == cv.EVENT_LBUTTONDBLCLK:   # zoom in
            if len(zoom_stack) < 3:         # max zoom
                zoom_stack.append((x, y))
        elif event == cv.EVENT_RBUTTONDBLCLK:   # zoom out
            if zoom_stack:
                zoom_stack.pop()

    layers, shape = read_Ps("frame_of_patterns_.pkl")
    ilayer = 0          # begin with layer0
    cv.namedWindow(window_name)
    cv.setMouseCallback(window_name, mouse_call_back)

    while True:
        if rerender:
            rerender = False
            try:
                show_layer(layers[ilayer], shape,
                           zoom_stack, selected_loc,
                           wname=window_name)
            except cv.error as e:
                if "error: (-215:Assertion failed) !ssize.empty() in function 'cv::resize'" not in str(e):
                    raise e
                zoom_stack.pop()
                rerender = True
            selected_loc = None

        k = cv.waitKey(1)
        if k == 27:              # ESC key
            break
        elif k == ord('w'):      # up
            ilayer = max(ilayer-1, 0)
            print(ordinal(ilayer+1), "layer")
            rerender = True
        elif k == ord('s'):      # down
            ilayer = min(ilayer+1, len(layers)-1)
            print(ordinal(ilayer+1), "layer")
            rerender = True

    cv.destroyAllWindows()