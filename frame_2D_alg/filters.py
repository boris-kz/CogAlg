import math as __math__
# filters are eventually updated by higher-level feedback, initialized here as constants

ave = 20            # g value that coincides with average match: gP filter
ave_blob = 20      # additional filter per blob
div_ave = 1023      # filter for div_comp(L) -> rL, summed vars scaling
flip_ave = 10000    # cost of form_P and deeper?
ave_rate = 0.25     # match rate: ave_match_between_ds / ave_match_between_ps, init at 1/4: I / M (~2) * I / D (~2)
angle_coef = 128 / __math__.pi  # coef to convert radian to 256 degrees
A_cost = 1000
a_cost = 15
input_path = './../images/raccoon_eye.jpg'

'''    
ave_blob should be estimated as a multiple of ave (variable cost of refining g and a per dert):
ave_blob = (hypot_g() + angle_blobs()) blob_delay / (hypot_g() + angle_blobs()) dert_delay
ave: variable cost per dert = sum_g mag that coincides with positive adjustment value.
tentative value of adjustment:
((sum_g - hypot_g) + val_deriv) - (hypot_grad_dert_delay + angle_blobs_dert_delay) / sum_g_delay
val_deriv above is value of angle, computation of which is also conditional on ave,
or added eval of refined blobs for angle_blobs? 

def alt_form_P_(y, dert__):  # horizontally cluster and sum consecutive pixels and their derivatives into Ps

    P_ = deque()  # P buffer
    L, I, Dy, Dx, G = 0, 0, 0, 0, 0
    Pdert_ = []
    dert_ = dert__[y]  # row of pixels + derivatives
    _i, _dy, _dx, _g = dert_[0]
    _s = _g > 0

    for x, (i, dy, dx, g) in enumerate(dert_[1:]):
        s = g > 0
        if s != _s:
            P_.append([_s, L, I, Dy, Dx, G, Pdert_])  # P is packed into P_
            L, I, Dy, Dx, G = 0, 0, 0, 0, 0   # new P
            Pdert_ = []
        L += 1
        I += _i  # accumulate P params
        Dy += _dy
        Dx += _dx
        G += _g
        Pdert_.append((y, x-1, i, dy, dx, g))
        _s = s; _i = i; _dy = dy; _dx = dx; _g = g  # convert dert to prior dert

    return P_

op = np.array([[0] * width] * height)
for blob in frame_of_blobs[1]:
    y0, yn, x0, xn = blob.box
    map = blob.map
    slice = op[y0:yn, x0:xn]

    if blob.sign:
        slice[map] = 255
    else:
        slice[map] = 0

cv2.imwrite('./debug/frame.bmp', op)

'''
def get_filters(obj):
    " imports all variables in filters.py "
    str_ = [item for item in globals() if not item.startswith("__")]
    for str in str_:
        obj[str] = globals()[str]