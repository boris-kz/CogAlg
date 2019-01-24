from time import time
import frame_blobs
from angle_blobs import comp_angle
from misc import draw_blobs
'''
    intra_blob() is an extension to frame_blobs, it performs evaluation for comp_P and recursive frame_blobs within each blob.
    Currently it's mostly a draft, combined with frame_blobs it will form a 2D version of first-level algorithm
    inter_blob() will be second-level 2D algorithm, and a prototype for meta-level algorithm

    colors will be defined as color / sum-of-colors, color Ps are defined within sum_Ps: reflection object?
    relative colors may match across reflecting objects, forming color | lighting objects?     
    comp between color patterns within an object: segmentation?

    inter_olp_blob: scan alt_typ_ ) alt_color, rolp * mL > ave * max_L?
    intra_blob rdn is eliminated by merging blobs, reduced by full inclusion: mediated access?

    dCx = max_x - min_x + 1;  dCy = max_y - min_y + 1
    rC = dCx / dCy  # width / height, vs shift / height: abs(xD) / Ly for oriented blobs only?
    rD = max(abs_Dx, abs_Dy) / min(abs_Dx, abs_Dy)  # lateral variation / vertical variation, for flip and comp_P eval
'''
def eval_blob(blob, dert__):
    s, [min_x, max_x, min_y, max_y, xD, abs_xD, Ly], [L, I, G, Dx, Dy], root_, _ = blob
    Ave = ave * L  # whole-blob reprocessing filter, fixed: no if L?
    rdn = 1  # redundant representation counter
    val_deriv, val_range = 0, 0

    if s:  # positive blob, primary orientation match eval: noisy or directional gradient
        if G > -255:  # likely edge, ave d_angle = ave g?
            rdn += 1  # or greater?
            comp_angle(blob, dert__)  # angle comp, ablob def; a, da, sda accum in higher-composition reps
            sDa = blob[2][6]

            val_deriv = G * -sDa  # -sDa indicates proximate angle match -> recursive comp(d_): dderived?
        val_range = G  # G without angle is not directional, thus likely d reversal and match among distant pixels
    val_comp_Py_ = L + I + G + Dx + Dy  # max P match, also abs_Dx, abs_Dy: more accurate but not needed for most blobs?

    # Three branches of recursion, begin with three generic function call, each including:
    # - val: evaluation of recursion
    # - func: called function
    # - args: arguments of called function
    # They are sorted descending based on val
    val_ = [val_range, val_deriv, val_comp_Py_]             # three instances of generic evaluation for three branches of recursion
    func_ = [inc_range, inc_deriv, comp_Py_]                # corresponding functions of different branches of recursion
    args_ = [[blob, dert__], [blob, dert__], [val_comp_Py_, 0, blob, xD]]   # corresponding functions' argument
    call_sequence = sorted(zip(val_, func_, args_), key= lambda item: item[0], reverse=True)  # sort based on evaluation
    recursion(call_sequence, Ave, rdn)                 # begin recursion

    return blob
def recursion(call_sequence, Ave, rdn):
    " root function for recursive comps functions, selectively call each of them "
    val, func, arg = call_sequence.pop(0)
    if val > Ave * rdn:
        func(*arg, rdn=rdn)     # make specific recursive function call
        # continue recursion:
        if call_sequence:       # stop if call_sequence is empty
            recursion(call_sequence, Ave, rdn+1)
def recursion1(call_sequence, Ave, rdn):
    ''' root function for recursive comps functions, selectively call each of them.
    In this version, result of the call is also evaluated and inserted into call sequence for deeper recursion '''
    val, func, arg = call_sequence.pop(0)
    if val > Ave * rdn:
        new_val, new_func, new_arg = func(*arg, rdn=rdn)
        # insert new function call into the such a position so that the order (in val) is maintained
        new_call_sequence = []
        while call_sequence and call_sequence[0][0] > new_val:
            new_call_sequence.append(call_sequence.pop(0))
        new_call_sequence.append((new_val, new_func, new_arg))
        while call_sequence:
            new_call_sequence.append(call_sequence.pop(0))
        call_sequence = new_call_sequence
        # continue recursion:
        if call_sequence:
            recursion(call_sequence, Ave, rdn+1)

def inc_range(blob, dert__, rdn):  # frame_blobs recursion if sG
    return 0, inc_range, [blob]
def inc_deriv(blob, dert__, rdn):  # frame_blobs recursion if Dx + Dy: separately, or abs_Dx + abs_Dy: directional, but for both?
    return 0, inc_deriv, [blob]
def comp_Py_(val_comp_Py_, norm, blob, xD, rdn):  # leading to comp_P
    return 0, comp_Py_, [val_comp_Py_, norm, blob, xD]
def intra_blob(frame):  # scan of vertical Py_ -> comp_P -> 2D mPPs and dPPs
    I, G, Dx, Dy, xD, abs_xD, Ly, blob_, dert__ = frame
    new_blob_ = []
    for blob in blob_:
        new_blob_.append(eval_blob(blob, dert__))
    frame = I, G, Dx, Dy, xD, abs_xD, Ly, new_blob_, dert__
    return frame
# ************ MAIN FUNCTIONS END ***************************************************************************************

# ************ PROGRAM BODY *********************************************************************************************
# Pattern filters ----------------------------------------------------------------
# eventually updated by higher-level feedback, initialized here as constants:
from misc import get_filters
get_filters(globals())          # imports all filters at once
# --------------------------------------------------------------------------------
# Main ---------------------------------------------------------------------------
start_time = time()
frame = intra_blob(frame_blobs.frame_of_blobs)
end_time = time() - start_time
print(end_time)
draw_blobs('./debug', frame[7], (frame_blobs.Y, frame_blobs.X), oablob=1, debug=1)