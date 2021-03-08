import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg')  # disable visible figure during the processing to speed up the process
from comp_slice_ import *

aveG = 50
flip_ave = 2000


def draw_PP_(blob):
    colour_list = []  # list of colours:
    colour_list.append([192, 192, 192])  # silver
    colour_list.append([200, 130, 1])  # blue
    colour_list.append([75, 25, 230])  # red
    colour_list.append([25, 255, 255])  # yellow
    colour_list.append([75, 180, 60])  # green
    colour_list.append([212, 190, 250])  # pink
    colour_list.append([240, 250, 70])  # cyan
    colour_list.append([48, 130, 245])  # orange
    colour_list.append([180, 30, 145])  # purple
    colour_list.append([40, 110, 175])  # brown

    img_dir_path = "./images/PPs/"

    # get box
    if blob.fflip:
        x0 = blob.box[0]
        xn = blob.box[1]
        y0 = blob.box[2]
        yn = blob.box[3]
    else:
        x0 = blob.box[2]
        xn = blob.box[3]
        y0 = blob.box[0]
        yn = blob.box[1]

    # init
    img_colour_P = np.zeros((yn - y0, xn - x0, 3)).astype('uint8')
    img_separator = (np.ones((yn - y0, 1, 3)).astype('uint8')) * 255
    img_colour_PP = np.zeros((yn - y0, xn - x0, 3)).astype('uint8')
    img_colour_PP_Ps = np.zeros((yn - y0, xn - x0, 3)).astype('uint8')
    img_colour_FPP = np.zeros((yn - y0, xn - x0, 3)).astype('uint8')
    img_colour_FPP_Ps = np.zeros((yn - y0, xn - x0, 3)).astype('uint8')

    # colour index
    c_ind_P = 0  # P
    c_ind_PP = 0  # PP
    c_ind_PP_Ps = 0  # PP's Ps
    c_ind_FPP_section = 0  # FPP
    c_ind_FPP_section_Ps = 0  # FPP's Ps

    # draw Ps
    for P in blob.P__:
        for x, _ in enumerate(P.dert_):
            img_colour_P[P.y, P.x0 + x] = colour_list[c_ind_P % 10]
        c_ind_P += 1

    for blob_PP in blob.PP_:

        # draw PPs
        for derP in blob_PP.derP_:
            if derP.flip_val <= 0:
                # _P
                for _x, _dert in enumerate(derP._P.dert_):
                    img_colour_PP[derP._P.y, derP._P.x0 + _x, :] = colour_list[c_ind_PP % 10]
                    img_colour_PP_Ps[derP._P.y, derP._P.x0 + _x, :] = colour_list[c_ind_PP_Ps % 10]
                c_ind_PP_Ps += 1
                # P
                for x, dert in enumerate(derP.P.dert_):
                    img_colour_PP[derP.P.y, derP.P.x0 + x, :] = colour_list[c_ind_PP % 10]
                    img_colour_PP_Ps[derP.P.y, derP.P.x0 + x, :] = colour_list[c_ind_PP_Ps % 10]
                c_ind_PP_Ps += 1

        c_ind_PP += 1  # increase P index

        # draw FPPs
        if blob_PP.flip_val > 0:

            # get box
            x0FPP = min([P.x0 for P in blob_PP.P__])
            xnFPP = max([P.x0 + P.L for P in blob_PP.P__])
            y0FPP = min([P.y for P in blob_PP.P__])
            ynFPP = max([P.y for P in blob_PP.P__]) + 1  # +1 because yn is not inclusive, else we will lost last y value

            # init smaller image contains the flipped section only
            img_colour_FPP_section = np.zeros((ynFPP - y0FPP, xnFPP - x0FPP, 3))
            img_colour_FPP_section_Ps = np.zeros((ynFPP - y0FPP, xnFPP - x0FPP, 3))

            # fill colour
            for P in blob_PP.P__:
                for x, _ in enumerate(P.dert_):
                    img_colour_FPP_section[P.y, P.x0 + x] = colour_list[c_ind_FPP_section % 10]
                    img_colour_FPP_section_Ps[P.y, P.x0 + x] = colour_list[c_ind_FPP_section_Ps % 10]
                c_ind_FPP_section_Ps += 1
            c_ind_FPP_section += 1

            # flip back
            img_colour_FPP_section = np.rot90(img_colour_FPP_section, k=3)
            img_colour_FPP_section_Ps = np.rot90(img_colour_FPP_section_Ps, k=3)

            # fill back the bigger image
            img_colour_FPP[blob_PP.box[0]:blob_PP.box[1], blob_PP.box[2]:blob_PP.box[3]] = img_colour_FPP_section
            img_colour_FPP_Ps[blob_PP.box[0]:blob_PP.box[1], blob_PP.box[2]:blob_PP.box[3]] = img_colour_FPP_section_Ps

        # combine images with Ps, PPs and FPPs into 1 single image
        img_combined = np.concatenate((img_colour_P, img_separator), axis=1)
        img_combined = np.concatenate((img_combined, img_colour_PP), axis=1)
        img_combined = np.concatenate((img_combined, img_separator), axis=1)
        img_combined = np.concatenate((img_combined, img_colour_FPP), axis=1)
        img_combined = np.concatenate((img_combined, img_separator), axis=1)
        img_combined = np.concatenate((img_combined, img_colour_PP_Ps), axis=1)
        img_combined = np.concatenate((img_combined, img_separator), axis=1)
        img_combined = np.concatenate((img_combined, img_colour_FPP_Ps), axis=1)

        # save image to disk
        cv2.imwrite(img_dir_path + 'img_b' + str(blob.id) + '.bmp', img_combined)

ave_Dx = 10
ave_PP_Dx = 100


def form_PP_dx_(P__):
    '''
    Cross-comp of dx (incremental derivation) within Pd s of PP_dx, defined by > ave_Dx
    '''
    for P in P__:  # start from root P

        PPDx = 0    # comp_dx criterion per PP
        PPDx_ = []  # list of criteria
        P_dx_ = []  # list of selected Ps
        PP_dx_ = [] # list of PPs

        if P.downconnect_cnt == 0:
            if P.Dx > ave_Dx:
                PPDx += P.Dx
                P_dx_.append(P)

            if P.upconnect_:  # recursively process upconnects
                comp_PP_dx(P.upconnect_, PPDx, PPDx_, P_dx_, PP_dx_)

                PPDx_.append(PPDx)
                PP_dx_.append(P_dx_)  # after scanning all upconnects

            elif PPDx != 0:  # terminate PP_Dx and Dx if Dx != 0, else nothing to terminate
                PPDx_.append(PPDx)
                PP_dx_.append(P_dx_)

    # comp_dx
    for i, (PPDx, P_dx_) in enumerate(zip(PPDx_, PP_dx_)):
        if PPDx > ave_PP_Dx:
            Pdx_, Ddx, Mdx = comp_dx_(P_dx_)

    # we need to splice Pdx_ back into original Pd_, replacing Ps that were included into PP_dx s?


def comp_PP_dx(derP_, iPPDx, iPPDx_, P_dx_, PP_dx_):
    '''
    below is not edited, please check
    '''

    Dx = iDx
    PDx_ = iPDx_

    for derP in derP_:
        P = derP._P

        if P.Dx > ave_Dx:  # accumulate dx and Ps
            Dx += P.Dx
            PDx_.append(P)

        else:  # terminate and reset Pdx and Dx
            PP_Dx_.append(PDx_)
            Dx_.append(Dx)
            Dx = 0
            PDx_ = []

        if P.upconnect_:  # recursively process upconnects
            comp_PP_dx(P.upconnect_, Dx, PDx_, PP_Dx_, Dx_)

        elif Dx != 0 and id(PDx_) != id(iPDx_):  # terminate newly created PDx and Dx in current function call after processed their upconnects
            PP_Dx_.append(PDx_)
            Dx_.append(Dx)


def comp_dx_(P_):  # cross-comp of dx s in P.dert_

    dxP_ = []
    dxP_Ddx = 0
    dxP_Mdx = 0

    for P in P_:
        Ddx = 0
        Mdx = 0

        dxdert_ = []
        _dx = P.dert_[0][2]  # first dx
        for dert in P.dert_[1:]:
            dx = dert[2]
            ddx = dx - _dx
            mdx = min(dx, _dx)
            dxdert_.append((ddx, mdx))  # no dx: already in dert_
            Ddx += ddx  # P-wide cross-sign, P.L is too short to form sub_Ps
            Mdx += mdx
            _dx = dx

        P.dxdert_ = dxdert_
        P.Ddx = Ddx
        P.Mdx = Mdx

        dxP_.append(P)
        dxP_Ddx += Ddx
        dxP_Mdx += Mdx

    return dxP_, dxP_Ddx, dxP_Mdx


"""
usage: frame_blobs_find_adj.py [-h] [-i IMAGE] [-v VERBOSE] [-n INTRA] [-r RENDER]
                      [-z ZOOM]
optional arguments:
  -h, --help            show this help message and exit
  -i IMAGE, --image IMAGE
                        path to image file
  -v VERBOSE, --verbose VERBOSE
                        print details, useful for debugging
  -n INTRA, --intra INTRA
                        run intra_blobs after frame_blobs
  -r RENDER, --render RENDER
                        render the process
  -z ZOOM, --zoom ZOOM  zooming ratio when rendering
"""


def form_sstack_(stack_):
    '''
    form horizontal stacks of stacks, read backwards, sub-access by upconnects?
    '''
    sstack = []
    sstack_ = []
    _f_up_reverse = True  # accumulate sstack with first stack

    for _stack in reversed(stack_):  # access in termination order
        if _stack.downconnect_cnt == 0:  # this stack is not upconnect of lower _stack
            form_sstack_recursive(_stack, sstack, sstack_, _f_up_reverse)
        # else this _stack is accessed via upconnect_

    return sstack_


def form_sstack_recursive(_stack, sstack, sstack_, _f_up_reverse):
    '''
    evaluate upconnect_s of incremental elevation to form sstack recursively, depth-first
    '''
    id_in_layer = -1
    _f_up = len(_stack.upconnect_) > 0
    _f_ex = _f_up ^ _stack.downconnect_cnt > 0  # one of stacks is upconnected, the other is downconnected, both are exclusive

    if not sstack and not _stack.f_checked:  # if sstack is empty, initialize it with _stack
        sstack = CStack(I=_stack.I, Dy=_stack.Dy, Dx=_stack.Dx, G=_stack.G, M=_stack.M,
                        Dyy=_stack.Dyy, Dyx=_stack.Dyx, Dxy=_stack.Dxy, Dxx=_stack.Dxx,
                        Ga=_stack.Ga, Ma=_stack.Ma, A=_stack.A, Ly=_stack.Ly, y0=_stack.y0,
                        stack_=[_stack], sign=_stack.sign)
        id_in_layer = sstack.id

    for stack in _stack.upconnect_:  # upward access only
        if sstack and not stack.f_checked:

            horizontal_bias = ((stack.xn - stack.x0) / stack.Ly) * (abs(stack.Dy) / ((abs(stack.Dx)+1)))
            # horizontal_bias = L_bias (lx / Ly) * G_bias (Gy / Gx, preferential comp over low G)
            # Y*X / A: fill~elongation, flip value?
            f_up = len(stack.upconnect_) > 0
            f_ex = f_up ^ stack.downconnect_cnt > 0
            f_up_reverse = f_up != _f_up and (f_ex and _f_ex)  # unreliable, relative value of connects are not known?

            if horizontal_bias > 1:  # or f_up_reverse:  # stack is horizontal or vertical connectivity is reversed: stack combination is horizontal
                # or horizontal value += reversal value: vertical value cancel - excess: non-rdn value only?
                # accumulate stack into sstack:
                sstack.accumulate(I=stack.I, Dy=stack.Dy, Dx=stack.Dx, G=stack.G, M=stack.M, Dyy=stack.Dyy, Dyx=stack.Dyx, Dxy=stack.Dxy, Dxx=stack.Dxx,
                                  Ga=stack.Ga, Ma=stack.Ma, A=stack.A)
                sstack.Ly = max(sstack.y0 + sstack.Ly, stack.y0 + stack.Ly) - min(sstack.y0, stack.y0)  # Ly = max y - min y: maybe multiple Ps in line
                sstack.y0 = min(sstack.y0, stack.y0)  # y0 is min of stacks' y0
                sstack.stack_.append(stack)

                # recursively form sstack from stack
                form_sstack_recursive(stack, sstack, sstack_, f_up_reverse)
                stack.f_checked = 1

            # change in stack orientation, check upconnect_ in the next loop
            elif not stack.f_checked:  # check stack upconnect_ to form sstack
                form_sstack_recursive(stack, [], sstack_, f_up_reverse)
                stack.f_checked = 1

    # upconnect_ ends, pack sstack in current layer
    if sstack.id == id_in_layer:
        sstack_.append(sstack) # pack sstack only after scan through all their stacks' upconnect


def flip_sstack_(sstack_, dert__, verbose):
    '''
    evaluate for flipping dert__ and re-forming Ps and stacks per sstack
    '''
    out_stack_ = []

    for sstack in sstack_:
        x0_, xn_, y0_ = [],[],[]
        for stack in sstack.stack_:  # find min and max x and y in sstack:
            x0_.append(stack.x0)
            xn_.append(stack.xn)
            y0_.append(stack.y0)
        x0 = min(x0_)
        xn = max(xn_)
        y0 = min(y0_)
        sstack.x0, sstack.xn, sstack.y0 = x0, xn, y0

        horizontal_bias = ((xn - x0) / sstack.Ly) * (abs(sstack.Dy) / ((abs(sstack.Dx)+1)))
        # horizontal_bias = L_bias (lx / Ly) * G_bias (Gy / Gx, higher match if low G)
        # or select per P and merge across connected stacks: not relevant?
        # per contig box? or Gy/Gx, geometry is not precise?

        if horizontal_bias > 1 and (sstack.G * sstack.Ma * horizontal_bias > flip_ave):

            sstack_mask__ = np.ones((sstack.Ly, xn - x0)).astype(bool)
            # unmask sstack:
            for stack in sstack.Py_:
                for y, P in enumerate(stack.Py_):
                    sstack_mask__[y+(stack.y0-y0), P.x0-x0: (P.x0-x0 + P.L)] = False  # unmask P, P.x0 is relative to x0
            # flip:
            sstack_dert__ = tuple([np.rot90(param_dert__[y0:y0+sstack.Ly, x0:xn]) for param_dert__ in dert__])
            sstack_mask__ = np.rot90(sstack_mask__)

            row_stack_ = []
            for y, dert_ in enumerate(zip(*sstack_dert__)):  # same operations as in root sliced_blob(), but on sstack

                P_ = form_P_(list(zip(*dert_)), sstack_mask__[y])  # horizontal clustering
                P_ = scan_P_(P_, row_stack_)  # vertical clustering, adds P upconnects and _P downconnect_cnt
                next_row_stack_ = form_stack_(P_, y)  # initialize and accumulate stacks with Ps

                [sstack.stack_.append(stack) for stack in row_stack_ if not stack in next_row_stack_]  # buffer terminated stacks
                row_stack_ = next_row_stack_

            sstack.stack_ += row_stack_   # dert__ ends, all last-row stacks have no downconnects
            if verbose: check_stacks_presence(sstack.stack_, sstack_mask__, f_plot=0)

            out_stack_.append(sstack)
        else:
            out_stack_ += [sstack.stack_]  # non-flipped sstack is deconstructed, stack.stack_ s are still empty

    return out_stack_


def draw_stacks(stack_):
    # retrieve region size of all stacks
    y0 = min([stack.y0 for stack in stack_])
    yn = max([stack.y0 + stack.Ly for stack in stack_])
    x0 = min([P.x0 for stack in stack_ for P in stack.Py_])
    xn = max([P.x0 + P.L for stack in stack_ for P in stack.Py_])

    img = np.zeros((yn - y0, xn - x0))
    stack_index = 1

    for stack in stack_:
        for y, P in enumerate(stack.Py_):
            for x, dert in enumerate(P.dert_):
                img[y + (stack.y0 - y0), x + (P.x0 - x0)] = stack_index
        stack_index += 1  # for next stack

    colour_list = []  # list of colours:
    colour_list.append([255, 255, 255])  # white
    colour_list.append([200, 130, 0])  # blue
    colour_list.append([75, 25, 230])  # red
    colour_list.append([25, 255, 255])  # yellow
    colour_list.append([75, 180, 60])  # green
    colour_list.append([212, 190, 250])  # pink
    colour_list.append([240, 250, 70])  # cyan
    colour_list.append([48, 130, 245])  # orange
    colour_list.append([180, 30, 145])  # purple
    colour_list.append([40, 110, 175])  # brown
    # initialization
    img_colour = np.zeros((yn - y0, xn - x0, 3)).astype('uint8')
    total_stacks = stack_index

    for i in range(1, total_stacks + 1):
        colour_index = i % 10
        img_colour[np.where(img == i)] = colour_list[colour_index]
    #    cv2.imwrite('./images/stacks/stacks_blob_' + str(sliced_blob.id) + '_colour.bmp', img_colour)

    return img_colour

def check_stacks_presence(stack_, mask__, f_plot=0):
    '''
    visualize stack_ and mask__ to ensure that they cover the same area
    '''
    img_colour = draw_stacks(stack_)  # visualization to check if we miss out any stack from the mask
    img_sum = img_colour[:, :, 0] + img_colour[:, :, 1] + img_colour[:, :, 2]

    # check if all unmasked area is filled in plotted image
    check_ok_y = np.where(img_sum > 0)[0].all() == np.where(mask__ == False)[0].all()
    check_ok_x = np.where(img_sum > 0)[1].all() == np.where(mask__ == False)[1].all()

    # check if their shape is the same
    check_ok_shape_y = img_colour.shape[0] == mask__.shape[0]
    check_ok_shape_x = img_colour.shape[1] == mask__.shape[1]

    if not check_ok_y or not check_ok_x or not check_ok_shape_y or not check_ok_shape_x:
        print("----------------Missing stacks----------------")

    if f_plot:
        from matplotlib import pyplot as plt
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(img_colour)
        plt.subplot(1, 2, 2)
        plt.imshow(mask__ * 255)


def draw_sstack_(blob_fflip, sstack_):

    '''visualize stacks and sstacks and their scanning direction, runnable but not fully optimized '''

    colour_list = []  # list of colours:
    colour_list.append([200, 130, 0])  # blue
    colour_list.append([75, 25, 230])  # red
    colour_list.append([25, 255, 255])  # yellow
    colour_list.append([75, 180, 60])  # green
    colour_list.append([212, 190, 250])  # pink
    colour_list.append([240, 250, 70])  # cyan
    colour_list.append([48, 130, 245])  # orange
    colour_list.append([180, 30, 145])  # purple
    colour_list.append([40, 110, 175])  # brown
    colour_list.append([255, 255, 255])  # white

    x0_, xn_, y0_, yn_ = [], [], [], []
    for sstack in sstack_:
        for stack in sstack.Py_:
            x0_.append(min([Py.x0 for Py in stack.Py_]))
            xn_.append(max([Py.x0 + Py.L for Py in stack.Py_]))
            y0_.append(stack.y0)
            yn_.append(stack.y0 + stack.Ly)
    # sstack box:
    x0 = min(x0_)
    xn = max(xn_)
    y0 = min(y0_)
    yn = max(yn_)

    img_index_stacks = np.zeros((yn - y0, xn - x0))
    img_index_sstacks = np.zeros((yn - y0, xn - x0))
    img_index_sstacks_flipped = np.zeros((yn - y0, xn - x0)) # show flipped stacks of sstack only
    img_index_sstacks_mix = np.zeros((yn - y0, xn - x0)) # shows flipped and non flipped stacks

    stack_index = 1
    sstack_index = 1
    sstack_flipped_index = 1
    sstack_mix_index = 1

    # insert stack index and sstack index to X stacks
    for sstack in sstack_:
        for stack in sstack.Py_:
            for y, P in enumerate(stack.Py_):
                for x, dert in enumerate(P.dert_):
                    img_index_stacks[y + (stack.y0 - y0), x + (P.x0 - x0)] = stack_index
                    img_index_sstacks[y + (stack.y0 - y0), x + (P.x0 - x0)] = sstack_index
            stack_index += 1  # for next stack
        sstack_index += 1  # for next sstack

    # insert stack index and sstack index to flipped X stacks
    for sstack in sstack_:
        if sstack.stack_:  # sstack is flipped
            sx0_, sxn_, sy0_, syn_ = [], [], [], []
            for stack in sstack.Py_:
                sx0_.append(min([Py.x0 for Py in stack.Py_]))
                sxn_.append(max([Py.x0 + Py.L for Py in stack.Py_]))
                sy0_.append(stack.y0)
                syn_.append(stack.y0 + stack.Ly)
            # sstack box:
            # sx0 = min(sx0_)
            sxn = max(sxn_)
            sy0 = min(sy0_)
            # syn = max(syn_)

            for stack in sstack.stack_:
                for y, P in enumerate(stack.Py_):
                    for x, dert in enumerate(P.dert_):
                        img_index_sstacks_flipped[sy0+(x + (P.x0 - x0)),sxn-1-(y + (stack.y0 - y0))] = sstack_flipped_index
                        img_index_sstacks_mix[sy0+(x + (P.x0 - x0)),sxn-1-(y + (stack.y0 - y0))] = sstack_mix_index
                sstack_flipped_index += 1  # for next stack of flipped sstack
                sstack_mix_index += 1  # for next stack of flipped sstack

        else:  # sstack is not flipped
            for stack in sstack.Py_:
                for y, P in enumerate(stack.Py_):
                    for x, dert in enumerate(P.dert_):
                        img_index_sstacks_mix[y + (stack.y0 - y0),x + (P.x0 - x0)] = sstack_mix_index
                sstack_mix_index += 1  # for next stack of sstack

    # initialize colour image
    img_colour_stacks = np.zeros((yn-y0, xn-x0, 3)).astype('uint8')
    img_colour_sstacks = np.zeros((yn-y0, xn-x0, 3)).astype('uint8')
    img_colour_sstacks_flipped = np.zeros((yn-y0, xn-x0, 3)).astype('uint8')
    img_colour_sstacks_mix = np.zeros((yn-y0, xn-x0, 3)).astype('uint8')

    # draw stacks and sstacks
    for i in range(1, stack_index):
        colour_index = i % 10
        img_colour_stacks[np.where(img_index_stacks == i)] = colour_list[colour_index]
    for i in range(1, sstack_index):
        colour_index = i % 10
        img_colour_sstacks[np.where(img_index_sstacks == i)] = colour_list[colour_index]
    for i in range(1, sstack_flipped_index):
        colour_index = i % 10
        img_colour_sstacks_flipped[np.where(img_index_sstacks_flipped == i)] = colour_list[colour_index]
    for i in range(1, sstack_mix_index):
        colour_index = i % 10
        img_colour_sstacks_mix[np.where(img_index_sstacks_mix == i)] = colour_list[colour_index]

    # draw image to figure and save it to disk
    plt.figure(1)

    plt.subplot(1,4,1)
    plt.imshow(img_colour_sstacks)
    if blob_fflip: plt.title('sstacks, \nY blob')
    else: plt.title('sstacks, \nX blob')

    plt.subplot(1,4,2)
    plt.imshow(np.uint8(img_colour_stacks))
    plt.title('X stacks')

    if any([sstack.stack_ for sstack in sstack_]):  # for sstacks with not empty stack_, show flipped stacks

        plt.subplot(1,4,3)
        plt.imshow(img_colour_sstacks_flipped)
        plt.title('Y stacks (flipped)')
        plt.subplot(1,4,4)
        plt.imshow(img_colour_sstacks_mix)
        plt.title('XY stacks')


    plt.savefig('./images/slice_blob/sstack_'+str(id(sstack_))+'.png')
    plt.close()
