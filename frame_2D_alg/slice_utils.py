import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg')  # disable visible figure during the processing to speed up the process
from comp_slice_ import *

aveG = 50
# flip ave
flip_ave = 0.1
flip_ave_FPP = 5
# dx ave
ave_Dx = 10
ave_PP_Dx = 100


def visualize_ortho():
    '''
    visualize dy & dx and their scaled oDy & oDx
    oDy1 = (Dy / hyp - Dx * hyp) / 2  # estimated along-axis D
    oDx1 = (Dx * hyp + Dy / hyp) / 2  # estimated cross-axis D
    oDy2 = (Dy / hyp) + (Dx * hyp) / 2
    oDx2 = (Dy * hyp) + (Dx / hyp) / 2
    oDy3 = np.hypot( Dy / hyp, Dx * hyp)
    oDx3 = np.hypot( Dy * hyp, Dx / hyp)
    '''
    # Dy, Dx, hyp, oDy1, oDx1, oDy2, oDx2, oDy3, oDx3
    values_  = [ [],[],[],[],[],[],[],[],[]]

    f = open("values.txt","r")
    flines = f.readlines()
    for n, val in enumerate(flines):
        values_[n%9].append(round(float(val[:-1]),1))

    num_data = range(len(values_[0]))

    for n in num_data:
        plt.figure()
        plt.plot([values_[1][n]], [values_[0][n]], marker='o', markersize=5, color="black")
        plt.text(values_[1][n], values_[0][n],'Dy, Dx')

        plt.plot([values_[1][n],values_[4][n]],[values_[0][n],values_[3][n]],'r')
        plt.plot([values_[4][n]], [values_[3][n]], marker='o', markersize=5, color="red")
        plt.text(values_[4][n], values_[3][n],'oDy1, oDx1')

        plt.plot([values_[1][n],values_[6][n]],[values_[0][n],values_[5][n]],'g')
        plt.plot([values_[6][n]], [values_[5][n]], marker='o', markersize=5, color="green")
        plt.text(values_[6][n], values_[5][n],'oDy2, oDx2')

        plt.plot([values_[1][n],values_[8][n]],[values_[0][n],values_[7][n]],'b')
        plt.plot([values_[8][n]], [values_[7][n]], marker='o', markersize=5, color="blue")
        plt.text(values_[8][n], values_[7][n],'oDy3, oDx3')

        plt.xlabel('Dx')
        plt.ylabel('Dy')
        plt.savefig('./images/oDyoDx/points_'+str(n)+'.png')
        plt.close()


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
    # PP
    img_colour_PP = np.zeros((yn - y0, xn - x0, 3)).astype('uint8')
    img_colour_PP_Ps = np.zeros((yn - y0, xn - x0, 3)).astype('uint8')
    img_colour_FPP = np.zeros((yn - y0, xn - x0, 3)).astype('uint8')
    img_colour_FPP_Ps = np.zeros((yn - y0, xn - x0, 3)).astype('uint8')
    # PPd
    img_colour_PPd = np.zeros((yn - y0, xn - x0, 3)).astype('uint8')
    img_colour_PPd_Pds = np.zeros((yn - y0, xn - x0, 3)).astype('uint8')
    img_colour_FPPd = np.zeros((yn - y0, xn - x0, 3)).astype('uint8')
    img_colour_FPPd_Pds = np.zeros((yn - y0, xn - x0, 3)).astype('uint8')

    # colour index
    c_ind_P = 0  # P
    c_ind_PP = 0  # PP
    c_ind_PP_Ps = 0  # PP's Ps
    c_ind_FPP_section = 0  # FPP
    c_ind_FPP_section_Ps = 0  # FPP's Ps
    c_ind_PPd = 0  # PPd
    c_ind_PPd_Pds = 0  # PPd's Pds
    c_ind_FPPd_section = 0  # FPPd
    c_ind_FPPd_section_Pds = 0  # FPPd's Pds

    # draw Ps
    for P in blob.P__:
        for x, _ in enumerate(P.dert_):
            img_colour_P[P.y, P.x0 + x] = colour_list[c_ind_P % 10]
        c_ind_P += 1

    for blob_PP in blob.PP_: # draw PP

        # draw PPs
        for derP in blob_PP.derP__:
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
        if blob_PP.derPP.flip_val > flip_ave_FPP :

            # get box
            x0FPP = min([P.x0 for P in blob_PP.Pf__])
            xnFPP = max([P.x0 + P.L for P in blob_PP.Pf__])
            y0FPP = min([P.y for P in blob_PP.Pf__])
            ynFPP = max([P.y for P in blob_PP.Pf__]) + 1  # +1 because yn is not inclusive, else we will lost last y value

            # init smaller image contains the flipped section only
            img_colour_FPP_section = np.zeros((ynFPP - y0FPP, xnFPP - x0FPP, 3))
            img_colour_FPP_section_Ps = np.zeros((ynFPP - y0FPP, xnFPP - x0FPP, 3))

            # fill colour
            for P in blob_PP.Pf__:
                for x, _ in enumerate(P.dert_):
                    img_colour_FPP_section[P.y-y0FPP, P.x0 + x - x0FPP] = colour_list[c_ind_FPP_section % 10]
                    img_colour_FPP_section_Ps[P.y-y0FPP, P.x0 + x - x0FPP] = colour_list[c_ind_FPP_section_Ps % 10]
                c_ind_FPP_section_Ps += 1
            c_ind_FPP_section += 1

            # flip back
            img_colour_FPP_section = np.rot90(img_colour_FPP_section, k=3)
            img_colour_FPP_section_Ps = np.rot90(img_colour_FPP_section_Ps, k=3)

            # fill back the bigger image
            img_colour_FPP[blob_PP.box[0]:blob_PP.box[1], blob_PP.box[2]:blob_PP.box[3]] = img_colour_FPP_section
            img_colour_FPP_Ps[blob_PP.box[0]:blob_PP.box[1], blob_PP.box[2]:blob_PP.box[3]] = img_colour_FPP_section_Ps


    for blob_PPd in blob.PPd_: # draw PPd

        # draw PPds
        for derPd in blob_PPd.derP__:
            if derPd.flip_val <= 0:
                # _P
                for _x, _dert in enumerate(derPd._P.dert_):
                    img_colour_PPd[derPd._P.y, derPd._P.x0 + _x, :] = colour_list[c_ind_PPd % 10]
                    img_colour_PPd_Pds[derPd._P.y, derPd._P.x0 + _x, :] = colour_list[c_ind_PPd_Pds % 10]
                c_ind_PPd_Pds += 1
                # P
                for x, dert in enumerate(derPd.P.dert_):
                    img_colour_PPd[derPd.P.y, derPd.P.x0 + x, :] = colour_list[c_ind_PPd % 10]
                    img_colour_PPd_Pds[derPd.P.y, derPd.P.x0 + x, :] = colour_list[c_ind_PPd_Pds % 10]
                c_ind_PPd_Pds += 1

        c_ind_PPd += 1  # increase P index

        # draw FPPds
        if blob_PPd.derPP.flip_val > flip_ave_FPP :

            # get box
            x0FPPd = min([P.x0 for P in blob_PPd.Pdf__])
            xnFPPd = max([P.x0 + P.L for P in blob_PPd.Pdf__])
            y0FPPd = min([P.y for P in blob_PPd.Pdf__])
            ynFPPd = max([P.y for P in blob_PPd.Pdf__]) + 1  # +1 because yn is not inclusive, else we will lost last y value

            # init smaller image contains the flipped section only
            img_colour_FPPd_section = np.zeros((ynFPPd - y0FPPd, xnFPPd - x0FPPd, 3))
            img_colour_FPPd_section_Pds = np.zeros((ynFPPd - y0FPPd, xnFPPd - x0FPPd, 3))

            # fill colour
            for P in blob_PPd.Pdf__:
                for x, _ in enumerate(P.dert_):
                    img_colour_FPPd_section[P.y-y0FPPd, P.x0 + x - x0FPPd] = colour_list[c_ind_FPPd_section % 10]
                    img_colour_FPPd_section_Pds[P.y-y0FPPd, P.x0 + x- x0FPPd] = colour_list[c_ind_FPPd_section_Pds % 10]
                c_ind_FPPd_section_Pds += 1
            c_ind_FPPd_section += 1

            # flip back
            img_colour_FPPd_section = np.rot90(img_colour_FPPd_section, k=3)
            img_colour_FPPd_section_Pds = np.rot90(img_colour_FPPd_section_Pds, k=3)

            # we need offset because Pds might be empty in certain rows or columns from the given blob_PPd.box
            xs_offset = y0FPPd - 0 # x start offset
            ys_offset = x0FPPd - 0 # y start offset
            xe_offset = (blob_PPd.box[3]-blob_PPd.box[2]) - ((ynFPPd - y0FPPd)+ xs_offset)# x end negative offset
            ye_offset = (blob_PPd.box[1]-blob_PPd.box[0]) - ((xnFPPd - x0FPPd)+ ys_offset )# y end negative offset

            # fill back the bigger image
            img_colour_FPPd[blob_PPd.box[0]+ys_offset:blob_PPd.box[1]-ye_offset, blob_PPd.box[2]+xs_offset:blob_PPd.box[3]-xe_offset] = img_colour_FPPd_section
            img_colour_FPPd_Pds[blob_PPd.box[0]+ys_offset:blob_PPd.box[1]-ye_offset, blob_PPd.box[2]+xs_offset:blob_PPd.box[3]-xe_offset] = img_colour_FPPd_section_Pds


    ## combine images with Ps, PPs and FPPs into 1 single image
    img_combined = np.concatenate((img_colour_P, img_separator), axis=1)
    # PP and their Ps
    img_combined = np.concatenate((img_combined, img_colour_PP), axis=1)
    img_combined = np.concatenate((img_combined, img_separator), axis=1)
    img_combined = np.concatenate((img_combined, img_colour_PP_Ps), axis=1)
    img_combined = np.concatenate((img_combined, img_separator), axis=1)
    # FPP and their Ps
    img_combined = np.concatenate((img_combined, img_colour_FPP), axis=1)
    img_combined = np.concatenate((img_combined, img_separator), axis=1)
    img_combined = np.concatenate((img_combined, img_colour_FPP_Ps), axis=1)
    img_combined = np.concatenate((img_combined, img_separator), axis=1)
    # PPd and their Ps
    img_combined = np.concatenate((img_combined, img_colour_PPd), axis=1)
    img_combined = np.concatenate((img_combined, img_separator), axis=1)
    img_combined = np.concatenate((img_combined, img_colour_PPd_Pds), axis=1)
    img_combined = np.concatenate((img_combined, img_separator), axis=1)
    # FPPd and their Pds
    img_combined = np.concatenate((img_combined, img_colour_FPPd), axis=1)
    img_combined = np.concatenate((img_combined, img_separator), axis=1)
    img_combined = np.concatenate((img_combined, img_colour_FPPd_Pds), axis=1)
    img_combined = np.concatenate((img_combined, img_separator), axis=1)

    # save image to disk
    cv2.imwrite(img_dir_path + 'img_b' + str(blob.id) + '.bmp', img_combined)


def form_PP_dx_(P__):
    '''
    Obsolete
    Cross-comp of dx (incremental derivation) within Pd s of PP_dx, defined by > ave_Dx
    '''

    # the params below should be preserved for comp_dx but not reinitialized on each new P with 0 downconnect count
    P_dx_ = []  # list of selected Ps
    PP_dx_ = [] # list of PPs

    for P in P__:
        if P.downconnect_cnt == 0: # start from root P

            PPDx = 0    # comp_dx criterion per PP
            PPDx_ = []  # list of criteria

            if P.Dx > ave_Dx:
                PPDx += P.Dx
                P_dx_.append(P)

            if P.upconnect_:  # recursively process upconnects
                comp_PP_dx(P.upconnect_, PPDx, PPDx_, P_dx_, PP_dx_)

            # after scanning all upconnects or not having upconnects
            if PPDx != 0:  # terminate PP_Dx and Dx if Dx != 0, else nothing to terminate
                PPDx_.append(PPDx)
                PP_dx_.append(P_dx_)

    # comp_dx
    for i, (PPDx, P_dx_) in enumerate(zip(PPDx_, PP_dx_)):
        if PPDx > ave_PP_Dx:
            comp_dx_(P_dx_)  # no need to return?


def comp_PP_dx(P_, iPPDx, iPPDx_, P_dx_, PP_dx_):
    '''
    Obsolete
    '''
    PPDx = iPPDx
    PPDx_ = iPPDx_

    for P in P_:

        if P.Dx > ave_Dx:  # accumulate dx and Ps
            PPDx += P.Dx
            P_dx_.append(P)

        elif PPDx != 0  :  # reset Pdx and Dx if PPDx is not zero, termination is done from the root after calling all upconnects
            PPDx = 0
            P_dx_ = []

        if P.upconnect_:  # recursively process upconnects
            comp_PP_dx(P.upconnect_, PPDx, PPDx_, P_dx_, PP_dx_)

        elif PPDx != 0 and id(PPDx_) != id(iPPDx_):  # terminate newly created PDx and Dx
            PP_dx_.append(P_dx_)
            PPDx_.append(PPDx)


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

    return dxP_, dxP_Ddx, dxP_Mdx  # no need to return? # since Ddx and Mdx are packed into P, how about dxP_Ddx, dxP_Mdx ? Where should we pack this?


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