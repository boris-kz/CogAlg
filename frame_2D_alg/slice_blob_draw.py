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

import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg') # disable visible figure during the processing to speed up the process


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
    if blob_fflip: plt.title('X sstacks, \nY blob')
    else: plt.title('X sstacks, \nX blob')

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
