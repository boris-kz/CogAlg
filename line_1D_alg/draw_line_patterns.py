"""
Visualize output of line_patterns
"""

import cv2
import argparse
from time import time
from utils import *
from itertools import zip_longest
from line_PPs_draft import comp_P, form_PPm
from line_patterns import cross_comp


def draw_frame_patterns(image, arguments, frame_of_patterns_):
    # initialize image
    img_ini = np.zeros_like(image).astype('float')
    img_p = img_ini.copy()
    img_d = img_ini.copy()
    img_m = img_ini.copy()

    img_mS = img_ini.copy()
    img_mI = img_ini.copy()
    img_mD = img_ini.copy()
    img_mM = img_ini.copy()

    # drawing section
    for y, P_ in enumerate(frame_of_patterns_):  # loop each y line
        x = 0  # x location start with 0 in each line

        for P_num, P in enumerate(P_[0]):  # loop each pattern

            colour_S = 0  # negative sign, colour = black (0)
            if P[0]:
                colour_S = 255  # if positive sign, colour = white (255)
            colour_I = P[2]  # I
            colour_D = P[3]  # D
            colour_M = P[4]  # M

            for dert_num, (p, d, m) in enumerate(P[5]):  # loop each dert
                if d is None:  # set None to 0
                    d = 0

                img_p[y, x] = p  # dert's p
                img_d[y, x] = d  # dert's d
                img_m[y, x] = m  # dert's m
                img_mS[y, x] = colour_S  # 1st layer mP's sign
                img_mI[y, x] = colour_I  # 1st layer mP's I
                img_mD[y, x] = colour_D  # 1st layer mP's D
                img_mM[y, x] = colour_M  # 1st layer mP's M

                x += 1

    # normalize value (scale from min to max) on negative or big numbers
    # remove 4 lines below to observe non-normalized images
    img_d = (img_d - img_d.min()) / (img_d.max() - img_d.min()) * 255
    img_m = (img_m - img_m.min()) / (img_m.max() - img_m.min()) * 255
    img_mI = (img_mI - img_mI.min()) / (img_mI.max() - img_mI.min()) * 255
    img_mD = (img_mD - img_mD.min()) / (img_mD.max() - img_mD.min()) * 255
    img_mM = (img_mM - img_mM.min()) / (img_mM.max() - img_mM.min()) * 255

    # save images to disk
    cv2.imwrite(arguments['output_path'] + 'p.jpg', img_p.astype('uint8'))
    cv2.imwrite(arguments['output_path'] + 'd.jpg', img_d.astype('uint8'))
    cv2.imwrite(arguments['output_path'] + 'm.jpg', img_m.astype('uint8'))
    cv2.imwrite(arguments['output_path'] + 'mPS.jpg', img_mS.astype('uint8'))
    cv2.imwrite(arguments['output_path'] + 'mPI.jpg', img_mI.astype('uint8'))
    cv2.imwrite(arguments['output_path'] + 'mPD.jpg', img_mD.astype('uint8'))
    cv2.imwrite(arguments['output_path'] + 'mPM.jpg', img_mM.astype('uint8'))


def draw_frame_dert_P(image, arguments, frame_of_patterns_):
    # initialize image
    img_ini = np.zeros_like(image).astype('float')
    img_smP = img_ini.copy()
    img_MP = img_ini.copy()
    img_Neg_M = img_ini.copy()
    img_ML = img_ini.copy()
    img_DL = img_ini.copy()
    img_MI = img_ini.copy()
    img_DI = img_ini.copy()
    img_MD = img_ini.copy()
    img_DD = img_ini.copy()
    img_MM = img_ini.copy()
    img_DM = img_ini.copy()

    # drawing section
    for y, dert_P_ in enumerate(frame_of_dert_P_):  # loop each y line

        x = 0
        for dert_P_num, dert_P in enumerate(dert_P_):  # loop each dert_P

            smP, MP, Neg_M, Neg_L, ML, DL, MI, DI, MD, DD, MM, DM, _ = dert_P  # get dert_P params

            # accumulate parameters for visualization
            # there are overlapping Ps, so Ps with more overlapping should get more values
            for i in range(Neg_L):
                img_smP[y, x + i] += smP
                img_MP[y, x + i] += MP
                img_Neg_M[y, x + i] += Neg_M
                img_ML[y, x + i] += ML
                img_DL[y, x + i] += DL
                img_MI[y, x + i] += MI
                img_DI[y, x + i] += DI
                img_MD[y, x + i] += MD
                img_DD[y, x + i] += DD
                img_MM[y, x + i] += MM
                img_DM[y, x + i] += DM

            # loop each Ps' dert to get next x starting location
            for dert_num, (p, d, m) in enumerate(dert_P[12][5]):
                x += 1

    cv2.imwrite(arguments['output_path'] + 'smP.jpg', img_smP.astype('uint8') * 255)
    cv2.imwrite(arguments['output_path'] + 'MP.jpg', img_MP.astype('uint8'))
    cv2.imwrite(arguments['output_path'] + 'Neg_M.jpg', img_Neg_M.astype('uint8'))
    cv2.imwrite(arguments['output_path'] + 'ML.jpg', img_ML.astype('uint8'))
    cv2.imwrite(arguments['output_path'] + 'DL.jpg', img_DL.astype('uint8'))
    cv2.imwrite(arguments['output_path'] + 'MI.jpg', img_MI.astype('uint8'))
    cv2.imwrite(arguments['output_path'] + 'DI.jpg', img_DI.astype('uint8'))
    cv2.imwrite(arguments['output_path'] + 'MD.jpg', img_MD.astype('uint8'))
    cv2.imwrite(arguments['output_path'] + 'DD.jpg', img_DD.astype('uint8'))
    cv2.imwrite(arguments['output_path'] + 'MM.jpg', img_MM.astype('uint8'))
    cv2.imwrite(arguments['output_path'] + 'DM.jpg', img_DM.astype('uint8'))


if __name__ == "__main__":
    # Parse argument (image)
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('-i', '--image',
                                 help='path to image file',
                                 default='.//raccoon.jpg')
    argument_parser.add_argument('-p', '--output_path',
                                 help='path to output folder',
                                 default='./images/line_patterns/')
    arguments = vars(argument_parser.parse_args())
    # Read image
    image = cv2.imread(arguments['image'], 0).astype(int)  # load pix-mapped image
    assert image is not None, "No image in the path"
    image = image.astype(int)

    start_time = time()
    fline_PPs = 1
    # Main

    # processing
    frame_of_patterns_ = cross_comp(image)
    if fline_PPs:  # debug line_PPs
        frame_of_dert_P_ = []
        frame_of_PPm_ = []
        for y, P_ in enumerate(frame_of_patterns_):
            dert_P_ = comp_P(P_[0])
            PPm_ = form_PPm(dert_P_)

            frame_of_dert_P_.append(dert_P_)
            frame_of_PPm_.append(PPm_)

            # draw frame of patterns
    draw_frame_patterns(image, arguments, frame_of_patterns_)

    # draw frame of dert P
    draw_frame_dert_P(image, arguments, frame_of_dert_P_)

    end_time = time() - start_time
    print(end_time)