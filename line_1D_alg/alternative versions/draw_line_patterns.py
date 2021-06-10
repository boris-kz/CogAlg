"""
Visualize output of line_patterns
"""

import cv2
import argparse
from time import time
from utils import *
from itertools import zip_longest
from line_PPs_draft import comp_P_, form_PPm
from line_patterns_class import cross_comp

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
    for y, P_ in enumerate(frame_of_patterns_): # loop each y line
        x = 0 # x location start with 0 in each line

        for P_num, P in enumerate(P_[0]): # loop each pattern

            colour_S = 0 # negative sign, colour = black (0)
            if P.sign:
                colour_S = 255 # if positive sign, colour = white (255)
            colour_I = P.I # I
            colour_D = P.D # D
            colour_M = P.M # M

            for dert_num, dert in enumerate(P.dert_): # loop each dert
                p,d,m = dert.p, dert.d, dert.m
                if d is None: # set None to 0
                    d = 0

                img_p[y,x] = p # dert's p
                img_d[y,x] = d # dert's d
                img_m[y,x] = m # dert's m
                img_mS[y,x] = colour_S # 1st layer mP's sign
                img_mI[y,x] = colour_I # 1st layer mP's I
                img_mD[y,x] = colour_D # 1st layer mP's D
                img_mM[y,x] = colour_M # 1st layer mP's M

                x+=1

    # normalize value (scale from min to max) on negative or big numbers
    img_d_norm = (img_d - img_d.min()) /(img_d.max()-img_d.min() )*255
    img_m_norm = (img_m - img_m.min()) /(img_m.max()-img_m.min() )*255
    img_mI_norm = (img_mI - img_mI.min()) /(img_mI.max()-img_mI.min() )*255
    img_mD_norm = (img_mD - img_mD.min()) /(img_mD.max()-img_mD.min() )*255
    img_mM_norm = (img_mM - img_mM.min()) /(img_mM.max()-img_mM.min() )*255

    # save images to disk
    cv2.imwrite(arguments['output_path'] + '1_p.jpg',  img_p.astype('uint8'))
    cv2.imwrite(arguments['output_path'] + '2_d.jpg',  img_d.astype('uint8'))
    cv2.imwrite(arguments['output_path'] + '4_m.jpg',  img_m.astype('uint8'))
    cv2.imwrite(arguments['output_path'] + '6_mPS.jpg',  img_mS.astype('uint8'))
    cv2.imwrite(arguments['output_path'] + '7_mPI.jpg',  img_mI.astype('uint8'))
    cv2.imwrite(arguments['output_path'] + '9_mPD.jpg',  img_mD.astype('uint8'))
    cv2.imwrite(arguments['output_path'] + '11_mPM.jpg',  img_mM.astype('uint8'))

    # save normalized images to disk
    cv2.imwrite(arguments['output_path'] + '3_d_norm.jpg',  img_d_norm.astype('uint8'))
    cv2.imwrite(arguments['output_path'] + '5_m_norm.jpg',  img_m_norm.astype('uint8'))
    cv2.imwrite(arguments['output_path'] + '8_mPI_norm.jpg',  img_mI_norm.astype('uint8'))
    cv2.imwrite(arguments['output_path'] + '10_mPD_norm.jpg',  img_mD_norm.astype('uint8'))
    cv2.imwrite(arguments['output_path'] + '12_mPM_norm.jpg',  img_mM_norm.astype('uint8'))


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
    for y, dert_P_ in enumerate(frame_of_dert_P_): # loop each y line

        x = 0
        for dert_P_num, dert_P in enumerate(dert_P_): # loop each dert_P

            # get dert_P params
            smP, MP, Neg_M, Neg_L, ML, DL, MI, DI, MD, DD, MM, DM = dert_P.smP, dert_P.MP, dert_P.Neg_M, dert_P.Neg_L, dert_P.ML, dert_P.DL, dert_P.MI, dert_P.DI, dert_P.MD, dert_P.DD, dert_P.MM, dert_P.DM


            # accumulate parameters for visualization
            # there are overlapping Ps, so Ps with more overlapping should get more values
            for i in range(Neg_L):

                img_smP[y,x+i] += smP
                img_MP[y,x+i] += MP
                img_Neg_M [y,x+i] += Neg_M
                img_ML[y,x+i] += ML
                img_DL[y,x+i] += DL
                img_MI[y,x+i] += MI
                img_DI[y,x+i] += DI
                img_MD[y,x+i] += MD
                img_DD[y,x+i] += DD
                img_MM[y,x+i] += MM
                img_DM[y,x+i] += DM

            # loop each Ps' dert to get next x starting location
            for dert_num, dert in enumerate(dert_P.P.dert_):
                x+=1



    # normalize value (scale from min to max) on negative or big numbers
    img_MP_norm = (img_MP - img_MP.min()) /(img_MP.max()-img_MP.min() )*255
    img_Neg_M_norm = (img_Neg_M - img_Neg_M.min()) /(img_Neg_M.max()-img_Neg_M.min() )*255
    img_ML_norm = (img_ML - img_ML.min()) /(img_ML.max()-img_ML.min() )*255
    img_DL_norm = (img_DL - img_DL.min()) /(img_DL.max()-img_DL.min() )*255
    img_MI_norm = (img_MI - img_MI.min()) /(img_MI.max()-img_MI.min() )*255
    img_DI_norm = (img_DI - img_DI.min()) /(img_DI.max()-img_DI.min() )*255
    img_MD_norm = (img_MD - img_MD.min()) /(img_MD.max()-img_MD.min() )*255
    img_DD_norm = (img_DD - img_DD.min()) /(img_DD.max()-img_DD.min() )*255
    img_MM_norm = (img_MM - img_MM.min()) /(img_MM.max()-img_MM.min() )*255
    img_DM_norm = (img_DM - img_DM.min()) /(img_DM.max()-img_DM.min() )*255

    # save images to disk
    cv2.imwrite(arguments['output_path'] + '13_smP.jpg',  img_smP.astype('uint8')*255)
    cv2.imwrite(arguments['output_path'] + '14_MP.jpg',  img_MP.astype('uint8'))
    cv2.imwrite(arguments['output_path'] + '16_Neg_M.jpg',  img_Neg_M.astype('uint8'))
    cv2.imwrite(arguments['output_path'] + '18_ML.jpg',  img_ML.astype('uint8'))
    cv2.imwrite(arguments['output_path'] + '20_DL.jpg',  img_DL.astype('uint8'))
    cv2.imwrite(arguments['output_path'] + '22_MI.jpg',  img_MI.astype('uint8'))
    cv2.imwrite(arguments['output_path'] + '24_DI.jpg',  img_DI.astype('uint8'))
    cv2.imwrite(arguments['output_path'] + '26_MD.jpg',  img_MD.astype('uint8'))
    cv2.imwrite(arguments['output_path'] + '28_DD.jpg',  img_DD.astype('uint8'))
    cv2.imwrite(arguments['output_path'] + '30_MM.jpg',  img_MM.astype('uint8'))
    cv2.imwrite(arguments['output_path'] + '32_DM.jpg',  img_DM.astype('uint8'))

    # save normalized images to disk
    cv2.imwrite(arguments['output_path'] + '15_MP_norm.jpg',  img_MP_norm.astype('uint8'))
    cv2.imwrite(arguments['output_path'] + '17_Neg_M_norm.jpg',  img_Neg_M_norm.astype('uint8'))
    cv2.imwrite(arguments['output_path'] + '19_ML_norm.jpg',  img_ML_norm.astype('uint8'))
    cv2.imwrite(arguments['output_path'] + '21_DL_norm.jpg',  img_DL_norm.astype('uint8'))
    cv2.imwrite(arguments['output_path'] + '23_MI_norm.jpg',  img_MI_norm.astype('uint8'))
    cv2.imwrite(arguments['output_path'] + '25_DI_norm.jpg',  img_DI_norm.astype('uint8'))
    cv2.imwrite(arguments['output_path'] + '27_MD_norm.jpg',  img_MD_norm.astype('uint8'))
    cv2.imwrite(arguments['output_path'] + '29_DD_norm.jpg',  img_DD_norm.astype('uint8'))
    cv2.imwrite(arguments['output_path'] + '31_MM_norm.jpg',  img_MM_norm.astype('uint8'))
    cv2.imwrite(arguments['output_path'] + '33_DM_norm.jpg',  img_DM_norm.astype('uint8'))

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
            dert_P_ = comp_P_(P_[0])
            if len(dert_P_)>1:
                PPm_ = form_PPm(dert_P_)
            else:
                PPm_ = []

            frame_of_dert_P_.append(dert_P_)
            frame_of_PPm_.append(PPm_)

    # draw frame of patterns
    draw_frame_patterns(image, arguments, frame_of_patterns_)

    # draw frame of dert P
    draw_frame_dert_P(image, arguments, frame_of_dert_P_)


    end_time = time() - start_time
    print(end_time)