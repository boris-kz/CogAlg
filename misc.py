import cv2
import numpy as np
import filters
# ***************************************************** MISCELLANEOUS FUNCTIONS *****************************************
# Functions:
# -draw_blobs()
# ***********************************************************************************************************************
def draw_blobs(path, blob_, size, out_ablob=0, debug=0):
    " Rebuilt data of blobs into an image, tuple/list version "
    Y, X = size
    frame_img = np.array([[[127] * 4] * X] * Y)

    for blob_idx, blob in enumerate(blob_):  # Iterate through blobs
        if debug: blob_img = np.array([[[127] * 4] * X] * Y)
        if not out_ablob:
            for seg_idx, seg in enumerate(blob[3]): # Iterate through segments
                if debug: seg_img = np.array([[[127] * 4] * X] * Y)
                y = seg[1][2]   # min_y
                for (P, xd) in seg[3]:
                    x = P[1][0]
                    for i in range(P[2][0]):
                        frame_img[y, x, :3] = [255, 255, 255] if P[0] else [0, 0, 0]
                        if debug:
                            seg_img[y, x, :3] = [255, 255, 255] if P[0] else [0, 0, 0]
                            blob_img[y, x, :3] = [255, 255, 255] if P[0] else [0, 0, 0]
                        x += 1
                    y += 1
                if debug:
                    min_x, max_x, min_y, max_y = seg[1][:4]
                    cv2.rectangle(seg_img, (min_x - 1, min_y - 1), (max_x + 1, max_y + 1), (0, 255, 255), 1)
                    cv2.imwrite(path + '/blob%dseg%d.bmp' % (blob_idx, seg_idx), seg_img)
        else:
            if type(blob[4]) == list:
                ablob_ = blob[4]
                for ablob_idx, ablob in enumerate(ablob_):
                    if debug: ablob_img = np.array([[[127] * 4] * X] * Y)
                    for aseg in ablob[3]:
                        y = aseg[1][2]  # min_y
                        for (aP, xd) in aseg[3]:
                            x = aP[1][0]
                            for i in range(aP[2][0]):
                                frame_img[y, x, :3] = [255, 255, 255] if aP[0] else [0, 0, 0]
                                if debug:
                                    blob_img[y, x, :3] = [255, 255, 255] if aP[0] else [0, 0, 0]
                                    ablob_img[y, x, :3] = [255, 255, 255] if aP[0] else [0, 0, 0]
                                x += 1
                            y += 1
                    if debug:
                        min_x, max_x, min_y, max_y = ablob[1][:4]
                        cv2.rectangle(ablob_img, (min_x - 1, min_y - 1), (max_x + 1, max_y + 1), (0, 255, 255), 1)
                        cv2.imwrite(path + '/blob%dablob%d.bmp' % (blob_idx, ablob_idx), ablob_img)
                if debug:
                    min_x, max_x, min_y, max_y = blob[1][:4]
                    cv2.rectangle(blob_img, (min_x - 1, min_y - 1), (max_x + 1, max_y + 1), (0, 255, 255), 1)
                    cv2.imwrite(path + '/blob%d.bmp' % (blob_idx), blob_img)
    cv2.imwrite(path + '/frame.bmp',frame_img)
    # ---------- out_blobs() end ----------------------------------------------------------------------------------------

def get_filters(obj):
    " imports all variables in filters.py "
    str_ = [item for item in dir(filters) if not item.startswith("__")]
    for str in str_:
        var = getattr(filters, str)
        obj[str] = var
# ***************************************************** MISCELLANEOUS FUNCTIONS END *************************************