import cv2
import numpy as np

def draw_blob(path, frame):
    " Rebuilt data of blobs into an image "

    height, width = frame[-1].shape[:2]
    frame_img = np.array([[127] * width] * height)

    for blob in frame[1]:
        sign = blob.sign
        for seg in blob.e_:
            for P in seg[2]:
                for dert in P[2]:
                    y, x = dert[:2]
                    frame_img[y, x] = 255 if sign else 0

    cv2.imwrite(path + '.bmp', frame_img)
    # ---------- draw_blob() end ----------------------------------------------------------------------------------------
