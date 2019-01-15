import cv2
import numpy as np
# ***************************************************** MISCELLANEOUS FUNCTIONS *****************************************
# Functions:
# -outBlobs()
# -drawFrame()
# ***********************************************************************************************************************
def outBlobs(path, blob_, size, debug=0):
    " Rebuilt data of blobs into an image, tuple/list version "
    Y, X = size
    blob_image = np.array([[127] * X] * Y)

    for index, blob in enumerate(blob_):  # Iterate through blobs
        if debug: blob_img = np.array([[127] * X] * Y)
        for seg in blob[3]: # Iterate through segments
            y = seg[1][2]   # y0
            for (P, dx) in seg[3]:
                x = P[1][0] # x0
                for i in range(P[2][0]):
                    blob_image[y, x] = 255 if P[0] else 0
                    if debug: blob_img[y, x] = 255 if P[0] else 0
                    x += 1
                y += 1
        if debug:
            min_x, max_x, min_y, max_y = blob[1][:4]
            cv2.rectangle(blob_img, (min_x - 1, min_y - 1), (max_x + 1, max_y + 1), (0, 255, 255), 1)
            cv2.imwrite('./images/blob%d.jpg' % (index), blob_img)

    cv2.imwrite(path,blob_image)
    # ---------- outBlobs() end -----------------------------------------------------------------------------------------
def drawFrame(path, frame):
    output = np.array([[127] * frame.X] * frame.Y).astype('uint8')
    for blob in frame.blob_:
        for segment in blob.root_:
            y = segment.min_y
            for (P, xd) in segment.Py_:
                for x in range(P.min_x, P.max_x):
                    output[y, x] = 255 if P.sign else 0
                y += 1
    cv2.imwrite(path, output)
    # ---------- drawFrame() end ----------------------------------------------------------------------------------------
# ***************************************************** MISCELLANEOUS FUNCTIONS END *************************************