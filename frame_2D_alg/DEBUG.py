import cv2
import numpy as np
import matplotlib.pyplot as plt

def draw_blob(path, frame, typ=0, debug_parts=0, debug_local=0, show=0):
    " Rebuilt data of blobs into an image "

    Y, X = frame.shape[:2]
    frame_img = np.array([[[127] * 4] * X] * Y)

    for blob_idx, blob in enumerate(frame.blob_):  # Iterate through blobs
        if debug_parts: blob_img = np.array([[[127] * 4] * X] * Y)
        # debug original blob - segment ----------------------------------------------------------------
        if typ == 0:
            if debug_local:
                x0, xn, y0, yn = blob.box
                slice = frame_img[y0:yn, x0:xn]
                slice[blob.map == True, :3] = [255, 255, 255] if blob.sign else [0, 0, 0]
                if debug_parts:
                    slice = blob_img[y0:yn, x0:xn]
                    slice[blob.map == True, :3] = [255, 255, 255] if blob.sign else [0, 0, 0]
            else:
                for seg_idx, seg in enumerate(blob.segment_): # Iterate through segments
                    if debug_parts: seg_img = np.array([[[127] * 4] * X] * Y)
                    y = blob.y0() + seg.y0()   # y0
                    for (P, xd) in seg.Py_:
                        x = blob.x0() + P.x0()
                        for i in range(P.L()):
                            frame_img[y, x, :3] = [255, 255, 255] if P.sign else [0, 0, 0]
                            if debug_parts:
                                seg_img[y, x, :3] = [255, 255, 255] if P.sign else [0, 0, 0]
                                blob_img[y, x, :3] = [255, 255, 255] if P.sign else [0, 0, 0]
                            x += 1
                        y += 1
                    if debug_parts:
                        x0, xn, y0, yn = [b + blob.x0() for b in seg.box[:2]] + [b + blob.y0() for b in seg.box[2:]]
                        cv2.rectangle(seg_img, (x0 - 1, y0 - 1), (xn, yn), (0, 255, 255), 1)
                        cv2.imwrite(path + '/blob%dseg%d.bmp' % (blob_idx, seg_idx), seg_img)
        # debug in_blob - segment ----------------------------------------------------------------------
        else:
            in_blob_ = []
            if typ == 1:
                if hasattr(blob, 'angle_in_blob'):
                    in_blob_ = blob.angle_in_blob.blob_
            else:
                if hasattr(blob, 'deriv_in_blob'):
                    in_blob_ = blob.deriv_in_blob.blob_
            if not in_blob_:
                pass
            for in_blob_idx, in_blob in enumerate(in_blob_):
                if debug_parts: in_blob_img = np.array([[[127] * 4] * X] * Y)
                for seg in in_blob.segment_:
                    y = seg.y0() + in_blob.y0() + blob.y0()
                    for (P, xd) in seg.Py_:
                        x = P.x0() + in_blob.x0() + blob.x0()
                        for i in range(P.L()):
                            frame_img[y, x, :3] = [255, 255, 255] if P.sign else [0, 0, 0]
                            if debug_parts:
                                blob_img[y, x, :3] = [255, 255, 255] if P.sign else [0, 0, 0]
                                in_blob_img[y, x, :3] = [255, 255, 255] if P.sign else [0, 0, 0]
                            x += 1
                        y += 1
                if debug_parts:
                    x0, xn, y0, yn = [b + blob.x0() for b in in_blob.box[:2]] + [b + blob.y0() for b in in_blob.box[2:]]
                    cv2.rectangle(in_blob_img, (x0 - 1, y0 - 1), (xn, yn), (0, 255, 255), 1)
                    cv2.imwrite(path + '/blob%din_blob%d.bmp' % (blob_idx, in_blob_idx), in_blob_img)
            del in_blob_
        if debug_parts:
            x0, xn, y0, yn = blob.box
            cv2.rectangle(blob_img, (x0 - 1, y0 - 1), (xn, yn), (0, 255, 255), 1)
            cv2.imwrite(path + '/blob%d.bmp' % (blob_idx), blob_img)
    if show:
        plt.clf()
        plt.imshow(frame_img)
        plt.show()
    else:
        cv2.imwrite(path + '/frame.bmp',frame_img)
    # ---------- DEBUG() end --------------------------------------------------------------------------------------------