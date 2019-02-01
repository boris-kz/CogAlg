import cv2
import numpy as np
import matplotlib.pyplot as plt

def draw_blob(path, frame, debug_ablob=0, debug_parts=0, debug_local=0, show=0):
    " Rebuilt data of blobs into an image "

    Y, X = frame.shape
    frame_img = np.array([[[127] * 4] * X] * Y)

    for blob_idx, blob in enumerate(frame.blob_):  # Iterate through blobs
        if debug_parts: blob_img = np.array([[[127] * 4] * X] * Y)
        if not debug_ablob and debug_local:
            x_start, x_end, y_start, y_end = blob.boundaries
            slice = frame_img[y_start:y_end, x_start:x_end]
            slice[blob.blob_map == True, :3] = [255, 255, 255] if blob.sign else [0, 0, 0]
            if debug_parts:
                slice = blob_img[y_start:y_end, x_start:x_end]
                slice[blob.blob_map == True, :3] = [255, 255, 255] if blob.sign else [0, 0, 0]
        elif not debug_ablob and not debug_local:
            for seg_idx, seg in enumerate(blob.segment_): # Iterate through segments
                if debug_parts: seg_img = np.array([[[127] * 4] * X] * Y)
                y = blob.y_start() + seg.y_start()   # y_start
                for (P, xd) in seg.Py_:
                    x = blob.x_start() + P.x_start()
                    for i in range(P.L()):
                        frame_img[y, x, :3] = [255, 255, 255] if P.sign else [0, 0, 0]
                        if debug_parts:
                            seg_img[y, x, :3] = [255, 255, 255] if P.sign else [0, 0, 0]
                            blob_img[y, x, :3] = [255, 255, 255] if P.sign else [0, 0, 0]
                        x += 1
                    y += 1
                if debug_parts:
                    x_start, x_end, y_start, y_end = [b + blob.x_start() for b in seg.boundaries[:2]] + [b + blob.y_start() for b in seg.boundaries[2:]]
                    cv2.rectangle(seg_img, (x_start - 1, y_start - 1), (x_end, y_end), (0, 255, 255), 1)
                    cv2.imwrite(path + '/blob%dseg%d.bmp' % (blob_idx, seg_idx), seg_img)
        else:
            if hasattr(blob, 'frame_of_ablob'):
                ablob_ = blob.frame_of_ablob.blob_
                for ablob_idx, ablob in enumerate(ablob_):
                    if debug_parts: ablob_img = np.array([[[127] * 4] * X] * Y)
                    for aseg in ablob.segment_:
                        y = aseg.y_start() + ablob.y_start() + blob.y_start()
                        for (aP, xd) in aseg.Py_:
                            x = aP.x_start() + ablob.x_start() + blob.x_start()
                            for i in range(aP.L()):
                                frame_img[y, x, :3] = [255, 255, 255] if aP.sign else [0, 0, 0]
                                if debug_parts:
                                    blob_img[y, x, :3] = [255, 255, 255] if aP.sign else [0, 0, 0]
                                    ablob_img[y, x, :3] = [255, 255, 255] if aP.sign else [0, 0, 0]
                                x += 1
                            y += 1
                    if debug_parts:
                        x_start, x_end, y_start, y_end = [b + blob.x_start() for b in ablob.boundaries[:2]] + [b + blob.y_start() for b in ablob.boundaries[2:]]
                        cv2.rectangle(ablob_img, (x_start - 1, y_start - 1), (x_end, y_end), (0, 255, 255), 1)
                        cv2.imwrite(path + '/blob%dablob%d.bmp' % (blob_idx, ablob_idx), ablob_img)
        if debug_parts:
            x_start, x_end, y_start, y_end = blob.boundaries
            cv2.rectangle(blob_img, (x_start - 1, y_start - 1), (x_end, y_end), (0, 255, 255), 1)
            cv2.imwrite(path + '/blob%d.bmp' % (blob_idx), blob_img)
    if show:
        plt.clf()
        plt.imshow(frame_img)
        plt.show()
    else:
        cv2.imwrite(path + '/frame.bmp',frame_img)
    # ---------- DEBUG() end --------------------------------------------------------------------------------------------