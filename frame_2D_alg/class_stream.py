"""
Provide streaming methods to monitor some of frame_2D_alg operations.
Used in frame_blobs_yx.
"""
import sys
import numpy as np
import cv2 as cv

from utils import (
    blank_image, over_draw, stack_box,
    draw_blob, draw_stack,
    BLACK, WHITE, GREY, DGREY, LGREY,
)

class Streamer:
    """
    Base class to stream visualizations.
    """
    def __init__(self, window_size=(640, 480),
                 winname='streamer', record_path=None):
        self.winname = winname
        self.window_size = window_size
        self.frame = np.empty((window_size[1], window_size[0], 3), 'uint8')
        self.view = self.frame
        self.view_box = (0, window_size[0], 0, window_size[1])
        self.is_zooming = False
        if record_path is not None:
            self._render = self._render_and_record
            self.video_writer = cv.VideoWriter(record_path,
                                               cv.VideoWriter_fourcc(*'XVID'),
                                               20.0, window_size)
        else:
            self._render = self._render_no_record
            self.video_writer = None
        self.render = self._render

        # initialize window
        def mouse_call(event, x, y, flags, param):
            x = min(self.window_size[0]-1, max(x, 0))
            y = min(self.window_size[1]-1, max(y, 0))
            self.x = x
            self.y = y

            if event == cv.EVENT_LBUTTONDOWN and not self.is_zooming:
                self.x1 = x
                self.y1 = y
                self.render = self._render_draw_rectangle
            elif event == cv.EVENT_LBUTTONUP:
                if not self.is_zooming:
                    if x != self.x1 and y != self.y1:
                        self.x2 = x
                        self.y2 = y
                        if x < self.x1:
                            self.x1, self.x2 = x, self.x1
                        if y < self.y1:
                            self.y1, self.y2 = y, self.y1
                        self.render = self._zoomed_render
                        self.is_zooming = True
                else:
                    self.is_zooming = False
                    self.render = self._render

        cv.namedWindow(self.winname)
        cv.setMouseCallback(self.winname, mouse_call)

    def update(self, *args, **kwargs):
        """Call this method each update."""
        self.view = np.copy(self.frame)

    def stop(self):
        cv.destroyAllWindows()
        if self.video_writer is not None:
            self.video_writer.release()

    def writeframe(self, path):
        cv.imwrite(path, self.frame)

    def _render_no_record(self, waitms=1):
        """Render visualization to screen."""
        cv.imshow(winname=self.winname, mat=self.view)
        return cv.waitKey(waitms)

    def _render_and_record(self, waitms=1):
        cv.imshow(winname=self.winname, mat=self.view)
        self.video_writer.write(self.view)
        return cv.waitKey(waitms)

    def _render_draw_rectangle(self, **kwargs):
        cv.rectangle(self.view,
                     (self.x1, self.y1),
                     (self.x, self.y),
                     (0, 0, 255), 2)
        self._render(**kwargs)

    def _zoomed_render(self, **kwargs):
        self.view = cv.resize(self.view[self.y1:self.y2,
                                        self.x1:self.x2],
                              self.window_size,
                              interpolation=cv.INTER_NEAREST)
        self._render(**kwargs)


class BlobStreamer(Streamer):
    """
    Use this class to monitor the actions of image_to_blobs in frame_blobs.
    """
    sign_map = {False: BLACK, True: WHITE}  # sign_map for terminated blobs
    sign_map_unterminated = {False: DGREY, True: LGREY}  # sign_map for unterminated blobs

    def __init__(self, blob_cls, crit__, mask=None,
                 window_size=None,
                 winname='image_to_blobs',
                 record_path=None):
        self.blob_cls = blob_cls
        height, width = crit__.shape
        if window_size is None:
            if height < 480:
                window_size = 640, 480
            else:
                window_size = width, height
        Streamer.__init__(self, window_size=window_size,
                          winname=winname,
                          record_path=record_path)
        # set background with g__
        self.img = blank_image((height, width))
        self.img[:, :, 0] = (255 * (crit__ - crit__.min()) /
                             (crit__.max() - crit__.min()))
        self.img[:, :, 2] = self.img[:, :, 1] = self.img[:, :, 0]
        if mask is not None:
            self.img[mask] = 128, 128, 128

        # initialize other variables
        self.incomplete_blob_ids = set()
        self.first_id = self.blob_cls.instance_cnt

        # for interactive adj highlight after conversion
        self.background = None
        self.id_map = None
        self._id_map = np.empty((height, width), 'uint64')
        self.pointing_blob_id = None

    def update(self, y, P_=()):
        """Call this method each update."""
        # draw Ps in new row
        for P in P_:
            self.img[y, P.x0 : P.x0+P.L] = self.sign_map_unterminated[P.sign]

        # add new blobs' ids, if any
        id_end = self.blob_cls.instance_cnt
        if self.first_id < id_end:
            new_blobs_ids = range(self.first_id, id_end)
            self.incomplete_blob_ids.update(new_blobs_ids)
            self.first_id = id_end

        # iterate through incomplete blobs
        for blob_id in set(self.incomplete_blob_ids):
            blob = self.blob_cls.get_instance(blob_id)
            if blob is None:  # blob has been merged, remove its ref
                self.incomplete_blob_ids.remove(blob_id)
                continue
            elif blob.open_stacks == 0:
                # has no open_stack, has been terminated,
                # re-draw with normal colors, remove ref
                self.incomplete_blob_ids.remove(blob_id)
                blob_img = draw_blob(blob, blob_box=blob.box,
                                     sign_map=BlobStreamer.sign_map)
                over_draw(self.img, blob_img, blob.box)

                # add to id_map
                over_draw(self._id_map, None, blob.box,
                          mask=blob.mask, fill_color=blob.id)

        # resize window to display
        self.frame = cv.resize(self.img, self.window_size,
                               interpolation=cv.INTER_NEAREST)
        super().update()

    def init_adj_disp(self):
        # image to blob conversion end,
        # return to default display mode
        self.is_zooming = False
        self.render = self._render
        self.background = self.img  # stay constant from here
        self.background[self.background == 255] = 32

        # id_map act like a hash table to look for blob id,
        # given mouse position
        self._id_map = cv.resize(self._id_map, self.window_size,
                                 interpolation=cv.INTER_NEAREST)
        self.id_map = self._id_map

        # New mouse callback, extra adj highlighting utility
        def mouse_call(event, x, y, flags, param):
            x = min(self.window_size[0]-1, max(x, 0))
            y = min(self.window_size[1]-1, max(y, 0))
            self.x = x
            self.y = y

            if event == cv.EVENT_LBUTTONDOWN and not self.is_zooming:
                self.x1 = x
                self.y1 = y
                self.render = self._render_draw_rectangle
            elif event == cv.EVENT_LBUTTONUP:
                if not self.is_zooming:
                    if x != self.x1 and y != self.y1:
                        self.x2 = x
                        self.y2 = y
                        if x < self.x1:
                            self.x1, self.x2 = x, self.x1
                        if y < self.y1:
                            self.y1, self.y2 = y, self.y1
                        self.render = self._zoomed_render
                        self.is_zooming = True
                        self.id_map = cv.resize(self._id_map[self.y1:self.y2,
                                                             self.x1:self.x2],
                                                self.window_size,
                                                interpolation=cv.INTER_NEAREST)

                else:
                    self.is_zooming = False
                    self.render = self._render
                    self.id_map = self._id_map

            elif event == cv.EVENT_MOUSEMOVE:
                if self.pointing_blob_id != self.id_map[y, x]:
                    self.pointing_blob_id = self.id_map[y, x]
                    # override color of the blob
                    self.img = np.copy(self.background)
                    blob = self.blob_cls.get_instance(self.pointing_blob_id)
                    if blob is None:
                        print("\r", end="\t\t\t\t\t\t\t")
                        sys.stdout.flush()
                        return
                    over_draw(self.img, None, blob.box,
                              mask=blob.mask,
                              fill_color=(255, 255, 255))  # gray
                    # ... and its adjacents
                    for adj_blob, pose in blob.adj_blobs[0]:
                        if pose == 0:  # internal
                            color = (0, 0, 255)  # red
                        elif pose == 1:  # external
                            color = (0, 255, 0)  # green
                        elif pose == 2:  # open
                            color = (255, 0, 0)  # blue
                        else:
                            raise ValueError("adj pose id incorrect. Something is wrong")
                        over_draw(self.img, None, adj_blob.box,
                                  mask=adj_blob.mask,
                                  fill_color=color)
                    # ... print blobs properties.
                    print("\rblob:",
                          "id =", blob.id,
                          "sign =", "'+'" if blob.sign else "'-'",
                          "I =", blob.Dert['I'],
                          "G =", blob.Dert['G'],
                          "Dy =", blob.Dert['Dy'],
                          "Dx =", blob.Dert['Dx'],
                          "S =", blob.Dert['S'],
                          "Ly =", blob.Dert['Ly'],
                          end="\t\t\t")
                    sys.stdout.flush()

        cv.setMouseCallback(self.winname, mouse_call)


    def update_adj_disp(self):

        # resize window to display
        self.frame = cv.resize(self.img, self.window_size,
                               interpolation=cv.INTER_NEAREST)
        super().update()

    def update_blob_conversion(self, y, P_):
        self.update(y, P_)
        k = self.render()
        if k == 32:  # press space to pause
            while True:
                self.update(y)
                k = self.render()
                if k == 32:
                    break
                elif k == ord('q'):
                    self.stop()
                    return False
        elif k == ord('q'):
            self.stop()
            return False
        return True

    def end_blob_conversion(self, y, img_out_path=None):
        self.update(y)
        if img_out_path is not None:
            self.writeframe(img_out_path)
        print("\rPress A for adjacent view, or press Q to quit...", end="\t\t\t")
        sys.stdout.flush()
        while True:  # press Q key to quit
            self.update(y)
            k = self.render()
            if k == ord('a'):
                self.update_adj_disp()
                self.init_adj_disp()
                while self.render() != ord('q'):  # press Q key to quit
                    self.update_adj_disp()
                break
            elif k == ord('q'):
                break
        self.stop()