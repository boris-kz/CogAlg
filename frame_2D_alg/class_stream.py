"""
Provide streaming methods to monitor some of frame_2D_alg operations.
"""

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

    def _render_no_record(self):
        """Render visualization to screen."""
        cv.imshow(winname=self.winname, mat=self.view)
        return cv.waitKey(1)

    def _render_and_record(self):
        cv.imshow(winname=self.winname, mat=self.view)
        self.video_writer.write(self.view)
        return cv.waitKey(1)

    def _render_draw_rectangle(self):
        cv.rectangle(self.view,
                     (self.x1, self.y1),
                     (self.x, self.y),
                     (0, 0, 255), 2)
        self._render()

    def _zoomed_render(self):
        self.view = cv.resize(self.view[self.y1:self.y2,
                                        self.x1:self.x2],
                              self.window_size)
        self._render()


class Img2BlobStreamer(Streamer):
    """
    Use this class to monitor the actions of image_to_blobs in frame_blobs.
    """
    sign_map = {False: BLACK, True: WHITE}  # sign_map for terminated blobs
    sign_map_unterminated = {False: DGREY, True: LGREY}  # sign_map for unterminated blobs

    def __init__(self, blob_cls, frame, winname='image_to_blobs',
                 record_path=None):
        self.blob_cls = blob_cls
        height, width = frame['dert__'].shape[1:]
        Streamer.__init__(self, window_size=(width, height),
                          winname=winname,
                          record_path=record_path)
        self.img = blank_image((height, width))
        self.incomplete_blob_ids = set()
        self.first_id = 0

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
            if blob is None:
                self.incomplete_blob_ids.remove(blob_id)
                continue
            elif blob.open_stacks == 0:  # terminated blob has no open_stack
                blob_box = blob.box
                self.incomplete_blob_ids.remove(blob_id)
                blob_img = draw_blob(blob, blob_box=blob_box,
                                     sign_map=Img2BlobStreamer.sign_map)
                over_draw(self.img, blob_img, blob_box)

        # resize window to display
        self.frame = cv.resize(self.img, self.window_size)
        super().update()

    def update_after_conversion(self):
        super().update()