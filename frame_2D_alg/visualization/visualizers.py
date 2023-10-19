import os
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple

WHITE = 255, 255, 255
BLACK = 0, 0, 0
RED = 0, 0, 255
GREEN = 0, 255, 0
BLUE = 255, 0, 0
GREY = 128, 128, 128
DARK_RED = 0, 0, 128
DARK_GREEN = 0, 128, 0
DARK_BLUE = 128, 0, 0
POSE2COLOR = {
    0:RED,
    1:GREEN,
    2:BLUE,
}

BACKGROUND_COLOR = 128  # Pixel at this value can be over-written

class Visualizer:
    def __init__(self, element_, root_visualizer=None,
                 shape=None, title="Visualization"):
        self.root_visualizer = root_visualizer
        self.element_ = element_
        self.element_cls = self.element_[0].__class__
        self.hovered_element = None
        self.element_stack = []
        self.img = None
        self.img_slice = None
        if self.root_visualizer is None:
            height, width = shape
            self.background = np.full((height, width, 3), BACKGROUND_COLOR, 'uint8')
            self.idmap = np.full((height, width), -1, 'int64')

            self.fig, self.ax = plt.subplots()
            self.fig.canvas.set_window_title(title)
            self.imshow_obj = self.ax.imshow(self.background)
        else:
            self.background = self.root_visualizer.background
            self.idmap = self.root_visualizer.idmap

            self.fig = self.root_visualizer.fig
            self.ax = self.root_visualizer.ax
            self.imshow_obj = self.root_visualizer.imshow_obj

    def reset(self):
        self.clear_plot()
        self.img = self.background.copy()
        self.hovered_element = None

    def update_img(self, flags):
        self.imshow_obj.set_data(self.img)
        self.fig.canvas.draw_idle()

    def update_info(self):
        # clear screen
        os.system('cls' if os.name == 'nt' else 'clear')
        print("hit 'q' to exit")

    def go_back(self):
        if self.element_stack:
            self.element_ = self.element_stack.pop()
            return self
        elif self.root_visualizer is not None:
            self.clear_plot()  # remove private plots
            return self.root_visualizer
        # return None otherwise

    def append_element_stack(self, new_element_):
        self.element_stack.append(self.element_)
        self.element_ = new_element_

    def get_hovered_element(self, x, y):
        if x is None or y is None: return None
        else: return self.element_cls.get_instance(self.idmap[round(y), round(x)])

    # to be replaced:
    def update_hovered_element(self, hovered_element):
        raise NotImplementedError

    def clear_plot(self):
        raise NotImplementedError

    def go_deeper(self, fd):
        raise NotImplementedError


class BlobVisualizer(Visualizer):

    def __init__(self, frame, **kwargs):     # only for frame
        super().__init__(element_=frame.rlayers[0], shape=frame.box[-2:], **kwargs)
        # private fields
        self.gradient = np.zeros((2, *self.background.shape[:2]), float)
        self.gradient_mask = np.zeros(self.background.shape[:2], bool)
        self.show_gradient = False
        self.quiver = None

    def reset(self):
        blob_ = self.element_
        self.img_slice = blob_[0].root_ibox.slice()
        # Prepare blob ID map and background
        self.background[:] = BACKGROUND_COLOR
        self.idmap[:] = -1
        local_idmap = self.idmap[self.img_slice]
        local_background = self.background[self.img_slice]
        for blob in blob_:
            local_idmap[blob.box.slice()][blob.mask__] = blob.id  # fill idmap with blobs' ids
            local_background[blob.box.slice()][blob.mask__] = blob.sign * 32  # fill image with blobs
        super().reset()

    def update_gradient(self):
        blob = self.hovered_element
        if blob is None: return

        # Reset gradient
        self.gradient[:] = 1e-3
        self.gradient_mask[:] = False

        # Use indexing to get the gradient of the blob
        box_slice = blob.ibox.slice()
        self.gradient[1][box_slice] = -blob.der__t.dy
        self.gradient[0][box_slice] = blob.der__t.dx
        self.gradient_mask[box_slice] = blob.mask__

        # Apply quiver
        self.quiver = self.ax.quiver(
            *self.gradient_mask.nonzero()[::-1],
            *self.gradient[:, self.gradient_mask])

    def update_img(self, flags):
        self.clear_plot()
        if flags.show_gradient: self.update_gradient()
        super().update_img(flags)

    def update_hovered_element(self, hovered_element):
        blob = self.hovered_element = hovered_element

        # override color of the element
        self.img[:] = self.background[:]

        if blob is None: return

        # paint over the blob ...
        self.img[self.img_slice][blob.box.slice()][blob.mask__] = WHITE
        # ... and its adjacents
        for adj_blob, pose in zip(*blob.adj_blobs):
            self.img[self.img_slice][adj_blob.box.slice()][adj_blob.mask__] = POSE2COLOR[pose]

    def update_info(self):
        super().update_info()

        print("hit 'd' to toggle show gradient")
        print("ctrl + click to show deeper rblobs")
        print("shift + click to show edge PPs")

        blob = self.hovered_element
        if blob is None:
            print("No blob highlighted")
            return

        print("blob:")
        print(f"┌{'─'*10}┬{'─'*6}┬{'─'*10}┬{'─'*10}┬{'─'*10}┬{'─'*10}"
              f"┬{'─'*10}┬{'─'*10}┬{'─'*16}┐")
        print(f"│{'id':^10}│{'sign':^6}│{'I':^10}│{'Dy':^10}│{'Dx':^10}│{'G':^10}│"
              f"{'M':^10}│{'A':^10}│{'box':^16}│")
        print(f"├{'─' * 10}┼{'─' * 6}┼{'─' * 10}┼{'─' * 10}┼{'─' * 10}┼{'─' * 10}"
              f"┼{'─' * 10}┼{'─' * 10}┼{'─' * 16}┤")
        print("\r"f"│{blob.id:^10}│{'+' if blob.sign else '-':^6}│"
              f"{blob.I:^10.2e}│{blob.Dy:^10.2e}│{blob.Dx:^10.2e}│{blob.G:^10.2e}│"
              f"{blob.M:^10.2e}│{blob.A:^10.2e}│"
              f"{'-'.join(map(str, blob.box)):^16}│")
        print(f"└{'─' * 10}┴{'─' * 6}┴{'─' * 10}┴{'─' * 10}┴{'─' * 10}┴{'─' * 10}"
              f"┴{'─' * 10}┴{'─' * 10}┴{'─' * 16}┘")

    def clear_plot(self):
        if self.quiver is not None:
            self.quiver.remove()
            self.quiver = None

    def go_deeper(self, fd):
        blob = self.hovered_element
        if blob is None: return

        if fd:
            from vectorize_edge_blob.classes import CPP
            # edge visualizer
            if not blob.dlayers or not blob.dlayers[0]: return
            edge = blob.dlayers[0][0]
            if not edge.node_t: return
            PP_ = [
                CPP(
                    fd=1,
                    ptuple=[blob.I, blob.G, edge.M, edge.Ma, [blob.Dy, blob.Dx], blob.A],
                    derH=edge.derH,
                    valt=edge.valt,
                    rdnt=edge.rdnt,
                    mask__=blob.mask__,
                    root=blob,
                    node_t=edge.node_t,
                    fback_t=edge.fback_t,
                    rng=edge.rng,
                    box=blob.box,
                )
            ]
            self.clear_plot()
            return SliceVisualizer(img_slice=blob.ibox.slice(), element_=PP_, root_visualizer=self)
        else:
            # frame visualizer (r+blob)
            if not blob.rlayers or not blob.rlayers[0]: return
            self.append_element_stack(blob.rlayers[0])
            return self


class SliceVisualizer(Visualizer):
    def __init__(self, img_slice, **kwargs):
        super().__init__(**kwargs)
        self.img_slice = img_slice  # all PPs have same frame of reference
        # private fields
        self.P__ = None
        self.show_slices = False
        self.show_links = False
        self.P_links = None
        self.blob_slices = None

    def reset(self):
        self.P__ = {PP.id:get_P_(PP) for PP in self.element_}
        # Prepare ID map and background
        self.background[:] = BACKGROUND_COLOR
        self.idmap[:] = -1
        local_idmap = self.idmap[self.img_slice]
        local_background = self.background[self.img_slice]
        for PP in self.element_:
            local_idmap[PP.box.slice()][PP.mask__] = PP.id  # fill idmap with PP's id
            local_background[PP.box.slice()][PP.mask__] = DARK_GREEN
        super().reset()

    def update_blob_slices(self):
        if not self.P__: return
        PP = self.hovered_element
        if PP is None: return
        y0 = self.img_slice[0].start
        x0 = self.img_slice[1].start
        self.blob_slices = []
        for P in self.P__[PP.id]:
            y, x = P.yx
            s, c = P.axis
            L = len(P.dert_)
            y_, x_ = (np.multiply([[s], [c]], [[-1, 0, 1]]) + [[y], [x]]) if (L == 1) else np.array(P.dert_).T[:2]

            self.blob_slices += [(
                *self.ax.plot(x_+x0, y_+y0, 'b-', linewidth=1, markersize=2),
                self.ax.text(x+x0, y+y0, str(L), color = 'b', fontsize = 12),
            )]

    def update_P_links(self):
        if not self.P__: return
        PP = self.hovered_element
        if PP is None: return
        y0 = self.img_slice[0].start
        x0 = self.img_slice[1].start
        self.P_links = []
        for P in self.P__[PP.id]:
            for derP in P.link_H[-1]:
                _P = derP if isinstance(derP, type(P)) else derP._P
                (_y, _x), (y, x) = _P.yx, P.yx
                self.P_links += self.ax.plot([_x+x0,x+x0], [_y+y0,y+y0], 'ko-', linewidth=2, markersize=4)

    def update_img(self, flags):
        self.clear_plot()
        if flags.show_slices: self.update_blob_slices()
        if flags.show_links: self.update_P_links()
        super().update_img(flags)

    def update_hovered_element(self, hovered_element):
        PP = self.hovered_element = hovered_element

        # override color of the element
        self.img[:] = self.background[:]

        if PP is None: return

        # paint over the blob ...
        self.img[self.img_slice][PP.box.slice()][PP.mask__] = WHITE

    def update_info(self):
        print("hit 'z' to toggle show slices")
        print("hit 'x' to toggle show links")
        print("ctrl + click to show deeper rPP")
        print("shift + click to show deeper dPP")

    def clear_plot(self):
        if self.P_links is not None:
            for line in self.P_links:
                line.remove()
            self.P_links = None
        if self.blob_slices is not None:
            for line, L_text in self.blob_slices:
                line.remove()
                L_text.remove()
            self.blob_slices = None

    def go_deeper(self, fd):
        PP = self.hovered_element
        if PP is None: return
        if not PP.node_t: return
        if not isinstance(PP.node_t[0], list): return  # stop if no deeper layer (PP.node_t filled with Ps)
        subPP_ = PP.node_t[fd]
        if not subPP_: return
        self.append_element_stack(subPP_)
        return self


def get_P_(PP):
    P_ = []
    subPP_ = [PP]
    while subPP_:
        subPP = subPP_.pop()
        if not subPP.node_t: continue
        if isinstance(subPP.node_t[0], list):
            subPP_ += subPP.node_t[0] + subPP.node_t[1]
        else:  # is P_
            P_ += subPP.node_t

    return P_