import os
import numpy as np
from collections import namedtuple
from types import SimpleNamespace

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

layerT = namedtuple("layerT", "root,type,visualizer")

class Visualizer:
    def __init__(self, state):
        self.state = state

    def clear_plots(self):
        if self.state.quiver is not None:
            self.state.quiver.remove()
            self.state.quiver = None
        if self.state.P_links is not None:
            for line in self.state.P_links:
                line.remove()
            self.state.P_links = None
        if self.state.blob_slices is not None:
            for line, L_text in self.state.blob_slices:
                line.remove()
                L_text.remove()
            self.state.blob_slices = None

    def reset(self):
        self.clear_plots()
        self.state.img = self.state.background.copy()
        self.state.element_id = -1
        self.update_img()
        self.update_info()

    def update_img(self):
        self.state.imshow_obj.set_data(self.state.img)
        self.state.fig.canvas.draw_idle()

    def update_element_id(self):
        raise NotImplementedError

    def update_info(self):
        # clear screen
        os.system('cls' if os.name == 'nt' else 'clear')
        print("hit 'q' to exit")


    def go_deeper(self, fd):
        raise NotImplementedError

    def go_back(self):
        if len(self.state.layers_stack) == 1: return
        self.state.layers_stack.pop()
        return self.state.layers_stack[-1].visualizer

    def get_highlighted_element(self):
        return self.state.element_cls.get_instance(self.state.element_id)


class BlobVisualizer(Visualizer):
    def reset(self):
        blob_ = self.state.layers_stack[-1].root.rlayers[0]
        self.state.element_cls = blob_[0].__class__
        self.state.img_slice = blob_[0].root_ibox.slice()
        self.state.background[:] = BACKGROUND_COLOR
        self.state.idmap[:] = -1
        # Prepare blob ID map and background
        local_idmap = self.state.idmap[self.state.img_slice]
        local_background = self.state.background[self.state.img_slice]
        for blob in blob_:
            local_idmap[blob.box.slice()][blob.mask__] = blob.id  # fill idmap with blobs' ids
            local_background[blob.box.slice()][blob.mask__] = blob.sign * 32  # fill image with blobs
        super().reset()


    def update_gradient(self):

        blob = self.get_highlighted_element()
        if blob is None or not self.state.show_gradient: return

        # Reset gradient
        self.state.gradient[:] = 1e-3
        self.state.gradient_mask[:] = False

        # Use indexing to get the gradient of the blob
        box_slice = blob.ibox.slice()
        self.state.gradient[1][box_slice] = -blob.der__t.dy
        self.state.gradient[0][box_slice] = blob.der__t.dx
        self.state.gradient_mask[box_slice] = blob.mask__

        # Apply quiver
        self.state.quiver = self.state.ax.quiver(
            *self.state.gradient_mask.nonzero()[::-1],
            *self.state.gradient[:, self.state.gradient_mask])

    def update_img(self):
        self.clear_plots()
        self.update_gradient()
        super().update_img()

    def update_element_id(self):
        # override color of the element
        self.state.img[:] = self.state.background[:]
        blob = self.get_highlighted_element()
        if blob is None: return

        # paint over the blob ...
        self.state.img[self.state.img_slice][blob.box.slice()][blob.mask__] = WHITE
        # ... and its adjacents
        for adj_blob, pose in zip(*blob.adj_blobs):
            self.state.img[self.state.img_slice][adj_blob.box.slice()][adj_blob.mask__] = POSE2COLOR[pose]

    def update_info(self):
        super().update_info()

        print("hit 'd' to toggle show gradient")
        print("ctrl + click to show deeper rblobs")
        print("shift + click to show edge PPs")

        blob = self.get_highlighted_element()
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

    def go_deeper(self, fd):
        blob = self.get_highlighted_element()
        if blob is None: return

        if fd:
            from vectorize_edge_blob.classes import CPP
            # edge visualizer
            if not blob.dlayers or not blob.dlayers[0]: return
            edge = blob.dlayers[0][0]
            if not edge.node_t: return
            PP_ = [
                # SimpleNamespace(
                    # id=edge.id,
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
            visualizer = SliceVisualizer(self.state, PP_)
            self.state.img_slice = blob.ibox.slice()
            self.state.layers_stack.append(layerT(edge, "edge", visualizer))
        else:
            # frame visualizer (r+blob)
            if not blob.rlayers or not blob.rlayers[0]: return
            visualizer = self
            self.state.layers_stack.append(layerT(blob, "rblob", visualizer))   # reuse this visualizer

        return visualizer


class SliceVisualizer(Visualizer):
    def __init__(self, state, PP_):
        super().__init__(state)
        self.PP_ = PP_
        self.P__ = {PP.id:get_P_(PP) for PP in PP_}

    def reset(self):
        self.state.element_cls = self.PP_[0].__class__
        self.state.background[:] = BACKGROUND_COLOR
        self.state.idmap[:] = -1
        # Prepare ID map and background
        local_idmap = self.state.idmap[self.state.img_slice]
        local_background = self.state.background[self.state.img_slice]
        for PP in self.PP_:
            local_idmap[PP.box.slice()][PP.mask__] = PP.id  # fill idmap with PP's id
            local_background[PP.box.slice()][PP.mask__] = DARK_GREEN
        super().reset()

    def update_blob_slices(self):
        if not self.state.show_slices: return
        if not self.P__: return
        PP = self.get_highlighted_element()
        if PP is None: return
        y0 = self.state.img_slice[0].start
        x0 = self.state.img_slice[1].start
        self.state.blob_slices = []
        for P in self.P__[PP.id]:
            y, x = P.yx
            s, c = P.axis
            L = len(P.dert_)
            y_, x_ = (np.multiply([[s], [c]], [[-1, 0, 1]]) + [[y], [x]]) if (L == 1) else np.array(P.dert_).T[:2]

            self.state.blob_slices += [(
                *self.state.ax.plot(x_+x0, y_+y0, 'b-', linewidth=1, markersize=2),
                self.state.ax.text(x+x0, y+y0, str(L), color = 'b', fontsize = 12),
            )]

    def update_P_links(self):
        if not self.state.show_links: return
        if not self.P__: return
        PP = self.get_highlighted_element()
        if PP is None: return
        y0 = self.state.img_slice[0].start
        x0 = self.state.img_slice[1].start
        self.state.P_links = []
        for P in self.P__[PP.id]:
            for derP in P.link_H[-1]:
                (_y, _x), (y, x) = (derP._P.yx, derP.P.yx)
                self.state.P_links += self.state.ax.plot([_x+x0,x+x0], [_y+y0,y+y0], 'ko-', linewidth=2, markersize=4)

    def update_img(self):
        self.clear_plots()
        self.update_blob_slices()
        self.update_P_links()
        super().update_img()

    def update_element_id(self):
        # override color of the element
        self.state.img[:] = self.state.background[:]
        PP = self.get_highlighted_element()
        if PP is None: return

        # paint over the blob ...
        self.state.img[self.state.img_slice][PP.box.slice()][PP.mask__] = WHITE

    def update_info(self):
        print("hit 'z' to toggle show slices")
        print("hit 'x' to toggle show links")
        print("ctrl + click to show deeper rPP")
        print("shift + click to show deeper dPP")

    def go_deeper(self, fd):
        PP = self.get_highlighted_element()
        if PP is None: return
        if not PP.node_t: return
        if not isinstance(PP.node_t, list): return  # stop if no deeper layer
        subPP_ = PP.node_t[fd]
        if not subPP_: return
        visualizer = SliceVisualizer(self.state, subPP_)
        self.state.layers_stack.append(layerT(PP, "PP", visualizer))
        return visualizer

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