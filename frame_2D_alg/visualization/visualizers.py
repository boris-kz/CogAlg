import os
from collections import namedtuple

WHITE = 255, 255, 255
BLACK = 0, 0, 0
RED = 0, 0, 255
GREEN = 0, 255, 0
BLUE = 255, 0, 0
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

    def reset(self):
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

    def update_img(self):
        self.state.imshow_obj.set_data(self.state.img)
        self.state.fig.canvas.draw_idle()

    def update_element_id(self):
        self.update_img()
        self.update_info()

    def update_info(self):
        # clear screen
        os.system('cls' if os.name == 'nt' else 'clear')
        print("hit 'q' to exit")


    def go_deeper(self, fd):
        raise NotImplementedError

    def go_back(self):
        self.state.layers_stack.pop()
        return self.state.layers_stack[-1].visualizer

    def get_highlighted_element(self):
        return self.state.element_cls.get_instance(self.state.element_id)


class BlobVisualizer(Visualizer):
    def reset(self):
        super().reset()
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
        self.state.img = self.state.background.copy()
        self.state.element_id = -1
        self.update_img()
        self.update_info()

    def update_gradient(self):
        if self.state.quiver is not None:
            self.state.quiver.remove()
            self.state.quiver = None

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
        self.update_gradient()
        super().update_img()

    def update_element_id(self):
        # override color of the element
        self.state.img[:] = self.state.background[:]
        blob = self.get_highlighted_element()
        if blob is None:
            self.update_img()
            return

        # paint over the blob ...
        self.state.img[self.state.img_slice][blob.box.slice()][blob.mask__] = WHITE
        # ... and its adjacents
        for adj_blob, pose in zip(*blob.adj_blobs):
            self.state.img[self.state.img_slice][adj_blob.box.slice()][adj_blob.mask__] = POSE2COLOR[pose]
        # Finally, update the image
        super().update_element_id()

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
        root_blob = self.get_highlighted_element()
        if root_blob is None: return

        if fd:
            # edge visualizer
            if not root_blob.dlayers or not root_blob.dlayers[0]: return
            edge = root_blob.dlayers[0][0]
            if not edge.node_t: return
            PP_ = [
                # TODO : add dummy PP from edge
                # TODO : add box and mask__ from blob box (1st layer dummy PP covers the whole edge)
            ]
            visualizer = SliceVisualizer(self.state, PP_)
            self.state.layers_stack.append(layerT(edge, "edge", visualizer))
        else:
            # frame visualizer (r+blob)
            if not root_blob.rlayers or not root_blob.rlayers[0]: return
            visualizer = self
            self.state.layers_stack.append(layerT(root_blob, "rblob", visualizer))   # reuse this visualizer

        return visualizer


class SliceVisualizer(Visualizer):
    def __init__(self, state, PP_):
        super().__init__(state)
        self.PP_ = PP_

    def reset(self):
        super().reset()

        # TODO : reset image and create PP id map

    # def update_blob_slices():
    #     if state.blob_slices is not None:
    #         for line, L_text in state.blob_slices:
    #             line.remove()
    #             L_text.remove()
    #         state.blob_slices = None
    #
    #     blob = state.blob_cls.get_instance(state.blob_id)
    #     if blob is None or not blob.dlayers or not blob.dlayers[0] or not state.show_slices:
    #         return
    #     edge = blob.dlayers[0][0]
    #     if not edge.node_t: return
    #     print(f"PP_fd = {state.PP_fd}", end="")
    #     y0, x0, *_ = blob.ibox
    #
    #     state.blob_slices = []
    #     for P in get_P_(edge, state.PP_fd):        # show last layer
    #         y, x = P.yx
    #         y_, x_, *_ = np.array([*zip(*P.dert_)])
    #         L = len(x_)
    #         if L > 1:
    #             blob_slices_plot = ax.plot(x_+x0, y_+y0, 'bo-', linewidth=1, markersize=2)[0]
    #         else:
    #             s, c = P.axis
    #             x_ = np.array([x-c, x, x+c])
    #             y_ = np.array([y-s, y, y+s])
    #             blob_slices_plot = ax.plot(x_ + x0, y_ + y0, 'b-', linewidth=1, markersize=2)[0]
    #         state.blob_slices += [(
    #             blob_slices_plot,
    #             ax.text(x+x0, y+y0, str(L), color = 'b', fontsize = 12),
    #         )]
    #
    # def update_P_links():
    #     if state.P_links is not None:
    #         for line in state.P_links:
    #             line.remove()
    #         state.P_links = None
    #
    #     blob = state.blob_cls.get_instance(state.blob_id)
    #     if blob is None or not blob.dlayers or not blob.dlayers[0] or not state.show_links:
    #         return
    #     edge = blob.dlayers[0][0]
    #     if not edge.node_t: return
    #     y0, x0, *_ = blob.ibox
    #     state.P_links = []
    #     for P in get_P_(edge, state.PP_fd):
    #         for derP in P.link_H[0]:
    #             (_y, _x), (y, x) = (derP._P.yx, derP.P.yx)
    #             state.P_links += ax.plot([_x+x0,x+x0], [_y+y0,y+y0], 'ko-', linewidth=2, markersize=4)

    def update_img(self):
        # TODO : update slices and links from all PP deeper layers
        super().update_img()

    def update_element_id(self):
        # TODO : high-light PPs
        super().update_element_id()

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