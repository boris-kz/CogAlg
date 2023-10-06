import os

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
            state.P_links = None
        if self.state.blob_slices is not None:
            for line, L_text in self.state.blob_slices:
                line.remove()
                L_text.remove()
            self.state.blob_slices = None

    def update_img(self):
        self.state.imshow_obj.set_data(self.state.img)
        self.state.fig.canvas.draw_idle()

    def update_element_id(self):
        raise NotImplementedError

    def update_info(self):
        # clear screen
        # os.system('cls' if os.name == 'nt' else 'clear')
        os.system("clear")
        print("hit 'q' to exit")
        # print("hit 'z' to toggle show slices")
        # print("hit 'x' to toggle show links")


    def go_deeper(self, fd):
        raise NotImplementedError

    def get_highlighted_element(self):
        return self.state.element_cls.get_instance(self.state.element_id)
class FrameVisualizer(Visualizer):
    def reset(self):
        super().reset()
        blob_ = self.state.layers_stack[-1][0].rlayers[0]
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
        if blob is None or not self.state.show_gradient:
            return

        # Reset gradient
        self.state.gradient[:] = 1e-3
        self.state.gradient_mask[:] = False

        # Use indexing to get the gradient of the blob
        box_slice = blob.ibox.slice()
        self.state.gradient[1][box_slice] = -blob.der__t.dy
        self.state.gradient[0][box_slice] = blob.der__t.dx
        self.state.gradient_mask[box_slice] = blob.mask__

        # Apply quiver
        self.state.quiver = ax.quiver(
            *self.state.gradient_mask.nonzero()[::-1],
            *self.state.gradient[:, state.gradient_mask])

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
        self.update_img()
        self.update_info()

    def update_info(self):
        super().update_info()

        print("hit 'd' to toggle show gradient")

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

    # def go_deeper(self):
    #     pass