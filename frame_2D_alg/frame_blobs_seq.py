import numpy as np
from collections import deque
from class_cluster import ClusterStructure, NoneType
from utils import minmax
'''
Khanh's version
'''
ave = 30  # filter or hyper-parameter, set as a guess, latter adjusted by feedback

class CBlob(ClusterStructure):
    # Derts
    I = int
    G = int
    Dy = int
    Dx = int
    S = int
    # other data
    box = list
    sign = NoneType
    dert_coord_ = set  # let derts' id be their coords
    root_dert__ = object
    adj_blobs = list
    fopen = bool

def frame_blobs_parallel(dert__):
    height, width = dert__[0].shape
    id_map = np.full((height, width), -1, 'int64')  # blob's id per dert, initialized with -1
    blob_ = []
    for y in range(height):
        for x in range(width):
            if id_map[y, x] == -1:  # ignore filled/clustered derts (blob id != -1)
                # initialize new blob
                blob = CBlob(I=dert__[0][y, x], G=dert__[1][y, x] - ave,
                             Dy=dert__[2][y, x], Dx=dert__[3][y, x],
                             sign=dert__[1][y, x] - ave > 0, root_dert__=dert__)
                blob_.append(blob)
                id_map[y, x] = blob.id

                # flood fill the blob, start from current position
                unfilled_derts = deque([(y, x)])
                while unfilled_derts:
                    y1, x1 = unfilled_derts.popleft()

                    # add dert to blob
                    blob.dert_coord_.add((y1, x1))  # add dert coordinate to blob
                    blob.I += dert__[0][y1, x1]
                    blob.G += dert__[1][y1, x1] - ave
                    blob.Dy += dert__[2][y1, x1]
                    blob.Dx += dert__[3][y1, x1]
                    blob.S += 1

                    # determine neighbors' coordinates, 4 for -, 8 for +
                    if blob.sign:   # include diagonals
                        adj_dert_coords = [(y1 - 1, x1 - 1), (y1 - 1, x1),
                                           (y1 - 1, x1 + 1), (y1, x1 + 1),
                                           (y1 + 1, x1 + 1), (y1 + 1, x1),
                                           (y1 + 1, x1 - 1), (y1, x1 - 1)]
                    else:
                        adj_dert_coords = [(y1 - 1, x1), (y1, x1 + 1),
                                           (y1 + 1, x1), (y1, x1 - 1)]

                    # search through neighboring derts
                    for y2, x2 in adj_dert_coords:
                        # check if image boundary is reached
                        if (y2 < 0 or y2 >= height or
                            x2 < 0 or x2 >= width):
                            blob.fopen = True
                        # check if same-signed
                        elif id_map[y2, x2] == -1:
                            # check if same-signed
                            if blob.sign == (dert__[1][y2, x2] - ave > 0):
                                id_map[y2, x2] = blob.id  # add blob ID to each dert
                                unfilled_derts.append((y2, x2))
                            # else assign adjacents
                            else:
                                # TODO: assign adjacents
                                pass
                # terminate blob
                y_coords, x_coords = zip(*blob.dert_coord_)
                y0, yn = minmax(y_coords)
                x0, xn = minmax(x_coords)
                blob.box = (
                    y0, yn + 1,  # y0, yn
                    x0, xn + 1,  # x0, xn
                )
                # got a set of coordinates, no need for mask?
    return blob_

if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt
    from time import time
    from frame_2D_alg.frame_blobs import comp_pixel
    from frame_2D_alg.utils import imread

    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('-i', '--image', help='path to image file', default='./images//raccoon_eye.jpg')
    argument_parser.add_argument('-v', '--verbose', help='print details, useful for debugging', type=int, default=1)
    argument_parser.add_argument('-n', '--intra', help='run intra_blobs after frame_blobs', type=int, default=0)
    argument_parser.add_argument('-r', '--render', help='render the process', type=int, default=1)
    arguments = vars(argument_parser.parse_args())
    image = imread(arguments['image'])
    verbose = arguments['verbose']
    intra = arguments['intra']
    render = arguments['render']

    start_time = time()
    dert__ = comp_pixel(image)
    blob_ = frame_blobs_parallel(dert__)
    bmap = np.full_like(image, 127, 'uint8')
    print(f"{len(blob_)} blobs formed in {time() - start_time} seconds")

    for blob in blob_:
        for y, x in blob.dert_coord_:
            bmap[y, x] = blob.sign * 255
    plt.imshow(bmap, 'gray')
    plt.show()