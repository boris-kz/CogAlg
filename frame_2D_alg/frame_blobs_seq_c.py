import ctypes
import numpy as np
import matplotlib.pyplot as plt
from time import time
from frame_blobs_yx import comp_pixel
from utils import imread
'''
Khanh's version, implemented in C
'''
ave = 30  # filter or hyper-parameter, set as a guess, latter adjusted by feedback

frame_blobs_parallel = ctypes.CDLL("frame_blobs_parallel.so").frame_blobs_parallel

img = imread('./images/raccoon.jpg')
dert__ = [*map(lambda a: a.astype('float32'),
               comp_pixel(img))]
start_time = time()
height, width = dert__[0].shape
i = dert__[0].ctypes.data
g = dert__[1].ctypes.data
dy = dert__[2].ctypes.data
dx = dert__[3].ctypes.data
bmap = np.empty((height, width), 'uint8')
nblobs = frame_blobs_parallel(i, g, dy, dx, height, width, ave,
                              bmap.ctypes.data)
print(f"{nblobs} blobs formed in {time() - start_time} seconds")
plt.imshow(bmap, 'gray')
plt.show()

''' C code: '''

#include <stdio.h>
#include <stdint.h>
#include <malloc.h>
#include <string.h>
#define QUEUE_MAX_LEN   2000    /* max queue length */

typedef struct {
    float I, G, Dy, Dx, S;
    char sign, fopen;
} struct_blob;

int adj_offset[8][2] = {
    {-1, 0},    /* top */
    {0, 1},     /* right */
    {1, 0},     /* bottom */
    {0, -1},    /* left */
    {-1, -1},   /* top-left */
    {-1, 1},    /* top-right */
    {1, 1},     /* btm-right */
    {1, -1},    /* btm-left */
};

int frame_blobs_parallel(float *i_, float *g_, float *dy_, float *dx_,
                         int height, int width, int ave,
                         uint8_t *bmap) {
    static int queue[QUEUE_MAX_LEN],    /* a queue for FIFO data */
               qbeg, qend;
    int nblobs = 0,                     /* total number of blobs */
        size = height * width;          /* total number of derts */
    char *fill_map;                     /* id_map to track flood fill */

    // initialize fill_map
    fill_map = (char*) malloc(size * sizeof(char));
    memset(fill_map, 0, sizeof(fill_map));

    // Loop through all derts
    for(int i = 0; i < size; i++)
        if(!fill_map[i]) {  /* ignore filled derts */
            fill_map[i] = 1;
            nblobs++;
            float I = 0, G = 0, Dy = 0, Dx = 0, S = 0;
            char sign = g_[i] - ave > 0,
                 fopen = 0;

            // do flood fill
            qbeg = 0;
            qend = 1;
            queue[qbeg] = i;
            while(qbeg != qend) {
                int j = queue[qbeg++];           /* pop last dert's index */
                if(qbeg >= QUEUE_MAX_LEN) qbeg = 0;
                // TODO: add coordinate container
                I += i_[j];
                G += g_[j];
                Dy += dy_[j];
                Dx += dx_[j];
                S += 1;
                bmap[j] = sign?255:0;

                int y = j / width,
                    x = j % width;
                // loop through adjacent coordinates, 8 if sign else 4
                for(int dir = 0; dir < (sign?8:4); dir++) {
                    int y2 = y + adj_offset[dir][0],
                        x2 = x + adj_offset[dir][1];
                    // check if image boundary is reached
                    if(y2 < 0 || y2 >= height ||
                       x2 < 0 || x2 >= width) fopen = 1;
                    else {
                        int k = y2 * width + x2;
                        // ignore filled
                        if(fill_map[k]) continue;
                        // check if same-signed
                        if(sign == (g_[k] - ave > 0)) {
                            fill_map[k] = 1;
                            queue[qend++] = k;  /* append this hash */
                            if(qend >= QUEUE_MAX_LEN) qend = 0;
                        }
                        // else assign adjacents
                        else {
                            // TODO: assign adjacents
                        }
                    }
                }
            }
        }
    free(fill_map);

    return nblobs;
}