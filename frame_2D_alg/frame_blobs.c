#include <stdio.h>
#include <malloc.h>
#include <limits.h>
#include "bitarray.h"
#include "linked_list.h"

typedef struct {
    double I, iDy, iDx, G, Dy, Dx, M;
    unsigned long long S;
    char sign;
    unsigned int box[4];
    char fopen;
} Blob;

typedef struct {
    double I, G, Dy, Dx;
    unsigned long nblobs;
    Blob *blobs;
    LinkedList adj_pairs;
} FrameOfBlobs;

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

FrameOfBlobs* flood_fill(double *i_, double *idy_, double *idx_, double *g_,
                         double *dy_, double *dx_, double *m_,
                         int height, int width, long long *idmap,
                         unsigned long long startid, char deep)
/**
 * (docstring)
 */
{
    long size = height * width;  /* total number of derts */
    FrameOfBlobs *frame;         /* Container for the array of blobs */
    Blob    *blobs;              /* The array of blobs */

    // Count of blobs
    long nblobs = 0;

    // id of the blob being filled
    long blobid = startid;

    // A queue to store hash code of coordinates to be filled. FIFO.
    long *queue, qbeg, qend;

    // Maximum number of queueing derts in worst case scenario
    long qlen = height<width?width:height + 1;

    // A binary hash table to store state of a dert:
    // 1: filled
    // 0: unfilled
    // Index is obtained from the hash function:
    // i = hash(x, y) = y + x * width.
    // Where 'width' is the width of the frame, and (x, y) is the
    // coordinate.
    // Coordinate can be obtained by reverse hashing the index:
    // y = i / width
    // x = i % width
    long *fill_map;

    // Number of blobs in worst case scenario, where positive and
    // negative signed derts form a chessboard-like pattern (there's
    // one positive blob and (size/2) negative blobs). Serve as a
    // baseline to allocate memory firsthand.
    long nb_wst = size / 2 + 2;

    // Adjacent pair
    LinkedList adj_pairs;
    adj_pairs.first = NULL;

    // Memory allocation
    frame = (FrameOfBlobs*) malloc(sizeof(FrameOfBlobs));
    blobs = (Blob*) malloc(nb_wst * sizeof(Blob));
    fill_map = (long*) malloc((size/32 + 1) * sizeof(long));
    queue = (long*) malloc(qlen * sizeof(long));

    // Fill values
    frame->I = 0;
    frame->G = 0;
    frame->Dy = 0;
    frame->Dx = 0;
    clearbits(fill_map, size);

    // Loop through all derts
    for(int i = 0; i < size; i++)
        if(!testbit(fill_map, i)) {  /* ignore filled derts */
            setbit(fill_map, i);    /* set current dert as filled */

            double I = 0, G = 0, Dy = 0, Dx = 0, S = 0,
                   iDy = 0, iDx = 0, M = 0;
            int box[4] = {INT_MAX, 0, INT_MAX, 0};
            char sign = g_[i] > 0,
                 fopen = 0;
            long *adj_filter;
            adj_filter = (long*) malloc((nblobs/32 + 1) * sizeof(long));
            clearbits(adj_filter, nblobs);

            // Do flood fill
            qbeg = 0;
            qend = 1;
            queue[qbeg] = i;
            while(qbeg != qend) {
                long j = queue[qbeg++];           /* pop last dert's index */
                if(qbeg >= qlen) qbeg = 0;

                // Accumulation
                I += i_[j];
                G += g_[j];
                Dy += dy_[j];
                Dx += dx_[j];
                if(deep) {
                    iDy += dy_[j];
                    iDx += dx_[j];
                    M += m_[j];
                }

                S++;

                idmap[j] = blobid;  /* Set blob's id to map */

                int y = j / width,  /* un-hash coordinate */
                    x = j % width;
                if(y < box[0]) box[0] = y;
                if(y > box[1]) box[1] = y;
                if(x < box[2]) box[2] = x;
                if(x > box[3]) box[3] = x;

                // loop through adjacent coordinates, 8 if sign else 4
                for(int dir = 0; dir < (sign?8:4); dir++) {
                    int y2 = y + adj_offset[dir][0],
                        x2 = x + adj_offset[dir][1];
                    // check if image boundary is reached
                    if(y2 < 0 || y2 >= height ||
                       x2 < 0 || x2 >= width) fopen = 1;
                    else {
                        int k = y2 * width + x2;  /* hash coordinate */
                        // ignore filled
                        if(testbit(fill_map, k)) {
                            if(sign != (g_[k] > 0)) {
                                // "bind" adjacents
                                if(!testbit(adj_filter, idmap[k])) {
                                    setbit(adj_filter, idmap[k]);
                                    long long packed_pair = (idmap[k] << 32) + blobid;
                                    ll_appendleft(&adj_pairs, packed_pair);
                                }
                            }
                            continue;
                        }
                        // check if same-signed
                        if(sign == (g_[k] > 0)) {
                            setbit(fill_map, k);    /* set current dert as filled */
                            queue[qend++] = k;  /* append this hash */
                            if(qend >= qlen) qend = 0;
                        }
                    }
                }
            }

            // Blob is terminated
            frame->I += I;
            frame->G += G;
            frame->Dy += Dy;
            frame->Dx += Dx;
            blobs[nblobs].I = I;
            blobs[nblobs].G = G;
            blobs[nblobs].Dy = Dy;
            blobs[nblobs].Dx = Dx;
            if(deep) {
                blobs[nblobs].iDy = iDy;
                blobs[nblobs].iDx = iDx;
                blobs[nblobs].M = M;
            }
            blobs[nblobs].S = S;
            blobs[nblobs].box[0] = box[0];
            blobs[nblobs].box[1] = box[1] + 1;
            blobs[nblobs].box[2] = box[2];
            blobs[nblobs].box[3] = box[3] + 1;
            blobs[nblobs].sign = sign;
            blobs[nblobs].fopen = fopen;
            nblobs++;
            blobid++;

            free(adj_filter);
        }
    // Deallocate memory
    free(fill_map);
    free(queue);

    // Return frame of blobs
    frame->nblobs = nblobs;
    frame->blobs = blobs;
    frame->adj_pairs = adj_pairs;

    return frame;
}

void clean_up(FrameOfBlobs *frame) {
    free(frame->blobs);
    free(frame);
}