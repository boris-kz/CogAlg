#include <stdio.h>
#include <malloc.h>
#include "bitarray.h"

typedef struct {
    int x, y;
} DertRef;

typedef struct {
    double I, G, Dy, Dx;
    unsigned long long S;
    char sign, fopen;
    DertRef *dert_ref;
} Blob;

typedef struct {
    double I, G, Dy, Dx;
    int nblobs;
    Blob *blobs_ptr;
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

FrameOfBlobs derts2blobs(double *i_, double *g_, double *dy_, double *dx_,
                         int height, int width, int ave,
                         unsigned int *idmap) {

    long size = height * width;  /* total number of derts */
    FrameOfBlobs *frame;         /* Container for the array of blobs */
    Blob    *blobs;              /* The array of blobs */

    // The table of dert coordinates, ordered by their blob ids. Each
    // blob has a pointer to its first DertRef.
    DertRef *dert_refs;

    // Filled dert counts, serve as tail for dert_refs table
    long nfilled = 0;

    // Total number of blobs. Also indicate id of the blob being filled
    long nblobs = 0;

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

    // A Binary hash table of adjacent pairs existence:
    // 0: is not an adjacent pair
    // 1: is an adjacent pair
    // Indices of the table per possible pair of ids:
    //  _____________________________
    // |         |         |         |
    // |  index  |   id1   |   id2   |
    // |_________|_________|_________|
    // |         |         |         |
    // |    0    |    0    |    1    |
    // |    1    |    0    |    2    |
    // |    2    |    1    |    2    |
    // |    3    |    0    |    3    |
    // |    4    |    1    |    3    |
    // |    5    |    2    |    3    |
    // |    6    |    0    |    4    |
    // |    7    |    1    |    4    |
    // |    8    |    2    |    4    |
    // |    9    |    3    |    4    |
    // |   ...   |   ...   |   ...   |
    // |_________|_________|_________|
    //
    // It can be proved that, the index is obtained from the hash
    // function: i = hash(id1, id2) = id1 + id2*(id2 - 1)/2
    // Where id1, id2 are the ids of blobs in question, and
    // id1 < id2.
    long *adj_table;

    // Size of adj_table in worst case scenario.
    unsigned long long adj_table_size = nb_wst*(nb_wst-1)/2;

    // Memory allocation
    frame = (FrameOfBlobs*) malloc(sizeof(FrameOfBlobs));
    blobs = (Blob*) malloc(nb_wst * sizeof(Blob));
    dert_refs = (DertRef*) malloc(size * sizeof(DertRef));
    fill_map = (long*) malloc((size/32 + 1) * sizeof(long));
    //adj_table = (long*) malloc((adj_table_size/32 + 1) * sizeof(long));
    queue = (long*) malloc(qlen * sizeof(long));

    // Fill values
    frame->I = 0;
    frame->G = 0;
    frame->Dy = 0;
    frame->Dx = 0;
    clearbits(fill_map, size);
    //clearbits(adj_table, adj_table_size);

    // Loop through all derts
    for(int i = 0; i < size; i++)
        if(!testbit(fill_map, i)) {  /* ignore filled derts */
            setbit(fill_map, i);    /* set current dert as filled */
            blobs[nblobs].dert_ref = &dert_refs[nfilled];  /* save pointer, length is S */

            double I = 0, G = 0, Dy = 0, Dx = 0, S = 0;
            char sign = g_[i] - ave > 0,
                 fopen = 0;
            unsigned long long id2hash = nblobs*(nblobs - 1) / 2;

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
                S++;

                idmap[j] = nblobs;  /* Set blob's id to map */

                int y = j / width,  /* un-hash coordinate */
                    x = j % width;
                dert_refs[nfilled].x = x;  /* save filled dert position */
                dert_refs[nfilled].y = y;
                nfilled++;

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
                        if(testbit(fill_map, k)) continue;
                        // check if same-signed
                        if(sign == (g_[k] - ave > 0)) {
                            setbit(fill_map, k);    /* set current dert as filled */
                            queue[qend++] = k;  /* append this hash */
                            if(qend >= qlen) qend = 0;
                        }
                        // else assign adjacents
                        else {
                            // unsigned long long adj_hash = idmap[k] + id2hash;
                            // setbit(adj_table, adj_hash);
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
            blobs[nblobs].S = S;
            blobs[nblobs].sign = sign;
            blobs[nblobs].fopen = fopen;
            nblobs++;
        }
    // Deallocate memory
    free(fill_map);
    free(queue);
    //free(adj_table);

    // Return frame of blobs
    realloc(blobs, nblobs * sizeof(Blob));
    frame->blobs_ptr = blobs;
    frame->nblobs = nblobs;

    return *frame;
}


/**
 * Bit array manipulation. These operations assume array elements of
 * type long.
 * Source:
 * http://www.mathcs.emory.edu/~cheung/Courses/255/Syllabus/1-C-intro/bit-array.html
 */

#include <string.h>

#define setbit(A, i) (A[i >> 5] |= (1 << (i & 0x1F)))
#define clearbit(A, i) (A[i >> 5] &= ~(1 << (i & 0x1F)))
#define testbit(A, i) (A[i >> 5] & (1 << (i & 0x1F)))
#define setbits(A, i) memset(A, 0xFF, (n >> 3) + 1)
#define clearbits(A, n) memset(A, 0x00, (n >> 3) + 1)


# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
# *.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
#  Usually these files are written by a python script from a template
#  before PyInstaller builds the exe, so as to inject date/other infos into it.
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/
cover/

# Translations
*.mo
*.pot

# Django stuff:
*.log
local_settings.py
db.sqlite3
db.sqlite3-journal

# Flask stuff:
instance/
.webassets-cache

# Scrapy stuff:
.scrapy

# Sphinx documentation
docs/_build/

# PyBuilder
.pybuilder/
target/

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# pyenv
#   For a library or package, you might want to ignore these files since the code is
#   intended to run in multiple environments; otherwise, check them in:
# .python-version

# pipenv
#   According to pypa/pipenv#598, it is recommended to include Pipfile.lock in version control.
#   However, in case of collaboration, if having platform-specific dependencies or dependencies
#   having no cross-platform support, pipenv may install dependencies that don't work, or not
#   install all needed dependencies.
#Pipfile.lock

# PEP 582; used by e.g. github.com/David-OConnor/pyflow
__pypackages__/

# Celery stuff
celerybeat-schedule
celerybeat.pid

# SageMath parsed files
*.sage.py

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# Pyre type checker
.pyre/

# pytype static type analyzer
.pytype/

# Cython debug symbols
cython_debug/