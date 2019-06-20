"""For testing cablobs."""

import cablobs

blobs_cluster = cablobs.CABlobsCluster("./../images/raccoon.jpg", k2x2=0)

# from utils import draw
#
# for i, blob in enumerate(blobs_cluster.blobs):
#     if blobs_cluster.G(i) > 10000:
#         draw("./../DEBUG/blob" + str(blob.slice), blob.map * 255)