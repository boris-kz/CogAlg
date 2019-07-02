"""
Derts
--------
Recursively compare gradient magnitude.

AngleDerts
--------
Compare angle of gradient.

BlobsClusters
-------------
Form blobs from Derts or AngleDerts.
"""

from comparison import SimpleDerts
from partition import SimpleBlob
from partition import SimpleBlobsCluster

# -----------------------------------------------------------------------------
# Derts Class

class Derts(SimpleDerts):

    def __init__(self, input, mask):
        self.mask = mask
        SimpleDerts.__init__(self, input)

    def


# ----------------------------------------------------------------------
# -----------------------------------------------------------------------------