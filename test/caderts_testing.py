"""For testing caderts."""

import caderts

from utils import draw

ave = 15

derts = caderts.from_image("./../images/raccoon.jpg", k2x2=0)

draw("./../DEBUG/g", (derts.g > 15) * 255)