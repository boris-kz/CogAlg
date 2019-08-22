"""
A simple diagram generator.
"""

import matplotlib.pyplot as plt

class Presenter:
    def __init__(self, fig):
        self._fig = fig

    def display(self):
        self._fig.show()

    def to_file(self, path):
        self._fig.savefig(path)

class DiagramGenerator(Presenter):
    def __init__(self, *func_names):
        pass
        Presenter.__init__(self, fig)

# ----------------------------------------------------------------------
# -----------------------------------------------------------------------------