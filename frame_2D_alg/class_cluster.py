"""
Provide a base class for cluster objects in CogAlg.
"""

from copy import deepcopy
from itertools import count
from dataclasses import dataclass, field, astuple

import weakref

# ----------------------------------------------------------------------------
# ClusterStructure class
@dataclass
class ClusterStructure:
    _id: int = field(default_factory=count().__next__, init=False, repr=False)

    @property
    def id(self):
        return self._id

    __refs__ = []

    def __post_init__(self):
        """Save a weak reference to instance inside the class object."""
        self.__refs__.append(weakref.ref(self))

    def __init_subclass__(cls, **kwargs):
        """Save the need to use dataclass decorator on subclasses."""
        super().__init_subclass__(**kwargs)
        dataclass(repr=False)(cls)

    @classmethod
    def get_instance(cls, id : int):
        """Look for instance by its id. Return None if not found."""
        inst = cls.__refs__[id]()
        if inst is not None and inst.id == id:
            return inst

    def accumulate(self, **kwargs):
        for field in kwargs:
            setattr(self, field, getattr(self, field) + kwargs[field])
    def unpack(self):
        return astuple(self)

# ----------------------------------------------------------------------------
# functions
def init_param(default):  # initialize param value
    return field(default_factory=lambda: deepcopy(default),
                 repr=False) # repr=False to avoid recursion in __repr__

if __name__ == "__main__":  # for tests


    # ---- root layer  --------------------------------------------------------
    # using blob as example
    class CBlob(ClusterStructure):
        I : int = 0
        Dy : int = 0
        Dx : int = 0
        G : int = 0
        M : int = 0
        Day : int = 0
        Dax : float = 0.0

    # blob derivatives
    class CderBlob(ClusterStructure):
        mB : int
        dB : int = 0
        blob : object = None
        _blob : object = None

    class CBblob(CBlob, CderBlob):
        A : int = 5

    # ---- example  -----------------------------------------------------------

    # root layer
    blob1 = CBlob(I=5, Dy=5, Dx=7, G=5, M=6, Day=4 + 5j, Dax=8 + 9j)
    derBlob1 = CderBlob(mB=5, dB=5)

    # example of value inheritance, bblob now will having parameter values from blob1 and derBlob1
    # In this example, Dy and Dx are excluded from the inheritance
    bblob = CBblob(55)

    print(bblob)