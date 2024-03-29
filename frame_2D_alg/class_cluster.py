"""
Provide a base class for mutable params struct in CogAlg.
"""

from copy import deepcopy
from itertools import count
from dataclasses import astuple, dataclass, field, fields

import weakref


# ----------------------------------------------------------------------------
# Cbase class
@dataclass
class CBase:
    _id: int = field(default_factory=count().__next__, init=False, repr=True)

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
        dataclass(repr=False, eq=False)(cls)

    def __hash__(self):
        return self.id

    @classmethod
    def get_instance(cls, _id: int):
        """Look for instance by its id. Return None if not found."""
        inst = cls.__refs__[_id]()
        if inst is not None and inst.id == _id:
            return inst

    def accumulate(self, **kwargs):
        for key in kwargs:
            setattr(self, key, getattr(self, key) + kwargs[key])

    def unpack(self):
        return astuple(self)

    def copy(self, other):
        for attr, value in other.__dict__.items():
            if attr != '_id' and attr in self.__dict__.keys():  # copy only the available attributes and skip id
                setattr(self, attr, deepcopy(value))

# ----------------------------------------------------------------------------
# CbaseLite class
@dataclass
class CBaseLite:
    def __init_subclass__(cls, **kwargs):
        """Save the need to use dataclass decorator on subclasses."""
        super().__init_subclass__(**kwargs)
        dataclass()(cls)

    def __add__(self, other):
        return self.__class__(*((_value + value) for _value, value in zip(self, other)))

    def __sub__(self, other):
        return self.__class__(*((_value - value) for _value, value in zip(self, other)))

    def __iter__(self):
        return (getattr(self, key) for key in self.__dataclass_fields__)

    def __len__(self):
        return len(fields(self))

    def __getitem__(self, item):
        try:
            return getattr(self, fields(self)[item].name)
        except AttributeError as e:  # 'tuple' object has no attribute 'name'
            return [getattr(self, f.name) for f in fields(self)[item]]

    def __setitem__(self, item, value):
        try:
            setattr(self, fields(self)[item].name, value)
        except AttributeError as e:  # 'tuple' object has no attribute 'name'
            for f, v in zip(fields(self)[item], value):
                setattr(self, f.name, v)

# ----------------------------------------------------------------------------
# functions
def init_param(default):  # initialize param value
    return field(default_factory=lambda: deepcopy(default),
                 repr=False)  # repr=False to avoid recursion in __repr__

if __name__ == "__main__":  # for tests

    # ---- root layer  --------------------------------------------------------
    # using blob as example
    class CBlob(CBase):
        I: int = 0
        Dy: int = 0
        Dx: int = 0
        G: int = 0
        M: int = 0
        Day: int = 0
        Dax: float = 0.0
    # blob derivatives
    class CderBlob(CBase):
        mB: int
        dB: int = 0
        blob: object = None
        _blob: object = None

    class CBBlob(CBlob, CderBlob):
        A: int = 5

    '''---- example  -----------------------------------------------------------

    # root layer
    blob1 = CBlob(I=5, Dy=5, Dx=7, G=5, M=6, Day=4 + 5j, Dax=8 + 9j)
    derBlob1 = CderBlob(mB=5, dB=5)

    # example of value inheritance, b_blob now will having parameter values from blob1 and derBlob1
    # In this example, Dy and Dx are excluded from the inheritance
    b_blob = CBBlob(55)

    print(b_blob)
    '''