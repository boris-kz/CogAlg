"""
Provide a base class for cluster objects in CogAlg.
Features:
- Unique instance ids per class.
- Instances are retrievable by ids via class.
- Reduced memory usage compared to using dict.
- Methods generated via string templates so overheads caused by
differences in interfaces are mostly eliminated.
- Can be extended/modified further to support main implementation better.
"""

import weakref
from numbers import Number
from inspect import isclass
import numpy as np
from copy import deepcopy

NoneType = type(None)

# ----------------------------------------------------------------------------
# Template for class method generation
_methods_template = '''
@property
def id(self):
    return self._id
    
def pack(self{pack_args}):
    """Pack all fields/params back into {typename}."""
    {pack_assignments}
    
def unpack(self):
    """Unpack all fields/params back into the cluster."""
    return ({param_vals})
def accumulate(self, **kwargs):
    """Add a number to specified numerical fields/params."""
    {accumulations}
def __contains__(self, item):
    return (item in {params})
def __delattr__(self, item):
    raise AttributeError("cannot delete attribute from "
                         "'{typename}' object")
def __repr__(self):
    return "{typename}({repr_fmt})" % ({numeric_param_vals})
'''

# ----------------------------------------------------------------------------
# MetaCluster meta-class
class MetaCluster(type):
    """
    Serve as a factory for creating new cluster classes.
    """
    def __new__(mcs, typename, bases, attrs):  # called right before a new class is created
        # get fields/params and numeric params
        replace = attrs.get('replace', {})

        # inherit params
        new_bases = []
        for base in bases:
            if issubclass(base, ClusterStructure):
                new_bases.append(base)
                for param in base.numeric_params:
                    if param not in attrs:  # prevents duplication of base params
                        # not all inherited params are Cdm
                        if param in replace:
                            new_param, new_type = replace[param]
                            if new_param is not None:
                                attrs[new_param] = new_type
                        else:
                            attrs[param] = getattr(base,param+'_type') # if the param is not replaced, it will following type of base param
            else:
                print(f"Warning: {base} is not a subclass of {ClusterStructure}")

        bases = tuple(new_bases)   # remove

        if len(bases)>1:
            bases=(bases[0],)

        # only ignore param names start with double underscore
        params = tuple(attr for attr in attrs
                       if not attr.startswith('__') and
                       isclass(attrs[attr]))

        numeric_params = tuple(param for param in params
                               if (issubclass(attrs[param], Number)) and
                               not (issubclass(attrs[param], bool))) # avoid accumulate bool, which is flag

        list_params = tuple(param for param in params
                               if (issubclass(attrs[param], list)))

        dict_params = tuple(param for param in params
                               if (issubclass(attrs[param], dict)))

        # Fill in the template
        methods_definitions = _methods_template.format(
            typename=typename,
            params=str(params),
            param_vals=', '.join(f'self.{param}'
                                 for param in params),
            numeric_param_vals=', '.join(f'self.{param}'
                                         for param in numeric_params),
            list_param_vals=', '.join(f'self.{param}'
                                         for param in list_params),
            dict_param_vals=', '.join(f'self.{param}'
                                         for param in dict_params),
            pack_args=', '.join(param for param in ('', *params)),
            pack_assignments='; '.join(f'self.{param} = {param}'
                                  for param in params)
                             if params else 'pass',
            accumulations='; '.join(f"self.{param} += "
                                    f"kwargs.get('{param}', 0)"
                                    for param in numeric_params)
                          if params else 'pass',
            repr_fmt=', '.join(f'{param}=%r' for param in numeric_params),
        )
        # Generate methods
        namespace = dict(print=print)
        exec(methods_definitions, namespace)
        # Replace irrelevant names
        namespace.pop('__builtins__')
        namespace.pop('print')

        # Update to attrs
        attrs.update(namespace)

        # Save default types for fields/params
        for param in params:
            attrs[param + '_type'] = attrs.pop(param)
        # attrs['params'] = params
        attrs['numeric_params'] = numeric_params
        attrs['list_params'] = list_params
        attrs['dict_params'] = dict_params

        # Add fields/params and other instance attributes
        attrs['__slots__'] = (('_id', 'hid', *params, '__weakref__')
                              if not bases else ('_id', 'hid', *params))

        # Register the new class
        cls = super().__new__(mcs, typename, bases, attrs)

        # Create container for references to instances
        cls._instances = []

        return cls

    def __call__(cls, *args, **kwargs):  # call right before a new instance is created
        # register new instance
        instance = super().__call__(*args, **kwargs)

        # initialize fields/params
        for param in cls.__slots__[2:]:  # Exclude _id and __weakref__
            setattr(instance, param,
                    kwargs.get(param,
                               getattr(cls, param + '_type')()))

        # set inherited params
        if kwargs.get('inherit') is not None:

            excluded = []
            if kwargs.get('excluded') is not None:
                excluded = kwargs.get('excluded')

            for inherit_instance in kwargs.get('inherit'):
                for param in cls.numeric_params: # inherit numeric params
                    if hasattr(inherit_instance,param) and (param not in excluded):
                        setattr(instance, param, getattr(inherit_instance, param))

                for param in cls.list_params: # inherit list params
                    if hasattr(inherit_instance,param) and (param not in excluded):
                        list_param = getattr(inherit_instance, param)
                        if len(list_param)>0: # not empty list
                            setattr(instance, param, list_param )

                for param in cls.dict_params: # inherit dict params
                    if hasattr(inherit_instance,param) and (param not in excluded):
                        setattr(instance, param, getattr(inherit_instance, param))

                for param in cls.dict_params: # inherit dict params
                    if hasattr(inherit_instance,param) and (param not in excluded):
                        setattr(instance, param, getattr(inherit_instance, param))

        # Set id
        instance._id = len(cls._instances)
        # Create ref
        cls._instances.append(weakref.ref(instance))
        # no default higher cluster id, set to None
        instance.hid = None  # higher cluster's id

        return instance

    # original
    '''
    def __call__(cls, *args, **kwargs):  # call right before a new instance is created
        # register new instance
        instance = super().__call__(*args, **kwargs)
        # initialize fields/params
        for param in cls.__slots__[2:]:  # Exclude _id and __weakref__
            setattr(instance, param,
                    kwargs.get(param,
                               getattr(cls, param + '_type')()))
        # Set id
        instance._id = len(cls._instances)
        # Create ref
        cls._instances.append(weakref.ref(instance))
        # no default higher cluster id, set to None
        instance.hid = None  # higher cluster's id
        return instance
    '''

    def get_instance(cls, cluster_id):
        try:
            return cls._instances[cluster_id]()
        except IndexError:
            return None

    @property
    def instance_cnt(cls):
        return len(cls._instances)


# ----------------------------------------------------------------------------
# ClusterStructure class
class ClusterStructure(metaclass=MetaCluster):
    """
    Class for cluster objects in CogAlg.
    Each time a new instance is created, four things are done:
    - Set initialize field/param.
    - Set id.
    - Save a weak reference of instance inside the class object.
    (meaning that if there's no other references to instance,
    it will be garbage collected, weakref to it will return None
    afterwards)
    - Set higher cluster id to None (no higher cluster structure yet)
    Examples
    --------
    >>> from class_cluster import ClusterStructure
    >>> class CP(ClusterStructure):
    >>>     L = int  # field/param name and default type
    >>>     I = int
    >>>
    >>> P1 = CP(L=1, I=5) # initialized with values
    >>> print(P1)
    CP(L=1, I=5)
    >>> P2 = CP()  # default initialization
    >>> print(P2)
    CP(L=0, I=0)
    >>> print(P1.id, P2.id)  # instance's ids
    0 1
    >>> # look for object by instance's ids
    >>> print(CP.get_instance(0), CP.get_instance(1))
    CP(L=1, I=5) CP(L=0, I=0)
    >>> P2.L += 1; P2.I += 10  # assignment, fields are mutable
    >>> print(P2)
    CP(L=1, I=10)
    >>> # Accumulate using accumulate()
    >>> P1.accumulate(L=1, I=2)
    >>> print(P1)
    CP(L=2, I=7)
    >>> # ... or accum_from()
    >>> P2.accum_from(P1)
    >>> print(P2)
    CP(L=3, I=17)
    >>> # field/param types are not constrained, so be careful!
    >>> P2.L = 'something'
    >>> print(P2)
    CP(L='something', I=10)
    """

    def __init__(self, **kwargs):
        pass

    def accum_from(self, other, excluded=()):
        """Accumulate params from another structure."""

        # accumulate base params
        for param in self.numeric_params:
            if (param not in excluded) and (param in other.numeric_params):
                p = getattr(self,param)
                _p = getattr(other,param)
                setattr(self, param, p+_p)

        # accumulate layers 1 and above
        for layer_num in self.dict_params:
            if (layer_num in other.dict_params):

                layer = getattr(self,layer_num)   # self layer params
                _layer = getattr(other,layer_num) # other layer params

                if len(layer) == len(_layer):  # both layers have the same params
                    for i, ((param_name,dert), (_param_name,_dert)) in enumerate(zip(layer.items(), _layer.items())):
                        # accumulate _dert into dert
                        if not isinstance(dert, Cdert) and isinstance(_dert, Cdert):  # if base param < ave_comp?
                            layer[param_name] = _dert
                        elif isinstance(dert, Cdert) and isinstance(_dert, Cdert):
                            dert.p += _dert.p
                            dert.d += _dert.d
                            dert.m += _dert.m
                elif len(_layer)>0: # _layer is not empty but layer is empty
                    setattr(self,layer_num,_layer.copy())


class Cdert(Number): # Ppd and Ppdm might not relevant now, so remove it
    __slots__ = ('i','p','d','m')

    def __init__(self, i=0, p=0, d=0, m=0):
        self.i, self.p, self.d, self.m = i, p, d, m

    def __add__(self, other):
        return Cdert(self.i, self.p + other.p, self.d + other.d, self.m + other.m)

    def copy(self):
        return deepcopy(self)

    def __repr__(self):  # representation of object
        if isinstance(self.i, Cdert) or isinstance(self.p, Cdert) or isinstance(self.d, Cdert) or isinstance(self.m, Cdert):
            return "Cdert(i=Cdert, p=Cdert, d=Cdert, m=Cdert)"
        else:
            return "Cdert(i={}, p={}, d={}, m={})".format(self.i, self.p, self.d, self.m)


def comp_param(_param, param, param_name, ave):

    if isinstance(param,list):  # vector
        sin, cos = param[0], param[1]
        _sin, _cos = _param[0], _param[1]
        # difference of dy and dx
        sin_da = (cos * _sin) - (sin * _cos)  # sin(α - β) = sin α cos β - cos α sin β
        cos_da= (cos * _cos) + (sin * _sin)   # cos(α - β) = cos α cos β + sin α sin β
        da = np.arctan2(sin_da, cos_da)
        ma = ave - abs(da)  # indirect match
        dert = Cdert(i=param, p=param+_param, d=da, m=ma)  # d=None? p=param+_param will sum lists?
    else:  # numeric
        if param_name == 'L':
            d = _param/param
        else:
            d = param - _param    # difference
        if param_name == 'I':
            m = ave - abs(d)  # indirect match
        else:
            m = min(param,_param) - abs(d)/2 - ave  # direct match
        dert = Cdert(i=param, p=param+_param, d=d, m=m)

    return dert


def copy(self):
    return deepcopy(self)

def merge(self, other, self_instance_name, self_instance, excluded = ()):
    # accumulate numeric params
    self.accum_from(other,excluded=excluded)

    for list_param in self.list_params: # list params, such as pdert_, P_, sublayers, subDerts
        if hasattr(other, list_param):

            self_list = getattr(self,list_param)   # get self list param, such as P_, pdert_ and etc
            other_list = getattr(other,list_param) # get other list param, such as P_, pdert_ and et

            for other_param in other_list: # append each list param to self, for example other.P_ to self.P_
                self_list.append(other_param)

                # update reference, may add in more class later
                # for example update other P.Pp to self Pp
                if hasattr(other_param, self_instance_name) and isinstance(self, self_instance):
                    setattr(other_param, self_instance_name, self)


if __name__ == "__main__":  # for tests


    # ---- root layer  --------------------------------------------------------
    # using blob as example
    class CBlob(ClusterStructure):
        I = int
        Dy = int
        Dx = int
        G = int
        M = int
        Day = int
        Dax = int

    # blob derivatives
    class CderBlob(ClusterStructure):
        mB = int
        dB = int
        blob = object
        _blob = object

    class CBblob(CBlob, CderBlob):
        pass

    # ---- example  -----------------------------------------------------------

    # root layer
    blob1 = CBlob(I=5, Dy=5, Dx=7, G=5, M=6, Day=4 + 5j, Dax=8 + 9j)
    derBlob1 = CderBlob(mB=5, dB=5)

    # example of value inheritance, bblob now will having parameter values from blob1 and derBlob1
    # In this example, Dy and Dx are excluded from the inheritance
    bblob = CBblob(inherit=[blob1, derBlob1], excluded=['Dy','Dx'])

    print(bblob)