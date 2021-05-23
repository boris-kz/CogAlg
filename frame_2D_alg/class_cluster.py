'''
Provide a base class for cluster objects in CogAlg.
Features:
- Unique instance ids per class.
- Instances are retrievable by ids via class.
- Reduced memory usage compared to using dict.
- Methods generated via string templates so overheads caused by
differences in interfaces are mostly eliminated.
- Can be extended/modified further to support main implementation better.
'''

import weakref
from numbers import Number
from inspect import isclass
from cmath import phase

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

        # inherit params
        for base in bases:
            if issubclass(base, ClusterStructure):
                for key in base.__dict__:
                    if key[-5:] == "_type":
                        param = key[:-5]
                        if param not in attrs:
                            attrs[param] = list

        # only ignore param names start with double underscore
        params = tuple(attr for attr in attrs
                       if not attr.startswith('__') and
                       isclass(attrs[attr]))

        numeric_params = tuple(param for param in params
                               if (issubclass(attrs[param], Number)) and
                               not (issubclass(attrs[param], bool))) # avoid accumulate bool, which is flag

        # Fill in the template
        methods_definitions = _methods_template.format(
            typename=typename,
            params=str(params),
            param_vals=', '.join(f'self.{param}'
                                 for param in params),
            numeric_param_vals=', '.join(f'self.{param}'
                                         for param in numeric_params),
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

        # Add fields/params and other instance attributes
        attrs['__slots__'] = (('_id', 'hid', *params, '__weakref__')
                              if not bases else ('_id', 'hid', *params))

        # Register the new class
        cls = super().__new__(mcs, typename, bases, attrs)

        # Create container for references to instances
        cls._instances = []

        return cls

    def __call__(cls, *args, **kwargs):  # call right before a new instace is created
        # register new instace
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
    Class for cluster objects in 2D implementation of CogAlg.
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
        self.accumulate(**{param: getattr(other, p, 0)
                           for param in self.numeric_params
                           if param not in excluded})

    def difference(self, other, excluded=()):
        return {param:(getattr(self, param) - getattr(other, param))
                for param in self.numeric_params if param not in excluded}

    def min_match(self, other, excluded=()):
        return {param: min(getattr(self, param), getattr(other, param))
                for param in self.numeric_params if param not in excluded}

    def abs_min_match(self, other, excluded=()):
        return {param: min(abs(getattr(self, param)), abs(getattr(other, param)))
                for param in self.numeric_params if param not in excluded}


if __name__ == "__main__":  # for debugging
    from sys import getsizeof as size
    size(ClusterStructure)
    def min_match(self, other, excluded=()):

        return {param: min(getattr(self, param), getattr(other, param))
                for param in self.numeric_params if param not in excluded}

    def abs_min_match(self, other, excluded=()):

        return {param: min(abs(getattr(self, param)), abs(getattr(other, param)))
                for param in self.numeric_params if param not in excluded}
'''
    def min_match_da(self, other):
        results = {}
        for param in self.numeric_params:
            _e = getattr(other, param)
            e = getattr(self, param)
            if isinstance(e, complex):  # match between angle diffs
                results[param] = _e if phase(_e) < phase(e)  else e
            else:
                results[param] = min(_e, e)
        return results
'''

if __name__ == "__main__":  # for debugging
    from sys import getsizeof as size
    size(ClusterStructure)