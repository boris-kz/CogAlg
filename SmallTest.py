from collections import namedtuple

nt = namedtuple('nt', 'a b c')

a = nt(1, 2, 3)

print(type(a) == nt)