from math import hypot, sin, cos, atan2, degrees

class angle(int):
    max_val = 255
    middle_val = 128
    bit_len = 8
    def __new__(cls, y, x):
        " compress the angle vector "
        coef = 128.0 / hypot(y, x)
        y = int(y * coef)
        x = int(x * coef)
        if y < 0: y = bit_neg(-y, cls.max_val)
        if x < 0: x = bit_neg(-x, cls.max_val)
        return int.__new__(cls, (y << cls.bit_len) | x)

    def extract(self):
        a = self.__hash__()
        y, x = a >> self.bit_len, a & self.max_val
        return y, x

    def decoded(self):
        y, x = self.extract()
        if not y < self.middle_val: y -= self.max_val + 1
        if not x < self.middle_val: x -= self.max_val + 1
        return y, x

    def __add__(self, a2):
        y1, x1 = self.decoded()
        y2, x2 = a2.decoded()
        return angle_sum(y1 + y2, x1 + x2)

    def __radd__(self, a2):
        return self + a2

    def __sub__(self, other):
        y1, x1 = self.decoded()
        y2, x2 = a2.decoded()
        x = (y1 * y2 + x1 * x2) >> (self.bit_len * 2)
        y = int(((1 << (self.bit_len * 2)) - x * x) ** 0.5)
        return angle_sum(y, x)

    def in_radiant(self):
        return atan2(*self.decoded())

    def in_degree(self):
        return degrees(atan2(*self.decoded()))

class angle_sum(angle):
    max_val = 16777215
    middle_val = 8388608
    bit_len = 24
    def __new__(cls, y, x):
        " compress the angle vector "
        if y < 0: y = bit_neg(-y, cls.max_val)
        if x < 0: x = bit_neg(-x, cls.max_val)
        return int.__new__(cls, (y << cls.bit_len) | x)

    def to_angle(self):
        return angle(*self.decoded())

def bit_neg(x, max_val):
    return (~x + 1) & max_val

# Testing
a1 = angle(0, -1)
r = a1.in_radiant()
print(r, sin(r), cos(r))
print(a1.in_degree())

a2 = angle(5, 5)
r = a2.in_radiant()
print(r, sin(r), cos(r))
print(a2.in_degree())

d = a2 - a1

print