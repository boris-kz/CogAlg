from Operations import Compare

input_ = [(0, 0, 15), (0, 1, 18), (0, 2, 5),
          (1, 0, 19), (1, 1, 20), (1, 2, 18),
          (2, 0, 25), (2, 1, 35), (2, 2, 32)]

output_ = Compare(input_, offset=(1, -1))

print(output_)