from Operations import Axes_Rearrange

input_ = [(0, 0, 0, 15), (0, 0, 1, 18),
          (0, 1, 0, 19), (0, 1, 1, 20),
          (1, 0, 0, 15), (1, 0, 1, 18),
          (1, 1, 0, 19), (1, 1, 1, 20)]

print(input_)

sorted_input_ = Axes_Rearrange(input_, axes=(2, 1, 0))

print(sorted_input_)

sorted_input_ = Axes_Rearrange(input_, axes=(1, 2, 0))

print(sorted_input_)