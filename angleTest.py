from math import pi, sin, cos
import matplotlib.pyplot as plt

sins = []
cosins = []

slice = pi / 128

for i in range(-128, 128):
    a = i * slice
    sins.append(sin(a))
    cosins.append(cos(a))

plt.plot(sins, color='red', label='sin')
plt.plot(cosins, color='blue', label='cos')
plt.show()

offset = 64
diff = []
for i in range(256):
    diff.append(sins[i] - cosins[i - offset])


plt.plot(diff, color='black', label='difference between sin(a) and cos(a - %d)' % (offset))
plt.show()

print(cosins)