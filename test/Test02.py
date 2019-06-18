import matplotlib.pyplot as plt
from caderts import CADerts

Derts = CADerts('./../images/raccoon_eye.jpg')

plt.imshow(Derts.dy, cmap='Greys')
plt.show()