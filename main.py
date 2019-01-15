from time import time
from CAclasses import frame
from CAmiscs import drawFrame

start_time = time()
frame = frame('./images/raccoon_eye.jpg')
drawFrame('./images/raccoon_eye_blobs.bmp', frame)
print time() - start_time