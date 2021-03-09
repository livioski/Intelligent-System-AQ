'''
import cv2
import numpy as np

filename = 'roof.jpg'
img = cv2.imread(filename)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

gray = np.float32(gray)
dst = cv2.cornerHarris(gray,2,3,0.06)

#result is dilated for marking the corners, not important
dst = cv2.dilate(dst,None)

# Threshold for an optimal value, it may vary depending on the image.
img[dst>0.01*dst.max()]=[0,0,255]

cv2.imshow('dst',img)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()
'''

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


 
BLURRING = 0
NUM_OF_CORNERS = 100
QUALITY = 0.008
DISTANCE = 100

fig = plt.figure()

img = cv.imread('roof.jpg')

plt.subplot(2, 2, 1)
plt.imshow(img)

blur = cv.blur(img,(10,10))
plt.subplot(2, 2, 3)
plt.imshow(blur)

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
corners = cv.goodFeaturesToTrack(gray,NUM_OF_CORNERS,QUALITY,DISTANCE)
corners = np.int0(corners)
for i in corners:
    x,y = i.ravel()
    cv.circle(img,(x,y),3,255,-1)

plt.subplot(2, 2, 2)
plt.imshow(img)

gray = cv.cvtColor(blur,cv.COLOR_BGR2GRAY)
corners = cv.goodFeaturesToTrack(gray,NUM_OF_CORNERS,QUALITY,DISTANCE)
corners = np.int0(corners)
for i in corners:
    x,y = i.ravel()
    cv.circle(blur,(x,y),3,255,-1)

plt.subplot(2, 2, 4)
plt.imshow(blur)


plt.show()




