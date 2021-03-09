import cv2 
import numpy as np

def setup_detector():
    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 0
    params.maxThreshold = 25

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 10
    params.maxArea = 1000


    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.03

    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.9

    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.0001

    # Create a detector with the parameters
    ver = (cv2.__version__).split('.')
    if int(ver[0]) < 3 :
            
        detector = cv2.SimpleBlobDetector(params)
    else : 
        
        detector = cv2.SimpleBlobDetector_create(params)

    return detector

# Read image
im = cv2.imread("light.jpg", cv2.IMREAD_GRAYSCALE)

detector = setup_detector()

# Detect blobs.
keypoints = detector.detect(im)

# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Show keypoints
cv2.imshow("Keypoints", im_with_keypoints)
cv2.waitKey(0)

img = cv2.imread("light.jpg")
img = np.float32(img)
# ..la faccio in grayscale..
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = np.uint8(gray)
# .. e prendo i kaypoints grazie al (simpleblob) detector
keypoints = detector.detect(gray)
keypoints = cv2.KeyPoint_convert(keypoints)

# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


# Show keypoints
cv2.imshow("Keypoints", im_with_keypoints)
cv2.waitKey(0)