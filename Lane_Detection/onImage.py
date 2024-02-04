import os
import cv2
import numpy as np

folder = "test_images"
img_dir = os.listdir(folder)

image = cv2.imread(folder + "/" + img_dir[3])

gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

blur = cv2.blur(gray, ksize=(5, 5))
# cv2.imshow("blur", blur)
edges = cv2.Canny(blur, threshold1=100, threshold2=150)
# cv2.imshow("edges", edges)

# create mask
mask = np.zeros_like(image)
ignore_mask_color = 255
# Defining a four-sided polygon region to mask, using full length of bottom of image and \top vertices are defined
# to capture lanes in distance
vertices = np.array([[(0, image.shape[0]), (390, 350), (580, 350), (image.shape[1], image.shape[0])]], dtype=np.int32)


def region_of_interest(img, vertices):
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


masked_image = region_of_interest(edges, vertices)

# DEFINE PARAMETERS FOR HOUGH TRANSFORM
rho = 5  # distance resolution in pixels of the Hough grid
theta = np.pi / 180  # angular resolution in radians of the Hough grid
threshold = 20  # minimum number of votes (intersections in Hough grid cell)
min_line_length = 50  # minimum number of pixels making up a line
max_line_gap = 100  # maximum gap in pixels between connectable line segments

lines = cv2.HoughLinesP(masked_image, rho, theta, threshold, np.array([]), minLineLength=min_line_length,
                        maxLineGap=max_line_gap)
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

cv2.imshow("final", image)
cv2.waitKey(0)
