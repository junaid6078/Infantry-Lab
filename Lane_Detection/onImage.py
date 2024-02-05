import os
import cv2
import numpy as np

folder = "test_images"
img_dir = os.listdir(folder)

image = cv2.imread(folder + "/" + img_dir[3])

gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

blur = cv2.blur(gray, ksize=(5, 5))
# cv2.imshow("blur", blur)
edges = cv2.Canny(blur, threshold1=100, threshold2=150, apertureSize=3, L2gradient=True)
# cv2.imshow("edges", edges)

# create mask

# Defining a four-sided polygon region to mask, using full length of bottom of image and \top vertices are defined
# to capture lanes in distance
vertices = np.array([[(0, image.shape[0]), (350, 350), (580, 350), (image.shape[1], image.shape[0])]], dtype=np.int32)


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
cv2.imshow("Masked Image", masked_image)

# DEFINE PARAMETERS FOR HOUGH TRANSFORM
rho = 5  # distance resolution in pixels of the Hough grid
theta = np.pi / 180  # angular resolution in radians of the Hough grid
threshold = 20  # minimum number of votes (intersections in Hough grid cell)
min_line_length = 50  # minimum number of pixels making up a line
max_line_gap = 100  # maximum gap in pixels between connectable line segments

lines = cv2.HoughLinesP(masked_image, rho, theta, threshold, np.array([]), minLineLength=min_line_length,
                        maxLineGap=max_line_gap)
# for line in lines:
#     x1, y1, x2, y2 = line[0]
#     cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

x_left_lane = []
y_left_lane = []
x_right_lane = []
y_right_lane = []
ytop = int(350)  # need y coordinates of the top and bottom of left and right lane
ybtm = int(540)  # to calculate x values once a line is found

for line in lines:
    for x1, y1, x2, y2 in line:
        slope = float(((y2 - y1) / (x2 - x1)))
        if (slope > 0.3):  # if the line slope is greater than 0, it is the left lane
            x_left_lane.append(x1)
            x_left_lane.append(x2)
            y_left_lane.append(y1)
            y_left_lane.append(y2)

        if (slope < -0.3):  # if the line slope is less than 0, it is the right lane
            x_right_lane.append(x1)
            x_right_lane.append(x2)
            y_right_lane.append(y1)
            y_right_lane.append(y2)

    # only execute if there are points found that meet criteria
if (x_left_lane != []) & (x_right_lane != []) & (y_left_lane != []) & (y_right_lane != []):
    left_line_coeffs = (np.polyfit(x_left_lane, y_left_lane, 1))
    left_xtop = int((ytop - left_line_coeffs[1]) / left_line_coeffs[0])
    left_xbtm = int((ybtm - left_line_coeffs[1]) / left_line_coeffs[0])

    cv2.line(image, (left_xtop, ytop), (left_xbtm, ybtm), [255, 0, 0], 5)

    right_line_coeffs = np.polyfit(x_right_lane, y_right_lane, 1)
    right_xtop = int((ytop - right_line_coeffs[1]) / right_line_coeffs[0])
    right_xbtm = int((ybtm - right_line_coeffs[1]) / right_line_coeffs[0])

    cv2.line(image, (right_xtop, ytop), (right_xbtm, ybtm), [255, 0, 0], 5)

cv2.imshow("final", image)
cv2.waitKey(0)
