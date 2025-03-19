import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# Read image in color
source = "Sign4.jpg"
img = cv.imread(source)
assert img is not None, "file could not be read, check with os.path.exists()"

# Convert to HSV color space for better color segmentation
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

# Define color ranges in HSV (double ranges for better detection)
# Lower red
lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])
# Upper red
lower_red2 = np.array([160, 100, 100])
upper_red2 = np.array([180, 255, 255])

# Lower green
lower_green1 = np.array([35, 100, 100])
upper_green1 = np.array([85, 255, 255])
# Upper green
lower_green2 = np.array([85, 50, 100])
upper_green2 = np.array([95, 255, 255])

# Lower blue
lower_blue1 = np.array([85, 100, 100])
upper_blue1 = np.array([130, 255, 255])
# Upper blue
lower_blue2 = np.array([130, 50, 100])
upper_blue2 = np.array([140, 255, 255])

# Lower yellow
lower_yellow1 = np.array([20, 100, 100])
upper_yellow1 = np.array([35, 255, 255])
# Upper yellow
lower_yellow2 = np.array([35, 50, 100])
upper_yellow2 = np.array([45, 255, 255])

# Create masks for each color with double ranges
mask_red1 = cv.inRange(hsv, lower_red1, upper_red1)
mask_red2 = cv.inRange(hsv, lower_red2, upper_red2)
mask_red = cv.bitwise_or(mask_red1, mask_red2)

mask_green1 = cv.inRange(hsv, lower_green1, upper_green1)
mask_green2 = cv.inRange(hsv, lower_green2, upper_green2)
mask_green = cv.bitwise_or(mask_green1, mask_green2)

mask_blue1 = cv.inRange(hsv, lower_blue1, upper_blue1)
mask_blue2 = cv.inRange(hsv, lower_blue2, upper_blue2)
mask_blue = cv.bitwise_or(mask_blue1, mask_blue2)

mask_yellow1 = cv.inRange(hsv, lower_yellow1, upper_yellow1)
mask_yellow2 = cv.inRange(hsv, lower_yellow2, upper_yellow2)
mask_yellow = cv.bitwise_or(mask_yellow1, mask_yellow2)

# Apply erosion to remove noise
erode_red = cv.erode(mask_red, None, iterations=2)
erode_yellow = cv.erode(mask_yellow, None, iterations=2)
erode_green = cv.erode(mask_green, None, iterations=2)
erode_blue = cv.erode(mask_blue, None, iterations=2)

# Apply masks to original image to get only the specific color
result_red = cv.bitwise_and(img, img, mask=erode_red)
result_green = cv.bitwise_and(img, img, mask=erode_green)
result_blue = cv.bitwise_and(img, img, mask=erode_blue)
result_yellow = cv.bitwise_and(img, img, mask=erode_yellow)

# Find and draw contours for each color
img_with_boxes = img.copy()


def draw_boxes(mask, color):
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area > 2500:  # Filter small contours
            x, y, w, h = cv.boundingRect(cnt)
            cv.rectangle(img_with_boxes, (x, y), (x + w, y + h), color, 5)


# Draw boxes for each color
draw_boxes(erode_red, (0, 0, 255))  # Red
draw_boxes(erode_blue, (255, 0, 0))  # Blue
draw_boxes(erode_green, (0, 255, 0))  # Green
draw_boxes(erode_yellow, (0, 255, 255))  # Yellow

# Determine types of signals present
types_signs = []
threshold_area = 5000  # Minimum number of pixels to consider the presence of color


#  Checks if there is enough presence of a color in the image.
def check_color_presence(result_img):
    gray = cv.cvtColor(result_img, cv.COLOR_BGR2GRAY)
    return np.count_nonzero(gray) > threshold_area


# Check presence of each color
if check_color_presence(result_red):
    types_signs.append("Restrictiva (Rojo)")
if check_color_presence(result_yellow):
    types_signs.append("Preventiva (Amarillo)")
if check_color_presence(result_blue):
    types_signs.append("Servicios (Azul)")
if check_color_presence(result_green):
    types_signs.append("Destino (Verde)")

# Prints the classification results.
print("\nClasificación de señales detectadas:")
for type in types_signs:
    print("- " + type)

# Show original image
plt.subplot(2, 3, 1), plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.title("Original"), plt.xticks([]), plt.yticks([])

# Show results for each color
plt.subplot(2, 3, 2), plt.imshow(cv.cvtColor(result_red, cv.COLOR_BGR2RGB))
plt.title("Red"), plt.xticks([]), plt.yticks([])

plt.subplot(2, 3, 3), plt.imshow(cv.cvtColor(result_green, cv.COLOR_BGR2RGB))
plt.title("Green"), plt.xticks([]), plt.yticks([])

plt.subplot(2, 3, 4), plt.imshow(cv.cvtColor(result_blue, cv.COLOR_BGR2RGB))
plt.title("Blue"), plt.xticks([]), plt.yticks([])

plt.subplot(2, 3, 5), plt.imshow(cv.cvtColor(result_yellow, cv.COLOR_BGR2RGB))
plt.title("Yellow"), plt.xticks([]), plt.yticks([])

plt.show()
