import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# Read image in color
source = "./images/sign2.jpg"
img = cv.imread(source)
assert img is not None, "file could not be read, check with os.path.exists()"

# Convert to HSV color space for better color segmentation
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

# Define color ranges in HSV - adjusted to be more specific
# Lower red
lower_red1 = np.array([0, 120, 100])
upper_red1 = np.array([10, 255, 255])
# Upper red
lower_red2 = np.array([160, 120, 100])
upper_red2 = np.array([180, 255, 255])

# Green for traffic signs
lower_green = np.array([40, 120, 100])
upper_green = np.array([80, 255, 255])

# Blue for traffic signs
lower_blue = np.array([100, 150, 100])
upper_blue = np.array([130, 255, 255])

# Yellow
lower_yellow = np.array([20, 120, 100])
upper_yellow = np.array([35, 255, 255])

# Create masks for each color
mask_red1 = cv.inRange(hsv, lower_red1, upper_red1)
mask_red2 = cv.inRange(hsv, lower_red2, upper_red2)
mask_red = cv.bitwise_or(mask_red1, mask_red2)

mask_green = cv.inRange(hsv, lower_green, upper_green)
mask_blue = cv.inRange(hsv, lower_blue, upper_blue)
mask_yellow = cv.inRange(hsv, lower_yellow, upper_yellow)

# Apply morphological operations to remove noise
kernel = np.ones((5, 5), np.uint8)

# Erosion to remove small noise
erode_red = cv.erode(mask_red, kernel, iterations=1)
erode_yellow = cv.erode(mask_yellow, kernel, iterations=1)
erode_green = cv.erode(mask_green, kernel, iterations=1)
erode_blue = cv.erode(mask_blue, kernel, iterations=1)

# Dilation to restore the size of main objects
dilate_red = cv.dilate(erode_red, kernel, iterations=1)
dilate_yellow = cv.dilate(erode_yellow, kernel, iterations=1)
dilate_green = cv.dilate(erode_green, kernel, iterations=1)
dilate_blue = cv.dilate(erode_blue, kernel, iterations=1)

# Apply masks to original image to get only the specific color
result_red = cv.bitwise_and(img, img, mask=dilate_red)
result_green = cv.bitwise_and(img, img, mask=dilate_green)
result_blue = cv.bitwise_and(img, img, mask=dilate_blue)
result_yellow = cv.bitwise_and(img, img, mask=dilate_yellow)

# Find and draw contours for each color
img_with_boxes = img.copy()

# Determine types of signals present
types_signs = []
threshold_area = 5000  # Minimum number of pixels to consider the presence of color


# Updated function to check color presence based on contour analysis
def check_color_presence(mask):
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area > threshold_area:
            # Additional shape analysis for traffic signs
            x, y, w, h = cv.boundingRect(cnt)
            aspect_ratio = float(w) / h
            if 0.5 <= aspect_ratio <= 2.0:
                return True
    return False


# Check presence of each color
if check_color_presence(dilate_red):
    types_signs.append("Restrictiva (Rojo)")
if check_color_presence(dilate_yellow):
    types_signs.append("Preventiva (Amarillo)")
if check_color_presence(dilate_blue):
    types_signs.append("Servicios (Azul)")
if check_color_presence(dilate_green):
    types_signs.append("Destino (Verde)")

# Prints the classification results.
print("\nClasificación de señales detectadas:")
if types_signs:
    for type in types_signs:
        print("- " + type)
else:
    print("No se detectaron señales de tráfico.")

# Show original image and results
plt.figure(figsize=(15, 10))
plt.subplot(2, 3, 1), plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.title("Original"), plt.xticks([]), plt.yticks([])

plt.subplot(2, 3, 2), plt.imshow(cv.cvtColor(result_red, cv.COLOR_BGR2RGB))
plt.title("Red"), plt.xticks([]), plt.yticks([])

plt.subplot(2, 3, 3), plt.imshow(cv.cvtColor(result_green, cv.COLOR_BGR2RGB))
plt.title("Green"), plt.xticks([]), plt.yticks([])

plt.subplot(2, 3, 4), plt.imshow(cv.cvtColor(result_blue, cv.COLOR_BGR2RGB))
plt.title("Blue"), plt.xticks([]), plt.yticks([])

plt.subplot(2, 3, 5), plt.imshow(cv.cvtColor(result_yellow, cv.COLOR_BGR2RGB))
plt.title("Yellow"), plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.show()
