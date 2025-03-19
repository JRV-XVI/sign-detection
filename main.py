import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# Read image in color
img = cv.imread("Sign6.webp")
assert img is not None, "file could not be read, check with os.path.exists()"

# Convert to HSV
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

# Define color ranges in HSV (double ranges for better detection)
# Red
lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 100, 100])
upper_red2 = np.array([180, 255, 255])

# Green
lower_green1 = np.array([35, 100, 100])
upper_green1 = np.array([85, 255, 255])
lower_green2 = np.array([85, 50, 100])
upper_green2 = np.array([95, 255, 255])

# Blue
lower_blue1 = np.array([85, 100, 100])
upper_blue1 = np.array([130, 255, 255])
lower_blue2 = np.array([130, 50, 100])
upper_blue2 = np.array([140, 255, 255])

# Yellow
lower_yellow1 = np.array([20, 100, 100])
upper_yellow1 = np.array([35, 255, 255])
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
erodeRed = cv.erode(mask_red, None, iterations=2)
erodeYellow = cv.erode(mask_yellow, None, iterations=2)
erodeGreen = cv.erode(mask_green, None, iterations=2)
erodeBlue = cv.erode(mask_blue, None, iterations=2)

# Apply masks to original image
result_red = cv.bitwise_and(img, img, mask=erodeRed)
result_green = cv.bitwise_and(img, img, mask=erodeGreen)
result_blue = cv.bitwise_and(img, img, mask=erodeBlue)
result_yellow = cv.bitwise_and(img, img, mask=erodeYellow)

# Determine types of signals present
tipos_senales = []
threshold_area = 5000  # Minimum number of pixels to consider the presence of color

def check_color_presence(result_img):
    gray = cv.cvtColor(result_img, cv.COLOR_BGR2GRAY)
    return np.count_nonzero(gray) > threshold_area

if check_color_presence(result_red):
    tipos_senales.append("Restrictiva (Rojo)")
if check_color_presence(result_yellow):
    tipos_senales.append("Preventiva (Amarillo)")
if check_color_presence(result_blue):
    tipos_senales.append("Servicios (Azul)")
if check_color_presence(result_green):
    tipos_senales.append("Destino (Verde)")

print("\nClasificación de señales detectadas:")
for tipo in tipos_senales:
  print("- " + tipo)

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

plt.show()
