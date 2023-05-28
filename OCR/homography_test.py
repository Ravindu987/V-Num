import cv2
import numpy as np

sr = cv2.dnn_superres.DnnSuperResImpl_create()

path = "./OCR/EDSR_x3.pb"
sr.readModel(path)
sr.setModel("edsr", 3)

# Read the input image
image = cv2.imread("./OCR/detect4.jpg")


image = sr.upsample(image)
# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Perform edge detection using Canny
edges = cv2.Canny(gray, 50, 150)

cv2.imshow("Edges", edges)

# Find contours in the edge image
contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
# Sort contours based on area
sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
# print(contours)
# Select the largest contour as the license plate contour


filtered_contours = []

for contour in sorted_contours:
    x, y, w, h = cv2.boundingRect(contour)
    height, width, channels = image.shape
    if h / height < 0.5 or w / width < 0.5:
        continue
    else:
        blankx = np.zeros((image.shape[0], image.shape[1], 3))
        cv2.drawContours(blankx, contour, -1, (0, 255, 0), 1)
        cv2.imshow("Contour", blankx)
        cv2.waitKey(0)
        filtered_contours.append(contour)

license_plate_contour = filtered_contours[2]

blank = np.zeros((image.shape[0], image.shape[1], 3))
# Return all detected letters
# for contour in filtered_contours:
#     cv2.drawContours(blank, contour, -1, (0, 255, 0), 1)

# cv2.imshow("Co", blank)
# cv2.waitKey(0)


blank2 = np.zeros((image.shape[0], image.shape[1], 3))
# Return all detected letters
cv2.drawContours(blank2, license_plate_contour, -1, (0, 255, 0), 1)

cv2.imshow("Co2", blank2)
cv2.waitKey(0)

# Approximate the contour to a rectangle
epsilon = 0.1 * cv2.arcLength(license_plate_contour, True)
approx = cv2.approxPolyDP(license_plate_contour, epsilon, True)

# Find the bounding rectangle of the approximated contour
x, y, width, height = cv2.boundingRect(approx)

rect = cv2.minAreaRect(approx)
box = cv2.boxPoints(rect)
box = np.int0(box)
print(box)

blanky = np.zeros((image.shape[0], image.shape[1], 3))
cv2.rectangle(blanky, box[0], box[2], (255, 0, 0), 1)
# cv2.drawContours(blanky, approx, 0, (255, 0, 0), 1)
cv2.imshow("Box", blanky)

# Extract the corner points of the bounding rectangle
corner_points = np.array(
    [[x, y + 20], [x + width, y], [x + width, y + height - 20], [x, y + height]],
    dtype=np.float32,
)


# Define destination points as a rectangle with desired dimensions
front_view_width = 400
front_view_height = 200
dst_points = np.array(
    [
        [0, 0],
        [front_view_width - 1, 0],
        [front_view_width - 1, front_view_height - 1],
        [0, front_view_height - 1],
    ],
    dtype=np.float32,
)

# Compute perspective transformation matrix
matrix = cv2.getPerspectiveTransform(corner_points.astype(np.float32), dst_points)

# Apply perspective transformation to the image
output_image = cv2.warpPerspective(image, matrix, (front_view_width, front_view_height))

# Display or save the transformed image
cv2.imshow("Front View License Plate", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
