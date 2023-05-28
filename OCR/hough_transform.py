import cv2
import numpy as np

sr = cv2.dnn_superres.DnnSuperResImpl_create()

path = "./OCR/EDSR_x3.pb"
sr.readModel(path)
sr.setModel("edsr", 3)

# Read the input image
image = cv2.imread("./OCR/detect3.jpg")


image = sr.upsample(image)
# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Perform edge detection using Canny
edges = cv2.Canny(gray, 50, 150)
cv2.imshow("Edges", edges)
cv2.waitKey(0)

# Apply the probabilistic Hough Line Transform to detect lines in the edge image
lines = cv2.HoughLinesP(
    edges, rho=1, theta=np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10
)

# Find the longest line
longest_line_length = 0
longest_line_angle = 0

if len(lines):
    for line in lines:
        x1, y1, x2, y2 = line[0]
        line_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        if line_length > longest_line_length:
            p1, p2 = (x1, y1), (x2, y2)
            longest_line_length = line_length
            longest_line_angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi

print("Angle of the longest line:", longest_line_angle)

blank = np.zeros((image.shape[0], image.shape[1], 3))
cv2.line(blank, p1, p2, (0, 0, 255), 2)
cv2.imshow("Line", blank)
cv2.waitKey(0)

(h, w) = image.shape[:2]
center = (w // 2, h // 2)
rotation_matrix = cv2.getRotationMatrix2D(center, longest_line_angle, 1.0)
rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))

# Display or save the rotated image
cv2.imwrite("rotated.jpg", rotated_image)
cv2.imshow("Rotated Image", rotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
