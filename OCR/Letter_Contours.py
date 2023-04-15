import cv2 as cv
import os
import numpy as np


def x_cord_contour(contours):
    M = cv.moments(contours)
    return int(M["m10"] / M["m00"])


def y_cord_contour(contours):
    # Returns the Y cordinate for the contour centroid
    M = cv.moments(contours)
    return int(M["m01"] / M["m00"])


def sort_contours(contours):
    contours_boxes = [list(cv.boundingRect(contour)) for contour in contours]

    for i in range(len(contours_boxes)):
        contours_boxes[i].append(i)

    c = np.array(contours_boxes)
    if len(c[:] > 0):
        max_height = np.max(c[:, 3])

        # Sort the contours by y-value
        by_y = sorted(contours_boxes, key=lambda x: x[1])  # y values

        line_y = by_y[0][1]  # first y
        line = 1
        by_line = []

        # Assign a line number to each contour
        for x, y, w, h, i in by_y:
            if y > line_y + 2 * max_height / 3:
                line_y = y
                line += 1

            by_line.append((line, x, y, w, h, i))

        # This will now sort automatically by line then by x
        return [i for line, x, y, w, h, i in sorted(by_line)]

    return []


# Filter contours with size ratio to drop too small and too large contours
def filter_contours(contours, img):
    filtered_contours = []

    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        height, width, channels = img.shape
        if h / height < 0.3 or w / width < 0.2 or h / height > 0.75 or w / width > 0.4:
            continue
        else:
            filtered_contours.append(contour)

    sorted_contours = sorted(filtered_contours, key=y_cord_contour, reverse=False)

    return sorted_contours


# Filter contours with size ratio to drop too small and too large contours and
# eliminate contours with overlap
def filter_contours_without_overlap(contours, hierarchy, img):
    filtered_contours = []
    indexes = []

    for i in range(len(contours)):
        x, y, w, h = cv.boundingRect(contours[i])
        height, width, channels = img.shape
        if h / height < 0.2 or w / width < 0.05 or h / height > 0.75 or w / width > 0.4:
            continue
        elif h / height < 0.325 and w / width < 0.325:
            continue
        elif y > height * 0.6:
            continue
        else:
            print(h / height, w / width)
            # filtered_contours.append(contours[i])
            indexes.append(i)

    for index in indexes:
        if hierarchy[0][index][3] in indexes:
            continue
        else:
            filtered_contours.append(contours[index])

    sorted_contours_indexes = sort_contours(filtered_contours)

    sorted_contours = [filtered_contours[i] for i in sorted_contours_indexes]

    return sorted_contours


# Preprocess image and find contours
def find_contours(img, path):
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    # ret, thresh = cv.threshold(gray, 100, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    # thresh = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY,15,4)
    noise_reduced = cv.fastNlMeansDenoising(gray, None, h=7, searchWindowSize=31)
    thresh = cv.adaptiveThreshold(
        noise_reduced, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 25, 2
    )

    erode_kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    eroded = cv.erode(thresh, erode_kernel)

    dilate_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
    dilated = cv.dilate(eroded, dilate_kernel)

    try:
        contours, hierarchy = cv.findContours(
            dilated, cv.RETR_TREE, cv.CHAIN_APPROX_NONE
        )
    except:
        ret_img, contours, hierarchy = cv.findContours(
            dilated, cv.RETR_TREE, cv.CHAIN_APPROX_NONE
        )

    sorted_contours = filter_contours_without_overlap(contours, hierarchy, img)

    # for contour in sorted_contours:
    #     blank = np.zeros((img.shape[0],img.shape[1],3), dtype='uint8') + 255
    #     cv.drawContours(blank, contour, -1, (0,0,0), 1)
    # cv.imshow('Contours blank', blank)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    # blank = np.zeros((img.shape[0],img.shape[1],3))
    # for contour in sorted_contours:
    #     cv.drawContours(blank,contour, -1, (255,0,0),1)
    # cv.imshow("All contours", blank)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    save_contours(sorted_contours, img.shape, path, img)


# Save detected character as jpg
def save_contours(contours, img_size, path, img):
    i = 1
    prefix = path.split("/")[-1].split(".")[0][6:]
    print(prefix)
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        roi_image = img[y - 3 : y + h + 3, x - 3 : x + w + 3]
        roi_image = cv.resize(roi_image, dsize=(128, 128), interpolation=cv.INTER_CUBIC)
        cv.imwrite(
            "./Cropped Letters/Video 18/v18_" + prefix + "_" + str(i) + ".jpg",
            roi_image,
        )
        i += 1


# Upsample image
def upscale_image(imgpath, sr):
    img = cv.imread(imgpath)
    upsampled = sr.upsample(img)
    return upsampled


# Find and save contours with letters
def find_characters(imgpath, sr):
    upsampled = upscale_image(imgpath, sr)
    find_contours(upsampled, imgpath)


if __name__ == "__main__":
    sr = cv.dnn_superres.DnnSuperResImpl_create()

    path = "./OCR/EDSR_x3.pb"
    sr.readModel(path)
    sr.setModel("edsr", 2)

    src_path = "./Cropped License Plates/Video 18/"
    image_list = os.listdir(src_path)

    for image_path in image_list:
        image_path = os.path.join(src_path, image_path)
        find_characters(image_path, sr)
        # upscaled = upscale_image("./Cropped_New/"+image_path, sr)

    # image_path = "detected106.jpg"
    # img = cv.imread(image_path)
    # upsampled = sr.upsample(img)
    # upsampled = cv.detailEnhance(img,10,0.15)
    # upsampled = cv.resize(upsampled, (img.shape[1]*3,img.shape[0]*3))
    # cv.imwrite("esdr_2x.jpg",upsampled)
