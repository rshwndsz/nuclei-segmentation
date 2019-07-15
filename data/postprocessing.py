# Adapted From https://github.com/ugent-korea/pytorch-unet-segmentation/blob/master/src/post_processing.py
import numpy as np
import cv2


def postprocess(image_path):
    # TODO Add more comments
    # TODO Display final images on Visdom
    img_original = cv2.imread(image_path)
    img = cv2.imread(image_path)

    # Convert to 1 channel if saved in 3 channels
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold
    ret, bin_image = cv2.threshold(gray, 127, 255,
                                   cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((1, 1), np.uint8)
    kernel1 = np.ones((1, 1), np.uint8)

    closing = cv2.morphologyEx(bin_image, cv2.MORPH_CLOSE,
                               kernel, iterations=1)

    sure_bg = cv2.dilate(closing, kernel1, iterations=1)

    dist_transform = cv2.distanceTransform(closing, cv2.DIST_L2, 5)

    ret, sure_fg = cv2.threshold(dist_transform,
                                 0.2*dist_transform.max(),
                                 255, 0)
    sure_fg = np.uint8(sure_fg)

    unknown = cv2.subtract(sure_bg, sure_fg)

    ret, markers = cv2.connectedComponents(sure_fg)
    markers_plus1 = markers + 1
    markers_plus1[unknown == 255] = 0

    markers_watershed = cv2.watershed(img, markers_plus1)

    img_x, img_y = img_original.shape[0], img_original.shape[1]
    white, white_color = np.zeros((img_x, img_y, 3)), np.zeros((img_x, img_y, 3))
    white += 255
    white_color += 255

    white[markers_watershed != 1] = [0, 0, 0]
    white_color[markers_watershed != 1] = [255, 0, 0]

    white_np = np.asarray(white)
    watershed_grayscale = white_np.transpose([2, 0, 1])[0, :, :]

    img[markers_watershed != 1] = [255, 0, 0]

    return watershed_grayscale
