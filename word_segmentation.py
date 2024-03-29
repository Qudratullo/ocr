import math
import cv2
import numpy as np
import os


def read_images_and_segmentation():
    """reads images from data/ and outputs the word-segmentation to out/"""

    # read input images from 'in' directory
    img_files = os.listdir('data/')
    for (i, f) in enumerate(img_files):
        print('Segmenting words of sample %s' % f)

        # read image, prepare it by resizing it to fixed height and converting it to grayscale
        img = prepare_img(cv2.imread('data/%s' % f))

        # execute segmentation with given parameters
        # -kernelSize: size of filter kernel (odd integer)
        # -sigma: standard deviation of Gaussian function used for filter kernel
        # -theta: approximated width/height ratio of words, filter function is distorted by this factor
        # - minArea: ignore word candidates smaller than specified area
        res = word_segmentation(img, kernel_size=45, sigma=40, theta=7, min_area=50)

        # write output to 'out/inputFileName' directory
        if not os.path.exists('out/%s' % f):
            os.mkdir('out/%s' % f)

        # iterate over all segmented words
        print('Segmented into %d words' % len(res))
        for (j, w) in enumerate(res):
            (wordBox, wordImg) = w
            (x, y, w, h) = wordBox
            cv2.imwrite('out/%s/%d.png' % (f, j), wordImg)  # save word
            cv2.rectangle(img, (x, y), (x + w, y + h), 0, 1)  # draw bounding box in summary image

        # output summary image with bounding boxes around words
        cv2.imwrite('out/%s/summary.png' % f, img)


def word_segmentation(img, kernel_size=25, sigma=11, theta=7, min_area=0, sort_second = False):
    """Scale space technique for word segmentation proposed by R. Manmatha: http://ciir.cs.umass.edu/pubfiles/mm-27.pdf

    Args:
        img: grayscale uint8 image of the text-line to be segmented.
        kernel_size: size of filter kernel, must be an odd integer.
        sigma: standard deviation of Gaussian function used for filter kernel.
        theta: approximated width/height ratio of words, filter function is distorted by this factor.
        min_area: ignore word candidates smaller than specified area.
        kernel_filter: apply kernel filter.

    Returns:
        List of tuples. Each tuple contains the bounding box and the image of the segmented word.
    """

    # apply filter kernel
    kernel = create_kernel(kernel_size, sigma, theta)
    img_filtered = cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REPLICATE).astype(np.uint8)
    (_, img_thres) = cv2.threshold(img_filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img_thres = 255 - img_thres

    # find connected components. OpenCV: return type differs between OpenCV2 and 3
    if cv2.__version__.startswith('3.'):
        (_, components, _) = cv2.findContours(img_thres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    else:
        (components, _) = cv2.findContours(img_thres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # append components to result
    res = []
    for c in components:
        # skip small word candidates
        if cv2.contourArea(c) < min_area:
            continue
        # append bounding box and image of word to result list
        curr_box = cv2.boundingRect(c)  # returns (x, y, w, h)
        (x, y, w, h) = curr_box
        curr_img = img[y:y + h, x:x + w]
        res.append((curr_box, curr_img))

    # return list of words, sorted by y-coordinate
    return sorted(res, key=lambda entry: entry[0][sort_second])


def prepare_img(img):
    """convert given image to grayscale image (if needed) and resize to desired height"""
    assert img.ndim in (2, 3)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img


def create_kernel(kernel_size, sigma, theta):
    """create anisotropic filter kernel according to given parameters"""
    assert kernel_size % 2  # must be odd size
    half_size = kernel_size // 2

    kernel = np.zeros([kernel_size, kernel_size])
    sigma_x = sigma
    sigma_y = sigma * theta

    for i in range(kernel_size):
        for j in range(kernel_size):
            x = i - half_size
            y = j - half_size

            exp_term = np.exp(-x ** 2 / (2 * sigma_x) - y ** 2 / (2 * sigma_y))
            x_term = (x ** 2 - sigma_x ** 2) / (2 * math.pi * sigma_x ** 5 * sigma_y)
            y_term = (y ** 2 - sigma_y ** 2) / (2 * math.pi * sigma_y ** 5 * sigma_x)

            kernel[i, j] = (x_term + y_term) * exp_term

    kernel = kernel / np.sum(kernel)
    return kernel
