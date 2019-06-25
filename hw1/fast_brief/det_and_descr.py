import cv2
import numpy as np


def convolve(img, kernel):
    kernel_size = len(kernel)
    sobelled = np.zeros((img.shape[0]-kernel_size-1, img.shape[1]-kernel_size-1))
    for y in range(1, img.shape[0]-kernel_size-1):
        for x in range(1, img.shape[1]-kernel_size-1):
            sobelled[y-1, x-1] = np.sum(np.multiply(img[y-1:y+kernel_size-1, x-1:x+kernel_size-1], kernel))
    return sobelled

def gaussian_blur(img, kernel_sz, sigma):
    kernel = np.ones((kernel_sz, kernel_sz)) / sigma
    return convolve(img, kernel)


def circle(row, col):
    point1 = (row+3, col)
    point3 = (row+3, col-1)
    point5 = (row+1, col+3)
    point7 = (row-1, col+3)
    point9 = (row-3, col)
    point11 = (row-3, col-1)
    point13 = (row+1, col-3)
    point15 = (row-1, col-3)
    return [point1, point3, point5, point7, point9, point11, point13, point15]

def fast(img, threshold = 50):
    im_x, im_y = img.shape
    pts = []
    for row in range(5, im_x-5):
        for col in range(5, im_y-5):
            c = circle(row, col)
            center = img[row, col]
            row1, col1 = c[0]
            row9, col9 = c[4]
            row5, col5 = c[2]
            row13, col13 = c[6]
            intensity1 = int(img[row1][col1])
            intensity9 = int(img[row9][col9])
            intensity5 = int(img[row5][col5])
            intensity13 = int(img[row13][col13])
            count = 0
            if abs(intensity1 - center) > threshold:
                count += 1 
            if abs(intensity9 - center) > threshold:
                count += 1
            if abs(intensity5 - center) > threshold:
                count += 1
            if abs(intensity13 - center) > threshold:
                count += 1
            if count >= 3:
                pts.append((col, row))
    return pts
                
            
def find_keypoints_candidates(img):
    return fast(img, threshold=120)


def get_descriptor_points(S, n=256):
    np.random.seed(50)
    xs= np.random.randint(0, S, n*5)
    ys = np.random.randint(0, S, n*5)
    points = set()
    i = 0
    while len(points) < n:
        points.add((xs[i], ys[i]))
        i+=1
        
    xs= np.random.randint(0, S, n*5)
    ys = np.random.randint(0, S, n*5)
    points_two = set()
    i = 0
    while len(points_two) < n:
        points_two.add((xs[i], ys[i]))
        i+=1
    

    return list(zip(list(points), list(points_two)))

def binarize(image, point_pairs):
    result = []
    max_x, max_y = image.shape
    for (left, right) in point_pairs:
        if left[0] >= max_x or left[1] >= max_y:
            left_val = 0
        else:
            left_val = image[left[0], left[1]]

        if right[0] >= max_x or right[1] >= max_y:
            right_val = left_val
        else:
            right_val = image[right[0], right[1]]
        if(left_val < right_val):
            result.append(1)
        else:
            result.append(0)
            
    return result
                
def adjust(points, left):
    adjusted = []
    for p1, p2 in points:
        adjusted.append(((p1[0] + left[0], p1[1] + left[1]), (p2[0] + left[0], p2[1] + left[1])))
    return adjusted

        
def brief(img, points, S=10, n=64):
    result = []
    x_max, y_max = img.shape
    for point in points:
        l, r = (max(0, point[0] - S), max(0, point[1] - S)), (min(x_max, point[0] + S), min(y_max, point[1] + S))
        point_pairs = get_descriptor_points(S * 2, n=n)
        point_pairs = adjust(point_pairs, l)
        descriptor = binarize(img, point_pairs)
        result.append(descriptor)
    return result
        

def compute_descriptors(img, kp_arr):
    return np.vstack(brief(img, kp_arr)).astype(np.uint8)


# function for keypoints and descriptors calculation
def detect_keypoints_and_calculate_descriptors(img):
    # img - numpy 2d array (grayscale image)
    img_blur = gaussian_blur(img, 3, 3.0)

    # keypoints
    kp_arr = find_keypoints_candidates(img_blur)
    # kp_arr is array of 2d coordinates-tuples, example:
    # [(x0, y0), (x1, y1), ...]
    # xN, yN - integers

    # descriptors
    descr_arr = compute_descriptors(img_blur, kp_arr)
    # cv_descr_arr is array of descriptors (arrays), example:
    # [[v00, v01, v02, ...], [v10, v11, v12, ...], ...]
    # vNM - floats

    return kp_arr, descr_arr
