import cv2
import yaml
import numpy as np

def correct(img_in, k, d, dims):
    dim1 = img_in.shape[:2][::-1]
    assert abs(dim1[0] / dim1[1] - dims[0] / dims[1]) < 1e-4, \
        "Image to undistort must have the same aspect ratio as calibration images"

    map1, map2 = cv2.fisheye.initUndistortRectifyMap(k, d, np.eye(3), k, dims, cv2.CV_16SC2)
    img_out = cv2.remap(img_in, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return img_out


if __name__ == '__main__':
    # Load calibration from YAML
    with open('./parameters/calibration_output.yaml', 'r') as f:
        calib = yaml.safe_load(f)

    Dims = (calib['image_width'], calib['image_height'])
    K = np.array(calib['camera_matrix']['data']).reshape(3,3)
    D = np.array(calib['distortion_coefficients']['data']).reshape(-1,1)

    img = cv2.imread('distorted.jpg')
    img_undistorted = correct(img, k=K, d=D, dims=Dims)

    cv2.imshow('Undistorted', img_undistorted)
    cv2.imwrite('undistorted.jpg', img_undistorted)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
