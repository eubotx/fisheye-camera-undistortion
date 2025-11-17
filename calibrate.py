#!/usr/bin/env python3

import numpy as np
import cv2
import os
import yaml

CHESSBOARD_SIZE = (6, 9)


def calibrate(chessboard_path, show_chessboard=False):
    # Logical coordinates of chessboard corners
    obj_p = np.zeros((1, CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
    obj_p[0, :, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)

    obj_points = []
    img_points = []

    image_list = os.listdir(chessboard_path)
    gray = None
    for image in image_list:
        img = cv2.imread(os.path.join(chessboard_path, image))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(
            gray, CHESSBOARD_SIZE,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        if ret:
            obj_points.append(obj_p)
            cv2.cornerSubPix(
                gray, corners, (3, 3), (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
            )
            img_points.append(corners)
            print('Image ' + image + ' is valid for calibration')

            if show_chessboard:
                cv2.drawChessboardCorners(img, CHESSBOARD_SIZE, corners, ret)
                os.makedirs('./calibration_images_corners', exist_ok=True)
                cv2.imwrite(os.path.join('./calibration_images_corners', image), img)

    k = np.zeros((3, 3))
    d = np.zeros((4, 1))
    dims = gray.shape[::-1]
    num_valid_img = len(obj_points)

    if num_valid_img > 0:
        rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(num_valid_img)]
        tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(num_valid_img)]
        cv2.fisheye.calibrate(
            obj_points, img_points, gray.shape[::-1],
            k, d, rvecs, tvecs,
            cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC +
            cv2.fisheye.CALIB_FIX_SKEW,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
        )

    print(f"Found {num_valid_img} valid images for calibration")
    return k, d, dims


def write_yaml(K, D, dims, out_path="calibration_output.yaml"):
    calibration = {
        'image_width': int(dims[0]),
        'image_height': int(dims[1]),
        'camera_name': 'arena_camera',
        'camera_matrix': {
            'rows': 3,
            'cols': 3,
            'data': [float(x) for x in K.flatten()]
        },
        'distortion_model': "fisheye",
        'distortion_coefficients': {
            'rows': 1,
            'cols': len(D.flatten()),
            'data': [float(x) for x in D.flatten()]
        },
        'rectification_matrix': {
            'rows': 3,
            'cols': 3,
            'data': [1.0, 0.0, 0.0,
                     0.0, 1.0, 0.0,
                     0.0, 0.0, 1.0]
        },
        'projection_matrix': {
            'rows': 3,
            'cols': 4,
            'data': [
                float(K[0,0]), 0.0, float(K[0,2]), 0.0,
                0.0, float(K[1,1]), float(K[1,2]), 0.0,
                0.0, 0.0, 1.0, 0.0
            ]
        }
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        yaml.dump(calibration, f, default_flow_style=True, sort_keys=False)

    print(f"Calibration saved to {out_path}")


if __name__ == '__main__':
    os.makedirs('./parameters', exist_ok=True)

    K, D, Dims = calibrate('./calibration_images', show_chessboard=True)
    write_yaml(K, D, Dims, './parameters/calibration_output.yaml')
