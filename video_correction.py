import cv2
import yaml
import numpy as np

CAMERA_ID = 4  # Camera ID (usually 0 for built-in webcam)

def correct(img_in, k, d, dims, balance=1.0):
    dim1 = img_in.shape[:2][::-1]

    # Maintain aspect ratio
    assert abs(dim1[0] / dim1[1] - dims[0] / dims[1]) < 1e-4, \
        "Image to correct must maintain the same aspect ratio as calibration images."

    # Compute scaled camera matrix for full FOV (no cropping)
    new_k = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        k, d, dims, np.eye(3), balance=balance
    )

    # Generate undistortion maps
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        k, d, np.eye(3), new_k, dims, cv2.CV_16SC2
    )

    # Apply remap
    img_out = cv2.remap(
        img_in, map1, map2,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT
    )

    return img_out


if __name__ == '__main__':
    # Load calibration from YAML
    with open('./parameters/calibration_output.yaml', 'r') as f:
        calib = yaml.safe_load(f)

    Dims = (calib['image_width'], calib['image_height'])
    K = np.array(calib['camera_matrix']['data']).reshape(3,3)
    D = np.array(calib['distortion_coefficients']['data']).reshape(-1,1)

    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(3, Dims[0])
    cap.set(4, Dims[1])

    correct_flag = True
    balance = 1.0  # 1.0 = full uncropped view, 0 = max zoom-in

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if correct_flag:
            frame_undist = correct(frame, k=K, d=D, dims=Dims, balance=balance)

        cv2.imshow('Undistorted', frame_undist if correct_flag else frame)
        cv2.imshow('Distorted', frame)

        key = cv2.waitKey(1)
        if key == 13:       # ENTER toggles correction
            correct_flag = not correct_flag
        if key == 27:       # ESC to quit
            break
