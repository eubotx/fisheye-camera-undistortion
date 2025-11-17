# Fisheye Camera Undistortion

Fisheye camera distortion correction based on OpenCV's chessboard calibration algorithm.

---

## Getting Started

1. **Capture calibration images**  
   Run `capture_calibration_images.py` to take several standard chessboard images with the fisheye lens to be corrected. Place them into the `calibration_images` folder.

2. **Calibrate the camera**  
   Run `calibrate.py` to calculate the internal parameter matrix `K` and the distortion coefficient vector `D`.  
   The calibration is saved as a YAML file (`calibration_output.yaml`) that **can also be used in ROS as a `camera_info` file**.

3. **Undistort images or video**
   - **a. Correct a single image**  
     Run `image_correction.py` to correct a single image `distorted.jpg` captured by the camera.  
     Output: `undistorted.jpg`.

   - **b. Correct live video**  
     Run `video_correction.py` to correct the live camera feed. Make sure to set the correct camera source.

---

## Notes

- Make sure your camera resolution matches the one used for calibration.  
- Calibration images should cover the full field of view and have good lighting for best results.  
- The YAML calibration file is compatible with ROS `camera_info` messages, so it can be directly used in ROS-based image pipelines.
