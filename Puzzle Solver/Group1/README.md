# Image Matching and Transformation

This project is designed to compare two images based on multiple similarity metrics and align them based on patch matching and rotation. The following methods are used for comparison:

- **SSIM (Structural Similarity Index)** for overall image quality comparison.
- **ORB (Oriented FAST and Rotated BRIEF)** for detecting and matching keypoints.
- **Frequency Domain Similarity** using Fourier Transform for comparing frequency content.
- **Color Similarity** in the HSV color space using histograms.
- **Patch-Based Matching** where the images are split into 16 patches, compared, and reassembled.

### Contributors
- Hassan Ashkenani
- Alireza Khoshnood
- Mohammad Reza Arjamandi

### Requirements
- Python 3.x
- OpenCV
- Numpy
- scikit-image