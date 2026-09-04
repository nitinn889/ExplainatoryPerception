"""
Camera capture for the Jetson Nano (Phase 0/1).

Responsible for opening the CSI (IMX219 via nvarguscamerasrc) or USB (v4l2)
camera through OpenCV and yielding frames to detector.py.
"""
