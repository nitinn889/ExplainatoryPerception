"""
SSD-MobileNet-V2 inference wrapper (Phase 1). No YOLO — excluded by course
requirement.

Loads a pretrained (COCO to start) SSD-MobileNet-V2, runs TensorRT-optimized
inference on frames from camera.py. Outputs a list of
{class, confidence, bbox (xmin,ymin,xmax,ymax normalized 0-1)} per frame.
"""
