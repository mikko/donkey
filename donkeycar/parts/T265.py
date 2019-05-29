import os
import time
import numpy as np
from PIL import Image
import glob
import pyrealsense2 as rs

t265_handle = None
t265_streams = {}

def vec3d_to_dict(vec):
    return {
        'x': vec.x,
        'y': vec.y,
        'z': vec.z
    }

def quat_to_dict(vec):
    return {
        'x': vec.x,
        'y': vec.y,
        'z': vec.z,
        'w': vec.w
    }

class T265:
    def __init__(self, defishing=False):
        if (defishing):
            print('Defishing requires OpenCV: importing')
            import cv2
            self.cv2 = cv2
        self.pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.pose) # Positional data (translation, rotation, velocity etc)
        # cfg.enable_stream(rs.stream.fisheye, 1) # Left camera
        # cfg.enable_stream(rs.stream.fisheye, 2) # Right camera
        self.pipe.start(cfg)
        self.frame_l = None
        self.frame_r = None
        self.translation = None
        self.velocity = None
        self.acceleration = None
        self.rotation = None
        self.on = True
        print('T265 initialized')

    def update(self):
        while self.on:
            frames = self.pipe.wait_for_frames()

#            left = frames.get_fisheye_frame(1)
#            self.frame_l = np.asanyarray(left.get_data())
#            right = frames.get_fisheye_frame(2)
#            self.frame_r = np.asanyarray(right.get_data())

            pose = frames.get_pose_frame()
            if pose:

                data = pose.get_pose_data()
                self.translation = vec3d_to_dict(data.translation)
                self.acceleration = vec3d_to_dict(data.acceleration)
                self.velocity = vec3d_to_dict(data.velocity)
                self.rotation = quat_to_dict(data.rotation)

        self.pipe.stop()

    def run_threaded(self):
        return self.frame_l, self.frame_r, self.translation, self.acceleration, self.velocity, self.rotation

    def shutdown(self):
        print('Shutting down T265')
        self.pipe.stop()
        print('T265 stopped')
