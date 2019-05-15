import numpy as np

# IRCT CALIBRATION DATA
# Use a metering device to measure the geometric setup of the infrared CT system, namely the distances from the camera
# to the rotating table base.
TABLE_HEIGHT_MM = 79 # rotation table height in mm
CAMERA_HEIGHT_MM = 310 # distance in mm, in direction perpedicular to base, i.e. vertical axis
CAMERA_TO_TABLE_DX_MM = 760 # distance in mm, in direction from camera towards the rotating table
CAMERA_TO_TABLE_DY_MM = 3 # distance in mm, direction based on rhs coordinate frame (dx x dz), i.e. to the left w.r.t. dx

ROTATION_DIRECTION = -1# +1 corresponds to rotation direction about camera z axis, -1 is the opposite direction

# The volume is rectangular and camera-axis aligned. It is defined w.r.t. reference point on top of the caliber,
# which is determined by imaging the calibration object.
VOLUME_LENGTH = 200# x size
VOLUME_WIDTH = 200# y size
VOLUME_HEIGHT = 300# z size


def IRCT_CALIBRATION_OBJECT():
    """
    Marker locations on the IRCT calibration object based on manual measurements. The reference point should be
    at the center of rotating table base, i.e. all marker coordinate are to be defined w.r.t. to the center.

    :return: numpy array of 3D point coordinates corresponding to marker location on the calibration object
    """
    h = 8
    pts = [
        [0,0,134+h],
        [33,0,108+h],
        [-16.5,-28,92+h],
        [-16.5,28,75+h],
        [33,0,60.5+h],
        [-16.5,-28,43+h],
        [-16.5,28,28+h],
        [33,0,12+h]
    ]
    return np.array(pts)

# TRACKER CALIBRATION DATA
CHECKERBOARD_SQUARE_SIZE_MM = 25.4

