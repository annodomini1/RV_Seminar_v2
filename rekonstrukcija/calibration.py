import numpy as np

# IRCT CALIBRATION DATA
# Use a metering device to measure the geometric setup of the infrared CT system, namely the distances from the camera
# to the rotating table base.
TABLE_HEIGHT_MM = 79 # rotation table height in mm
CAMERA_HEIGHT_MM = 310 # distance in mm, in direction perpedicular to base, i.e. vertical axis
CAMERA_TO_TABLE_DX_MM = 760 # distance in mm, in direction from camera towards the rotating table
CAMERA_TO_TABLE_DY_MM = 3 # distance in mm, direction based on rhs coordinate frame (dx x dz), i.e. to the left w.r.t. dx

ROTATION_DIRECTION = -1 # +1 corresponds to rotation direction about camera z axis, -1 is the opposite direction

# The volume is rectangular and camera-axis aligned. It is defined w.r.t. reference point on top of the caliber,
# which is determined by imaging the calibration object.
VOLUME_LENGTH = 200 # x size
VOLUME_WIDTH = 200 # y size
VOLUME_HEIGHT = 350 # z size

def IRCT_CALIBRATION_OBJECT():
    """
    Marker locations on the IRCT calibration object based on manual measurements. The reference point should be
    at the center of rotating table base, i.e. all marker coordinate are to be defined w.r.t. to the center.
    
    :return: numpy array of 3D point coordinates corresponding to marker location on the calibration object 
    """
    d1=79.5
    d2=7.5
    d3=235.0
    d4=64.5
    d5=95.5
    mb=4.5
    md=21.5
    mc=mb+md/2
    
    s=d1+d2/2+mc
    h1=d3-d2*3/2-d4
    h2=h1-d2-d5
    
    pts = [
        [ 0, 0, d3+mc],
        [ 0, s, h1],
        [-s, 0, h1],
        [ 0,-s, h1],
        [ 0, s, h2],
        [-s, 0, h2],
        [ 0,-s, h2]
    ]
    
    return np.array(pts)


def IRCT_CALIBRATION_OBJECT_V1():
    """
    Marker locations on the IRCT calibration object based on manual measurements. The reference point was 
    at the top marker of caliber object.

    V1  was used in 2015.
    
    :return: numpy array of 3D point coordinates corresponding to marker location on the calibration object 
    """
    # define basic measurements
    d = 18.5  # ball diameter
    dz = 22.0  # ball height from base
    dh = np.round((131 - 12) / 15)  # distance between lego holes
    df = 20 + 1  # center to based of attachement frame + 1 mm gap

    # reference point is the top ball
    b0 = np.array((0, 0, 0))
    # define top frame center
    p_top_frame = np.array((0, 0, - d / 2 - (dz - d) - dh / 2))
    # on one side of the frame (first ball is second highest)
    d_fb = df + d / 2 + (dz - d)  # distance from base
    b1 = p_top_frame + np.array((d_fb, 0, -dh))
    b4 = p_top_frame + np.array((d_fb, 0, -7 * dh))
    b7 = p_top_frame + np.array((d_fb, 0, -13 * dh))
    # on the other side of the frame (first ball is third highest)
    b2 = p_top_frame + np.array((-d_fb * np.cos(np.pi / 3), -d_fb * np.sin(np.pi / 3), -3 * dh))
    b5 = p_top_frame + np.array((-d_fb * np.cos(np.pi / 3), -d_fb * np.sin(np.pi / 3), -9 * dh))
    # on the other side of the frame (first ball is fourth highest)
    b3 = p_top_frame + np.array((-d_fb * np.cos(np.pi / 3), d_fb * np.sin(np.pi / 3), -5 * dh))
    b6 = p_top_frame + np.array((-d_fb * np.cos(np.pi / 3), d_fb * np.sin(np.pi / 3), -11 * dh))
    # define bottom frame center and ground
    p_bottom_frame = p_top_frame + (0, 0, -15 * dh - 4 - 8 - 4)
    p_bottom_ground = p_bottom_frame + (0, 0, -4 - 8 - 4)

    # create array of points
    pts = np.vstack((b0, b1, b2, b3, b4, b5, b6, b7))

    pts = np.array(((0, 0, 251), \
                    (-48.5, 84, 234), (-48.5, 84, 162), (-48.5, 84, 59), \
                    (97, 0, 234), (97, 0, 162), (97, 0, 59), \
                    (-48.5, -84, 234), (-48.5, -84, 162), (-48.5, -84, 59)))

    pts = pts - np.array(((0, 0, 251)))

    return pts


# TRACKER CALIBRATION DATA
CHECKERBOARD_SQUARE_SIZE_MM = 25.4

