import os
import dicom
import numpy as np
import math

_FLOAT_EPS_4 = np.finfo(float).eps * 4.0

def deg2rad(ang):
    '''
    Convert angle in degrees to radians.

    :param ang: Angle in degrees. 
    :return: Angle in radians.
    '''
    return ang * np.pi / 180.0


def rad2deg(ang):
    '''
    Convert angle in radians to degrees.

    :param ang: Angle in radians. 
    :return: Angle in degrees.
    '''
    return ang * np.pi / 180.0

def transRigid3D(trans=(0, 0, 0), rot=(0, 0, 0)):
    '''
    Rigid body transformation

    :param trans: Translation 3-vector (tx,ty,tz).
    :param rot: Rotation vector (rx,ry,rz).
    :return: Rigid-body 4x4 transformation matrix.
    '''
    Trotx = np.array(((1, 0, 0, 0),
                      (0, np.cos(rot[0]), -np.sin(rot[0]), 0),
                      (0, np.sin(rot[0]), np.cos(rot[0]), 0),
                      (0, 0, 0, 1)))
    Troty = np.array(((np.cos(rot[1]), 0, np.sin(rot[1]), 0),
                      (0, 1, 0, 0),
                      (-np.sin(rot[1]), 0, np.cos(rot[1]), 0),
                      (0, 0, 0, 1)))
    Trotz = np.array(((np.cos(rot[2]), -np.sin(rot[2]), 0, 0),
                      (np.sin(rot[2]), np.cos(rot[2]), 0, 0),
                      (0, 0, 1, 0),
                      (0, 0, 0, 1)))
    Ttrans = np.array(((1, 0, 0, trans[0]),
                       (0, 1, 0, trans[1]),
                       (0, 0, 1, trans[2]),
                       (0, 0, 0, 1)))
    return np.dot(np.dot(np.dot(Trotx, Troty), Trotz), Ttrans)


def mat2euler(M, cy_thresh=None):
    ''' Discover Euler angle vector from 3x3 matrix

    Uses the conventions above.

    Parameters
    ----------
    M : array-like, shape (3,3)
    cy_thresh : None or scalar, optional
       threshold below which to give up on straightforward arctan for
       estimating x rotation.  If None (default), estimate from
       precision of input.

    Returns
    -------
    z : scalar
    y : scalar
    x : scalar
       Rotations in radians around z, y, x axes, respectively

    Notes
    -----
    If there was no numerical error, the routine could be derived using
    Sympy expression for z then y then x rotation matrix, which is::

      [                       cos(y)*cos(z),                       -cos(y)*sin(z),         sin(y)],
      [cos(x)*sin(z) + cos(z)*sin(x)*sin(y), cos(x)*cos(z) - sin(x)*sin(y)*sin(z), -cos(y)*sin(x)],
      [sin(x)*sin(z) - cos(x)*cos(z)*sin(y), cos(z)*sin(x) + cos(x)*sin(y)*sin(z),  cos(x)*cos(y)]

    with the obvious derivations for z, y, and x

       z = atan2(-r12, r11)
       y = asin(r13)
       x = atan2(-r23, r33)

    Problems arise when cos(y) is close to zero, because both of::

       z = atan2(cos(y)*sin(z), cos(y)*cos(z))
       x = atan2(cos(y)*sin(x), cos(x)*cos(y))

    will be close to atan2(0, 0), and highly unstable.

    The ``cy`` fix for numerical instability below is from: *Graphics
    Gems IV*, Paul Heckbert (editor), Academic Press, 1994, ISBN:
    0123361559.  Specifically it comes from EulerAngles.c by Ken
    Shoemake, and deals with the case where cos(y) is close to zero:

    See: http://www.graphicsgems.org/

    The code appears to be licensed (from the website) as "can be used
    without restrictions".
    '''
    M = np.asarray(M)
    if cy_thresh is None:
        try:
            cy_thresh = np.finfo(M.dtype).eps * 4
        except ValueError:
            cy_thresh = _FLOAT_EPS_4
    r11, r12, r13, r21, r22, r23, r31, r32, r33 = M.flat
    # cy: sqrt((cos(y)*cos(z))**2 + (cos(x)*cos(y))**2)
    cy = math.sqrt(r33*r33 + r23*r23)
    if cy > cy_thresh: # cos(y) not close to zero, standard form
        z = math.atan2(-r12,  r11) # atan2(cos(y)*sin(z), cos(y)*cos(z))
        y = math.atan2(r13,  cy) # atan2(sin(y), cy)
        x = math.atan2(-r23, r33) # atan2(cos(y)*sin(x), cos(x)*cos(y))
    else: # cos(y) (close to) zero, so x -> 0.0 (see above)
        # so r21 -> sin(z), r22 -> cos(z) and
        z = math.atan2(r21,  r22)
        y = math.atan2(r13,  cy) # atan2(sin(y), cy)
        x = 0.0
    return z, y, x

def rotation_from_matrix(matrix):
    """Return rotation angle and axis from rotation matrix.

    >>> angle = (random.random() - 0.5) * (2*math.pi)
    >>> direc = np.random.random(3) - 0.5
    >>> point = np.random.random(3) - 0.5
    >>> R0 = rotation_matrix(angle, direc, point)
    >>> angle, direc, point = rotation_from_matrix(R0)
    >>> R1 = rotation_matrix(angle, direc, point)
    >>> is_same_transform(R0, R1)
    True

    """
    R = np.array(matrix, dtype=np.float64, copy=False)
    R33 = R[:3, :3]
    # direction: unit eigenvector of R33 corresponding to eigenvalue of 1
    w, W = np.linalg.eig(R33.T)
    i = np.where(abs(np.real(w) - 1.0) < 1e-4)[0]
    if not len(i):
        raise ValueError("no unit eigenvector corresponding to eigenvalue 1")
    direction = np.real(W[:, i[-1]]).squeeze()
    # point: unit eigenvector of R33 corresponding to eigenvalue of 1
    w, Q = np.linalg.eig(R)
    i = np.where(abs(np.real(w) - 1.0) < 1e-4)[0]
    if not len(i):
        raise ValueError("no unit eigenvector corresponding to eigenvalue 1")
    point = np.real(Q[:, i[-1]]).squeeze()
    point /= point[3]
    # rotation angle depending on direction
    cosa = (np.trace(R33) - 1.0) / 2.0
    if abs(direction[2]) > 1e-8:
        sina = (R[1, 0] + (cosa-1.0)*direction[0]*direction[1]) / direction[2]
    elif abs(direction[1]) > 1e-8:
        sina = (R[0, 2] + (cosa-1.0)*direction[0]*direction[2]) / direction[1]
    else:
        sina = (R[2, 1] + (cosa-1.0)*direction[1]*direction[2]) / direction[0]
    angle = math.atan2(sina, cosa)
    return angle, direction, point


def nextpow2(i):
    '''
    Find 2^n that is equal to or greater than

    :param i: arbitrary non-negative integer
    :return: integer with power of two
    '''
    n = 1
    while n < i: n *= 2
    return n


def load_dcm(folder_name):
    import os
    init_volume = False
    for file in os.listdir(folder_name):
        if file.endswith(".dcm"):
            try:
                print('loading file "%s"...' % file)
                ds = dicom.read_file('%s\\%s' % (folder_name, file))
                if not init_volume:
                    volume = ds.pixel_array
                    init_volume = True
                else:
                    volume = np.dstack((volume, ds.pixel_array))
            except ValueError:
                print('skipping file "%s"...' % file)

    return volume
