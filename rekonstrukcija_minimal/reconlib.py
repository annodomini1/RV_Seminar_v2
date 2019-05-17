import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interpn
import SimpleITK as itk
import nrrd
import matplotlib.cm as cm
import PIL.Image as im
from os.path import join

def rgb2gray(img):
    '''
    Convert RGB image to grayscale.

    :param img: Input RGB image. 
    :return: Grayscale image.
    '''
    if img.ndim == 3:
        return np.mean(img.astype('float'), axis=-1)
    return img

def load_images(pth, proc=rgb2gray, ext='.jpg'):
    imgs, angles = [], []
    for file in os.listdir(pth):
        if file.endswith(ext):
            try:
                print('Loading file "{}"...'.format(file))
                angles.append(float(file.replace(ext, '').split('_')[-1]))
                imgs.append(proc(np.array(im.open(join(pth, file)))))
            except ValueError:
                print('Error while reading file "{}", skipping.'.format(file))
    idx = np.argsort(angles)
    imgs = [imgs[i] for i in idx]
    angles = [angles[i] for i in idx]
    return imgs, angles

#kalibracija
# IRCT CALIBRATION DATA
# Use a metering device to measure the geometric setup of the infrared CT system, namely the distances from the camera
# to the rotating table base.
TABLE_HEIGHT_MM = 79 # rotation table height in mm
CAMERA_HEIGHT_MM = 310 # distance in mm, in direction perpedicular to base, i.e. vertical axis
#CAMERA_TO_TABLE_DX_MM = 530 # distance in mm, in direction from camera towards the rotating table
CAMERA_TO_TABLE_DX_MM = 440
CAMERA_TO_TABLE_DY_MM = 10 # distance in mm, direction based on rhs coordinate frame (dx x dz), i.e. to the left w.r.t. dx

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
    #h = 8
    h = 90
    #x kaže v smeri od kalibracijskega objekta do kamere
    #z kaže navzgor
    #y kaže tako, da imamo desnosučni KS
    pts = [
        [0,0,134+h], #[x,y,z]
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

def showImage(iImage, iTitle='', iCmap=cm.Greys_r):
    '''
    Prikaze sliko iImage in jo naslovi z iTitle

    Parameters
    ----------
    iImage : numpy.ndarray
        Vhodna slika 
    iTitle : str 
        Naslov za sliko
    iCmap : colormap 
        Barvna lestica za prikaz sivinske slike        

    Returns
    ---------
    None

    '''
    plt.figure()  # odpri novo prikazno okno

    # if iImage.ndim == 3:
    #     iImage = np.transpose(iImage, [1, 2, 0])

    plt.imshow(iImage, cmap=iCmap)  # prikazi sliko v novem oknu
    plt.suptitle(iTitle)  # nastavi naslov slike
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axes().set_aspect('equal', 'datalim')  # konstantno razmerje pri povecevanju, zmanjsevanju slike

def annotate_caliber_image(img, filename, n=8):
    pts_ann = []
    plt.close('all')
    showImage(img, iTitle='Oznaci sredisca krogel na sliki!')
    pts_ann.append(plt.ginput(n, timeout=-1))
    np.save(filename, pts_ann)

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

def dlt_calibration(pts2d, pts3d):
    '''
    Perform DLT camera calibration based on input points

    :param pts2d: Nx2 array of (x,y) coordinates.
    :param pts3d: Nx3 array of (x,y,z) coordinates.
    :return: Transformation matrix
    '''

    def get_mat_row(pt2d, pt3d): #vhod v funkcijo je posamezna točka v koordinatah (x,y,z)
        row1 = np.array((pt3d[0], pt3d[1], pt3d[2], 1, 0, 0, 0, 0, -pt3d[0]*pt2d[0], -pt3d[1]*pt2d[0], -pt3d[2]*pt2d[0]))
        row2 = np.array((0, 0, 0, 0, pt3d[0], pt3d[1], pt3d[2], 1, -pt3d[0]*pt2d[1], -pt3d[1]*pt2d[1], -pt3d[2]*pt2d[1]))
        return np.vstack((row1, row2)), np.vstack((pt2d[0], pt2d[1]))

    dmat = np.zeros((0, 11))
    dvec = np.zeros((0, 1))
    for i in range(pts2d.shape[0]):
        # print(dmat.shape)
        # print(get_mat_row(ptsW[i,:], ptsU[i,:]).shape)
        dmatp, dvecp = get_mat_row(pts2d[i, :], pts3d[i, :]) #notri da korespondenčni točki
        dmat = np.vstack((dmat, dmatp)) #ubistu appendamo v matriko matrike dmatp
        dvec = np.vstack((dvec, dvecp))
    return dmat, dvec

def calibrate_irct(pts2d, pts3d):
    '''
    Geometrically calibrate the IRCT system

    :param pts2d: Nx2 array of (x,y) coordinates.
    :param pts3d: Nx3 array of (x,y,z) coordinates.
    :return: 
    '''
    # setup transformations
    cam2table = [
        CAMERA_TO_TABLE_DX_MM,
        CAMERA_TO_TABLE_DY_MM,
        -CAMERA_HEIGHT_MM+TABLE_HEIGHT_MM
    ]
    Ttable = transRigid3D(trans=cam2table)

    # position points in space
    # pts3dcam = np.array((0,0,0,1))
    pts3dh = np.hstack((pts3d, np.ones((np.size(pts3d, 0), 1))))
    pts3dht = np.dot(np.dot(Ttable, transRigid3D(rot=(0, 0, -np.pi))), np.transpose(pts3dh))
    pts3dht = np.transpose(pts3dht)

    # sensor_size = 25.4/4 # 1/4" is the sensor size
    # pixel_size = (25.4/4) / np.sqrt(float(imgs[0].shape[0]**2 + imgs[0].shape[1]**2))

    # perform dlt calibration
    A, b = dlt_calibration(pts2d, pts3dht)
    D = np.linalg.lstsq(A, b)

    Tproj = np.reshape(np.vstack((D[0], 1.0)), (3, 4))

    ptsXYZproj = np.dot(pts3dht, Tproj.transpose())
    # to homogeneous coordinates
    ptsXYZproj[:, 0] = ptsXYZproj[:, 0] / ptsXYZproj[:, 2]
    ptsXYZproj[:, 1] = ptsXYZproj[:, 1] / ptsXYZproj[:, 2]

    return ((Tproj, Ttable), ptsXYZproj)

def ramp_flat(n):
    '''
    Create 1D ramp filter in the spatial domain.

    :param n: Size of the filter, should be of power of 2. 
    :return: Ramp filter response vector.
    '''
    nn = np.arange(-(n // 2), (n // 2))
    h = np.zeros((nn.size,), dtype='float')
    h[n // 2] = 1 / 4
    odd = np.mod(nn, 2) == 1
    h[odd] = -1.0 / (np.pi * nn[odd]) ** 2.0;
    return h, nn

def nextpow2(i):
    '''
    Find 2^n that is equal to or greater than

    :param i: arbitrary non-negative integer
    :return: integer with power of two
    '''
    n = 1
    while n < i: n *= 2
    return n

def create_filter(filter_type, ramp_kernel, order, d):
    '''
    Create 1D filter of selected type.

    :param filter_type: Select filter by name, e.g. "ram-lak", "shepp-logan", "cosine", "hamming", "hann". 
    :param ramp_kernel: Input ramp filter kernel, size defined by input image size.
    :param order: Filter order, should be of power of 2.
    :param d: Cut-off frequency (0-1).
    :return: Desired filter response vector.
    '''
    f_kernel = np.abs(np.fft.fft(ramp_kernel)) * 2  # transform ramp filter to freqency domain
    filt = f_kernel[0:order // 2 + 1].transpose()
    w = 2.0 * np.pi * np.arange(0, filt.size) / order  # frequency axis up to Nyquist

    filter_type = filter_type.lower()
    if filter_type == 'ram-lak':
        # do nothing
        None
    elif filter_type == 'shepp-logan':
        # be careful not to divide by 0:
        filt[1:] = filt[1:] * (np.sin(w[1:] / (2 * d)) / (w[1:] / (2 * d)))
    elif filter_type == 'cosine':
        filt[1:] = filt[1:] * np.cos(w[1:] / (2 * d))
    elif filter_type == 'hamming':
        filt[1:] = filt[1:] * (.54 + .46 * np.cos(w[1:] / d))
    elif filter_type == 'hann':
        filt[1:] = filt[1:] * (1 + np.cos(w[1:] / d)) / 2
    else:
        raise ValueError('filter_type: invalid filter selected "{}"'.format(filter_type))

    filt[w > np.pi * d] = 0  # crop the frequency response
    filt = np.hstack((filt, filt[-2:0:-1]))  # enforce symmetry of the filter

    return filt

def get_volume(voldims=(100, 100, 100), sampling_mm=1):
    '''
    Define volume size and sampling points.
    
    :param voldims: Define volume size in mm (for X x Y x Z axes).
    :param sampling_mm: Volume sampling step in mm. For anisotropic sampling define a tuple or list.
    :return: Grid of points in the volume (in homogeneous coordinates) in "grid" and grid sizes (X x Y x Z) in "volsiz". 
    '''
    if not isinstance(sampling_mm, (tuple, list)):
        sampling_mm = [sampling_mm] * 3
    # get sampling points; the grid is axis aligned, and defined wrt to reference
    # point on top of the caliber, which is determined by imaging the caliber
    xr = np.arange(-voldims[0] / 2, voldims[0] / 2, sampling_mm[0])
    yr = np.arange(-voldims[1] / 2, voldims[1] / 2, sampling_mm[1])
    zr = np.arange(0, voldims[2], sampling_mm[2])
    xg, yg, zg = np.meshgrid(xr, yr, zr, indexing='xy')
    # store volume shape
    grid_size = xg.shape
    # define matrix of homogeneous point coordinates
    grid = np.vstack((xg.flatten(), yg.flatten(), zg.flatten(), np.ones_like(xg.flatten()))).transpose()
    return grid, grid_size

def filter_projection(proj, filter_type='hann', cut_off=1, axis=0):
    '''
    Filter projection image using ramp-like filter. Be careful to select the filter axis, which is (corresponding to 
    the rotation axis.

    :param proj: Projection image (u x v)
    :param filter_type: Select filter by name, e.g. "ram-lak", "shepp-logan", "cosine", "hamming", "hann".
    :param cut_off: Cut-off frequency (0-1).
    :param axis: Select axis 0/1 to apply the filter.
    :return: Filtered projection image. 
    '''
    if filter_type == 'none':
        return proj

    if axis == 1:
        proj = proj.transpose()

    nu, nv = proj.shape
    filt_len = np.max([64, nextpow2(2 * nu)])
    ramp_kernel, nn = ramp_flat(filt_len)

    filt = create_filter(filter_type, ramp_kernel, filt_len, cut_off)
    filt = np.tile(filt[:, np.newaxis], nv)

    # append zeros
    fproj = np.zeros((filt_len, nv), dtype='float')
    fproj[filt_len // 2 - nu // 2:filt_len // 2 + nu // 2, :] = proj

    # filter using fourier theorem
    fproj = np.fft.fft(fproj, axis=0)
    fproj = fproj * filt
    fproj = np.real(np.fft.ifft(fproj, axis=0))
    fproj = fproj[filt_len // 2 - nu // 2:filt_len // 2 + nu // 2, :]

    if axis == 1:
        fproj = fproj.transpose()

    return fproj

def deg2rad(ang):
    '''
    Convert angle in degrees to radians.

    :param ang: Angle in degrees. 
    :return: Angle in radians.
    '''
    return ang * np.pi / 180.0

def fbp(imgs, angles, Tproj, out_fname='volume', sampling_mm=2, filter_type='hann', cut_off=0.75):
    '''
    Filtered backprojection 3D reconstruction.
    
    :param imgs: List of projection images for reconstruction.
    :param angles: List of relative rotation angles corresponding to each image. 
    :param Tproj: Transformation matrix obtained from imaging the calibration object.
    :param out_fname: Filename for output reconstructed 3D volume. 
    :param sampling_mm: Volume sampling step in mm. For anisotropic sampling define a tuple or list.
    :param filter_type: Select filter by name, e.g. "ram-lak", "shepp-logan", "cosine", "hamming", "hann".
    :return: Volume of reconstructed 3D grayscale image.
    '''

    # debug: show result
    # i = 0; rvlib.showImage(imgs[i], iTitle='image #%d' % i)
    # plt.plot(grid[:,0], grid[:,1],'rx')
    if not isinstance(sampling_mm, (tuple, list)):
        sampling_mm = [sampling_mm] * 3

    # get sampling points in homogeneous coordinates
    grid_raw, grid_size = get_volume(
        voldims=(VOLUME_LENGTH, VOLUME_WIDTH, VOLUME_HEIGHT),
        sampling_mm=sampling_mm
    )

    # initialize volume
    vol = np.zeros(grid_size)
    xs, ys, zs = grid_size

    for i in range(len(angles) - 1):
        # display current status
        print("processing image #%d/%d" % (i + 1, len(angles)))

        # normalize image
        img_t = rgb2gray(imgs[i]).astype('float')
        #    img_t = (img_t - np.min(img_t)) / (np.max(img_t) - np.min(img_t))
        img_t = (img_t - np.mean(img_t)) / (np.std(img_t))

        # filter projection image
        img_f = filter_projection(img_t, filter_type, cut_off=cut_off, axis=1)
        img_f = (img_f - np.mean(img_f)) / (np.std(img_f))

        # define function to put points in reference space
        get_grid_at_angle = lambda ang: \
            np.dot(np.dot(Tproj[1], transRigid3D(trans=(0, 0, 0), rot=(0, 0, ang))),
                   np.transpose(grid_raw)).transpose()

        # project points to imaging plane
        grid = np.dot(get_grid_at_angle(deg2rad(ROTATION_DIRECTION*angles[i])), Tproj[0].transpose())
        grid[:, 0] = grid[:, 0] / grid[:, 2]
        grid[:, 1] = grid[:, 1] / grid[:, 2]
        grid[:, 2] = 1

        # correct in-plane errors due to incorrect geometry
        #    grid = np.dot(grid, Tcorr[i].transpose())

        #    plt.close('all')
        #    rvlib.showImage(img_t, iTitle='original grid')
        #    plt.plot(grid[:,0], grid[:,1],'rx')
        #
        # rvlib.showImage(img_t, iTitle='corrected grid')
        # plt.plot(grid[:,0], grid2[:,1],'rx')

        # rvlib.showImage(img_f)

        # interpolate points to obtain backprojected volume
        us, vs = img_f.shape
        img_backprojected = interpn((np.arange(vs), np.arange(us)), img_f.transpose(),
                                    grid[:, :2], method='linear', bounds_error=False)
        img_backprojected = img_backprojected.reshape((xs, ys, zs))
        vol = vol + img_backprojected
        vol[np.isnan(vol)] = 0

    print('Writing volume to file "{}.nrrd"...'.format(out_fname))
    if os.path.splitext(out_fname)[-1] != '.nrrd':
        out_fname = '{}.nrrd'.format(out_fname)

    img = itk.GetImageFromArray(np.transpose(vol, [2,1,0]))
    img.SetSpacing(sampling_mm)
    itk.WriteImage(img, out_fname, True)

    return vol

def thresholdImage(iImage, iThreshold):
    oImage = 255 * np.array(iImage > iThreshold, dtype='uint8')
    return oImage

# def get_point_cloud(vol, Thres=15, Deci=5, startHeightShare=0.1, endHeightShare=0.9):

#     # def linScale(iImage, oMax):
#     #     k = oMax/np.max(iImage)
#     #     oImage = k*iImage
#     #     return oImage

#     pointCoorX = []
#     pointCoorY = []
#     pointCoorZ = []
#     dvol = np.ones_like(vol)

#     dZ = len(vol[0,0,:])
#     endZ = int(np.round(dZ*endHeightShare))
#     startZ = int(np.round(dZ*startHeightShare))

#     vol = vol[:,:,startZ:endZ]
#     [dx, dy, dz] = vol.shape

#     for z in range(dz):
#         dImage = vol[:,:,z]
#         #dImage = linScale(dImage, 255)
#         dImage = thresholdImage(dImage, np.max(dImage)/2)

#         for x in range(dx):
#             for y in range(dy):
#                 dvol[x,y,z] = dImage[x,y]

#                 if (dImage[x, y] == 0):
#                     pointCoorX.append(x)
#                     pointCoorY.append(y)
#                     pointCoorZ.append(z)

#     #redcenje tock
#     pointCoorX = pointCoorX[::Deci]
#     pointCoorY = pointCoorY[::Deci]
#     pointCoorZ = pointCoorZ[::Deci]
#     return pointCoorX, pointCoorY, pointCoorZ

def get_point_cloud(vol, ThresImageMaxShare=0.3, Deci=5, startHeightShare=0.1, endHeightShare=0.9):

    pointCoorX = []
    pointCoorY = []
    pointCoorZ = []

    initZ = len(vol[0,0,:])

    startZ = int(np.round(startHeightShare*initZ))
    endZ = int(np.round(endHeightShare*initZ))
    vol = vol[:,:,startZ:endZ]

    [dx, dy, dz] = vol.shape
    for dZ in range(dz):
        dImage = vol[:,:,dZ]
        dImage = dImage + abs(np.min(dImage))
        dImage = thresholdImage(dImage, np.max(dImage)*ThresImageMaxShare)
        for dX in range(dx):
            for dY in range(dy):
                if (dImage[dX, dY] == 0):
                    pointCoorX.append(dX)
                    pointCoorY.append(dY)
                    pointCoorZ.append(dZ)

    pointCoorX = pointCoorX[::Deci]
    pointCoorY = pointCoorY[::Deci]
    pointCoorZ = pointCoorZ[::Deci]

    return pointCoorX, pointCoorY, pointCoorZ


def plot_point_cloud(X, Y, Z):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_aspect('equal')

    scat = ax.scatter(X, Y, Z)

    # Create cubic bounding box to simulate equal aspect ratio
    max_range = np.max(np.array([np.max(X) - np.min(X), np.max(Y) - np.min(Y), np.max(Z) - np.min(Z)]))
    Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(np.max(X) - np.min(X))
    Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(np.max(Y) - np.min(Y))
    Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(np.max(Z) - np.min(Z))
    # Comment or uncomment following both lines to test the fake bounding box:
    for xb, yb, zb in zip(Xb, Yb, Zb):
        ax.plot([xb], [yb], [zb], 'w')

    plt.grid()
    plt.show()

def crop_image(iImageArray, pxFromRight, pxFromLeft, pxFromUp, pxFromDown):
    #original iz maina
    # # obrezovanje
    # slika_x, slika_y = slike[0].shape
    # nslike = []
    # for sl in range(len(slike)):
    #     dslika =  slike[sl]
    #     dslika = dslika[200:slika_x-100, 300:slika_y-300]
    #     nslike.append(dslika)

    # slike = nslike
    slika_x, slika_y = iImageArray[0].shape
    oImageArray = []
    for sl in range(len(iImageArray)):
        dImage =  iImageArray[sl]
        dImage = dImage[pxFromUp:slika_x-pxFromDown, pxFromLeft:slika_y-pxFromRight]
        oImageArray.append(dImage)

    return oImageArray