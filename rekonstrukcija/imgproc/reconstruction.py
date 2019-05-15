import os
import numpy as np
import SimpleITK as itk
import nrrd

from os.path import join
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interpn
from PIL import Image as im

import imgproc.imlib as imlib
import imgproc.functions as f
import calibration


# def load_images(pth, proc=imlib.rgb2gray, ext='.jpg'):
#     imgs, angles = [], []
#     for file in os.listdir(pth):
#         if file.endswith(ext):
#             try:
#                 print('Loading file "{}"...'.format(file))
#                 angles.append(float(file.replace(ext, '').split(sep='_')[-1]))
#                 imgs.append(proc(np.array(im.open(join(pth, file)))))
#             except ValueError:
#                 print('Error while reading file "{}", skipping.'.format(file))
#     idx = np.argsort(angles)
#     imgs = [imgs[i] for i in idx]
#     angles = [angles[i] for i in idx]
#     return imgs, angles

def load_images(pth, proc=imlib.rgb2gray, ext='.jpg'):
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

def show_points_in_3d(pts):
    # extract coordinates    
    X, Y, Z = pts[:, 0], pts[:, 1], pts[:, 2]
    # draw points in 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X, Y, Z, c='m', marker='o')
   
    max_range = np.array([
            X.max() - X.min(), 
            Y.max() - Y.min(), 
            Z.max() - Z.min()]
        ).max() / 2.0

    mean_x = X.mean()
    mean_y = Y.mean()
    mean_z = Z.mean()
    ax.set_xlim(mean_x - max_range, mean_x + max_range)
    ax.set_ylim(mean_y - max_range, mean_y + max_range)
    ax.set_zlim(mean_z - max_range, mean_z + max_range)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    plt.show()

def annotate_caliber_image(img, filename, n=8):
    pts_ann = []
    plt.close('all')
    imlib.showImage(img, iTitle='Oznaci sredisca krogel na sliki!')
    pts_ann.append(plt.ginput(n, timeout=-1))
    np.save(filename, pts_ann)

def dlt_calibration(pts2d, pts3d):
    '''
    Perform DLT camera calibration based on input points

    :param pts2d: Nx2 array of (x,y) coordinates.
    :param pts3d: Nx3 array of (x,y,z) coordinates.
    :return: Transformation matrix
    '''

    def get_mat_row(pt2d, pt3d):
        row1 = np.array((pt3d[0], pt3d[1], pt3d[2], 1, 0, 0, 0, 0, -pt3d[0]*pt2d[0], -pt3d[1]*pt2d[0], -pt3d[2]*pt2d[0]))
        row2 = np.array((0, 0, 0, 0, pt3d[0], pt3d[1], pt3d[2], 1, -pt3d[0]*pt2d[1], -pt3d[1]*pt2d[1], -pt3d[2]*pt2d[1]))
        return np.vstack((row1, row2)), np.vstack((pt2d[0], pt2d[1]))

    dmat = np.zeros((0, 11))
    dvec = np.zeros((0, 1))
    for i in range(pts2d.shape[0]):
        # print(dmat.shape)
        # print(get_mat_row(ptsW[i,:], ptsU[i,:]).shape)
        dmatp, dvecp = get_mat_row(pts2d[i, :], pts3d[i, :])
        dmat = np.vstack((dmat, dmatp))
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
        calibration.CAMERA_TO_TABLE_DX_MM,
        calibration.CAMERA_TO_TABLE_DY_MM,
        -calibration.CAMERA_HEIGHT_MM+calibration.TABLE_HEIGHT_MM
    ]
    Ttable = f.transRigid3D(trans=cam2table)

    # position points in space
    # pts3dcam = np.array((0,0,0,1))
    pts3dh = np.hstack((pts3d, np.ones((np.size(pts3d, 0), 1))))
    pts3dht = np.dot(np.dot(Ttable, f.transRigid3D(rot=(0, 0, -np.pi))), np.transpose(pts3dh))
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
    filt_len = np.max([64, f.nextpow2(2 * nu)])
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
        voldims=(calibration.VOLUME_LENGTH, calibration.VOLUME_WIDTH, calibration.VOLUME_HEIGHT),
        sampling_mm=sampling_mm
    )

    # initialize volume
    vol = np.zeros(grid_size)
    xs, ys, zs = grid_size

    for i in range(len(angles) - 1):
        # display current status
        print("processing image #%d/%d" % (i + 1, len(angles)))

        # normalize image
        img_t = imlib.rgb2gray(imgs[i]).astype('float')
        #    img_t = (img_t - np.min(img_t)) / (np.max(img_t) - np.min(img_t))
        img_t = (img_t - np.mean(img_t)) / (np.std(img_t))

        # filter projection image
        img_f = filter_projection(img_t, filter_type, cut_off=cut_off, axis=1)
        img_f = (img_f - np.mean(img_f)) / (np.std(img_f))

        # define function to put points in reference space
        get_grid_at_angle = lambda ang: \
            np.dot(np.dot(Tproj[1], f.transRigid3D(trans=(0, 0, 0), rot=(0, 0, ang))),
                   np.transpose(grid_raw)).transpose()

        # project points to imaging plane
        grid = np.dot(get_grid_at_angle(f.deg2rad(calibration.ROTATION_DIRECTION*angles[i])), Tproj[0].transpose())
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

