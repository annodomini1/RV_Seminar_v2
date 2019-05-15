import os
import numpy as np
import nrrd

from os.path import join
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interpn
from PIL import Image as im

import imgproc.imlib as imlib
import calibration


def deg2rad(ang):
    '''
    Convert angle in degrees to radians.
    
    :param ang: Angle in degrees. 
    :return: Angle in radians.
    '''
    return ang*np.pi/180.0


def rad2deg(ang):
    '''
    Convert angle in radians to degrees.

    :param ang: Angle in radians. 
    :return: Angle in degrees.
    '''
    return ang * np.pi / 180.0


def rgb2gray(img):
    '''
    Convert RGB image to grayscale.
    
    :param img: Input RGB image. 
    :return: Grayscale image.
    '''
    if img.ndim == 3:
        return np.mean(img.astype('float'), axis=-1)
    return img


def crop_image(img, crop=(150, 800, 2250, 2000)):
    '''
    Crop image.
    
    :param img: Crop input grayscale image. 
    :param crop: Region to crop defined by (lx,uy,rx,by). 
    :return: Cropped image.
    '''
    # return np.array(img[crop[0]:crop[2], crop[1]:crop[3]])
    return np.array(img[crop[1]:crop[3], crop[0]:crop[2]])


def load_images(pth, proc=rgb2gray, ext='.jpg'):
    imgs, angles = [], []
    for file in os.listdir(pth):
        if file.endswith(ext):
            try:
                print('Loading file "{}"...\r'.format(file))
                angles.append(float(file.replace(ext,'').split(sep='_')[-1]))
                imgs.append(proc(np.array(im.open(join(pth, file)))))
            except ValueError:
                print('Error while reading file "{}", skipping.'.format(file))
    idx = np.argsort(angles)
    imgs = [imgs[i] for i in idx]
    angles = [angles[i] for i in idx]
    
    return imgs, angles

#%% load images into workspace
#imgs = []
#for i in range(1,23):
#    imgs.append(np.array(im.open("./caliber/caliber-%d.jpg" % i)))    

#for i in range(1,22):
#    imgs.append(np.array(im.open("./phantom/phantom-%d.jpg" % i)))

# show images
#rvlib.showImage(imgs[14])
#rvlib.showImage(imgs[15])
# check difference between first and last (which should be nearly the same)
#rvlib.showImage(np.abs(np.array(imgs[0], dtype='float') - np.array(imgs[-1], dtype='float')), iCmap=cm.jet)

#%% check pairwise differences, remove duplicates and define rotation angles
#i = 0
#rvlib.plt.close('all')
#title = "displaying image pair %d-%d" % (i, i+1)
#rvlib.showImage(np.abs(np.array(imgs[i], dtype='float') - \
#    np.array(imgs[i+1], dtype='float')), iCmap=cm.jet, iTitle=title)
#i = i+1

# 
# 16-17 are duplicated, thus remove image at index 16 or 17
#imgs = imgs[:16] + imgs[17:]
#idx = list(range(0,22))
#idx.pop(16)

# define rotation angles with respect to first image
#angle_step = 360 / len(imgs)
#angles = angle_step * np.arange(len(imgs))

#%% define grid of points on the calibration object
def get_calibration_object():
    # define basic measurements
    d = 18.5 # ball diameter
    dz = 22.0 # ball height from base
    dh = np.round((131 - 12)/15) # distance between lego holes
    df = 20 + 1 # center to based of attachement frame + 1 mm gap
    
    # reference point is the top ball
    b0 = np.array((0, 0, 0))
    #define top frame center
    p_top_frame = np.array((0, 0, - d/2 - (dz-d) - dh/2))
    # on one side of the frame (first ball is second highest)
    d_fb = df + d/2 + (dz-d) # distance from base
    b1 = p_top_frame + np.array((d_fb, 0, -dh))
    b4 = p_top_frame + np.array((d_fb, 0, -7*dh))
    b7 = p_top_frame + np.array((d_fb, 0, -13*dh))
    # on the other side of the frame (first ball is third highest)
    b2 = p_top_frame + np.array((-d_fb*np.cos(np.pi/3),-d_fb*np.sin(np.pi/3),-3*dh))
    b5 = p_top_frame + np.array((-d_fb*np.cos(np.pi/3),-d_fb*np.sin(np.pi/3),-9*dh))
    # on the other side of the frame (first ball is fourth highest)
    b3 = p_top_frame + np.array((-d_fb*np.cos(np.pi/3),d_fb*np.sin(np.pi/3),-5*dh))
    b6 = p_top_frame + np.array((-d_fb*np.cos(np.pi/3),d_fb*np.sin(np.pi/3),-11*dh))
    # define bottom frame center and ground
    p_bottom_frame = p_top_frame + (0, 0, -15*dh-4-8-4)
    p_bottom_ground = p_bottom_frame + (0, 0, -4-8-4)
    
    # create array of points
    pts = np.vstack((b0,b1,b2,b3,b4,b5,b6,b7))
    
    pts = np.array(((0,0,251),\
    (-48.5,84,234),(-48.5,84,162),(-48.5,84,59),\
    (97,0,234),(97,0,162),(97,0,59),\
    (-48.5,-84,234),(-48.5,-84,162),(-48.5,-84,59)))
    
    pts = pts - np.array(((0,0,251)))
    
    return pts

def show_points_in_3d(pts):
    # draw point in 3D

#    plt.close('all')
    fig = plt.figure()    
    ax = fig.add_subplot(111,projection='3d')
    ax.scatter(pts[:,0],pts[:,1],pts[:,2], c='m', marker='o')
    
    X = pts[:,0]; Y = pts[:,1]; Z = pts[:,2]
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
    
    mean_x = X.mean()
    mean_y = Y.mean()
    mean_z = Z.mean()
    ax.set_xlim(mean_x - max_range, mean_x + max_range)
    ax.set_ylim(mean_y - max_range, mean_y + max_range)
    ax.set_zlim(mean_z - max_range, mean_z + max_range)
    
    plt.show()

#pts3d = np.copy(pts)

#%% draw complete geometry with camera and caliber



#%% find ball center coordinates in all images

#%% annotate caliber image
def annotate_caliber_image(img, filename, n=8):
    pts_ann = []
    plt.close('all')
    imlib.showImage(img, iTitle='Oznaci sredisca krogel na sliki!')
    pts_ann.append(plt.ginput(n, timeout=-1))
    np.save(filename, pts_ann)
#    plt.plot(pts_ann[0][:,0], pts_ann[0][:,1],'mx',markersize=15)
    
# correct single entry
#    rvlib.showImage(imgs[2], iTitle='image #%d' % 2)
#    pts_ann2 = plt.ginput(n=8, timeout=-1)

#%% show all annotations
#import time
##pts = np.load('pts_ann.npy')
##pts = np.load('pts_ann_phantomInCenter.npy')
#pts = np.load('pts_ann_phantomInCenter-crop2.npy')
##for i in range(len(imgs)):
#i = 0
#plt.close('all')
#rvlib.showImage(imgs[i], iTitle='image #%d' % i)
#plt.plot(pts[i][:,0], pts[i][:,1],'mx')
#plt.show()
##    time.sleep(2)
##    wait = input("PRESS ENTER TO CONTINUE.")
#
#p_0 = np.squeeze(np.array([np.hstack(p[0,:]) for p in pts]))
#p_1 = np.squeeze(np.array([np.hstack(p[1,:]) for p in pts]))
#p_2 = np.squeeze(np.array([np.hstack(p[2,:]) for p in pts]))
#p_3 = np.squeeze(np.array([np.hstack(p[3,:]) for p in pts]))
#p_4 = np.squeeze(np.array([np.hstack(p[4,:]) for p in pts]))
#p_5 = np.squeeze(np.array([np.hstack(p[5,:]) for p in pts]))
#p_6 = np.squeeze(np.array([np.hstack(p[6,:]) for p in pts]))
#p_7 = np.squeeze(np.array([np.hstack(p[7,:]) for p in pts]))
#
#p_0_mean = np.mean(p_0, axis=0)
#p_1_mean = np.mean(p_1, axis=0)
#p_2_mean = np.mean(p_2, axis=0)
#p_3_mean = np.mean(p_3, axis=0)
#p_4_mean = np.mean(p_4, axis=0)
#p_5_mean = np.mean(p_5, axis=0)
#p_6_mean = np.mean(p_6, axis=0)
#p_7_mean = np.mean(p_7, axis=0)
#
#
#i = 0    
#rvlib.showImage(imgs[i], iTitle='image #%d' % i)
#plt.plot(p_0[:,0], p_0[:,1],'rx', markersize=3)
#plt.plot(p_1[:,0], p_1[:,1],'rx', markersize=3)
#plt.plot(p_2[:,0], p_2[:,1],'rx', markersize=3)
#plt.plot(p_3[:,0], p_3[:,1],'rx', markersize=3)
#plt.plot(p_4[:,0], p_4[:,1],'rx', markersize=3)
#plt.plot(p_5[:,0], p_5[:,1],'rx', markersize=3)
#plt.plot(p_6[:,0], p_6[:,1],'rx', markersize=3)
#plt.plot(p_7[:,0], p_7[:,1],'rx', markersize=3)
#
#
#plt.plot(p_0_mean[0], p_0_mean[1],'gx', markersize=5)
#plt.plot(p_1_mean[0], p_1_mean[1],'gx', markersize=5)
#plt.plot(p_2_mean[0], p_2_mean[1],'gx', markersize=5)
#plt.plot(p_3_mean[0], p_3_mean[1],'gx', markersize=5)
#plt.plot(p_4_mean[0], p_4_mean[1],'gx', markersize=5)
#plt.plot(p_5_mean[0], p_5_mean[1],'gx', markersize=5)
#plt.plot(p_6_mean[0], p_6_mean[1],'gx', markersize=5)
#plt.plot(p_7_mean[0], p_7_mean[1],'gx', markersize=5)


## draw point in 3D
#from mpl_toolkits.mplot3d import Axes3D
#plt.close('all')
#fig = plt.figure()    
#ax = fig.add_subplot(111,projection='3d')
#ax.scatter(pts3dcam[0],pts3dcam[1],pts3dcam[2], c='r', marker='o')
#ax.scatter(pts3dht[:,0],pts3dht[:,1],pts3dht[:,2], c='m', marker='o')
#
#X = np.hstack((pts3dcam,pts3dht[:,0]))
#Y = np.hstack((pts3dcam,pts3dht[:,1]))
#Z = np.hstack((pts3dcam,pts3dht[:,2]))
#
#max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
#
#mean_x = X.mean()
#mean_y = Y.mean()
#mean_z = Z.mean()
#ax.set_xlim(mean_x - max_range, mean_x + max_range)
#ax.set_ylim(mean_y - max_range, mean_y + max_range)
#ax.set_zlim(mean_z - max_range, mean_z + max_range)
#
#plt.show()


#%% draw all point in space
#getPts3D = lambda ang: np.dot(np.dot(Ttable, Trigid(rot=(0,0,-np.pi+ang))), np.transpose(pts3dh)).transpose()
#
#pts_all = np.zeros((0,4))
#for angle in angles:
#    pts_all = np.vstack((pts_all, getPts3D(deg2rad(- angle))))
#    
#
## draw points in 3D
#from mpl_toolkits.mplot3d import Axes3D
#plt.close('all')
#fig = plt.figure()    
#ax = fig.add_subplot(111,projection='3d')
#ax.scatter(pts3dcam[0],pts3dcam[1],pts3dcam[2], c='r', marker='o')    
#ax.scatter(pts_all[:,0],pts_all[:,1],pts_all[:,2], c='m', marker='o')
#
#X = np.hstack((pts3dcam,pts_all[:,0]))
#Y = np.hstack((pts3dcam,pts_all[:,1]))
#Z = np.hstack((pts3dcam,pts_all[:,2]))
#
#max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
#
#mean_x = X.mean()
#mean_y = Y.mean()
#mean_z = Z.mean()
#ax.set_xlim(mean_x - max_range, mean_x + max_range)
#ax.set_ylim(mean_y - max_range, mean_y + max_range)
#ax.set_zlim(mean_z - max_range, mean_z + max_range)
#
#plt.show()

#%% position points in space
def transRigid3D(trans=(0, 0, 0), rot=(0, 0, 0)):
    '''
    Rigid body transformation
    
    :param trans: 
    :param rot: 
    :return: 
    '''
    Trotx =  np.array(((1,0,0,0),\
                       (0,np.cos(rot[0]),-np.sin(rot[0]),0),\
                       (0,np.sin(rot[0]),np.cos(rot[0]),0),\
                       (0,0,0,1)))
    Troty =  np.array(((np.cos(rot[1]),0,np.sin(rot[1]),0),\
                       (0,1,0,0),\
                       (-np.sin(rot[1]),0,np.cos(rot[1]),0),\
                       (0,0,0,1)))
    Trotz =  np.array(((np.cos(rot[2]),-np.sin(rot[2]),0,0),\
                       (np.sin(rot[2]),np.cos(rot[2]),0,0),\
                       (0,0,1,0),\
                       (0,0,0,1)))
    Ttrans = np.array(((1,0,0,trans[0]),\
                       (0,1,0,trans[1]),\
                       (0,0,1,trans[2]),\
                       (0,0,0,1)))
    return np.dot(np.dot(np.dot(Trotx, Troty), Trotz), Ttrans)


def dlt_calibration(ptsW, ptsU):
    '''
    Perform DLT camera calibration based on input points
    
    :param ptsW: 
    :param ptsU: 
    :return: 
    '''
    def get_mat_row(ptW, ptU):
        row1 = np.array((ptU[0],ptU[1],ptU[2],1,0,0,0,0,-ptU[0]*ptW[0],-ptU[1]*ptW[0],-ptU[2]*ptW[0]))
        row2 = np.array((0,0,0,0,ptU[0],ptU[1],ptU[2],1,-ptU[0]*ptW[1],-ptU[1]*ptW[1],-ptU[2]*ptW[1]))
        return np.vstack((row1, row2)), np.vstack((ptW[0], ptW[1]))
    dmat = np.zeros((0,11))
    dvec = np.zeros((0,1))
    for i in range(ptsW.shape[0]):
        # print(dmat.shape)
        # print(get_mat_row(ptsW[i,:], ptsU[i,:]).shape)
        dmatp, dvecp = get_mat_row(ptsW[i,:], ptsU[i,:])
        dmat = np.vstack((dmat,dmatp))
        dvec = np.vstack((dvec,dvecp))
    return dmat, dvec

def calibrate_irct(pts2d, pts3d):
    '''
    Geometrically calibrate the IRCT system
    
    :param pts2d: 
    :param pts3d: 
    :return: 
    '''

    cam2table = [
        calibration.CAMERA_TO_TABLE_DX_MM,
        calibration.CAMERA_TO_TABLE_DY_MM,
        calibration.CAMERA_HEIGHT_MM
    ]

    # setup transformations
    cam2table[2] = cam2table[2] - calibration.TABLE_HEIGHT_MM
    Ttable = transRigid3D(trans = cam2table)

    # position points in space
    # pts3dcam = np.array((0,0,0,1))
    pts3dh = np.hstack((pts3d, np.ones((np.size(pts3d,0),1))))
    pts3dht = np.dot(np.dot(Ttable, transRigid3D(rot=(0, 0, -np.pi))), np.transpose(pts3dh))
    pts3dht = np.transpose(pts3dht)

    #sensor_size = 25.4/4 # 1/4" is the sensor size
    #pixel_size = (25.4/4) / np.sqrt(float(imgs[0].shape[0]**2 + imgs[0].shape[1]**2))

    # perform dlt calibration 
    A, b = dlt_calibration(pts2d, pts3dht)
    D = np.linalg.lstsq(A, b)

    Tproj = np.reshape(np.vstack((D[0],1.0)),(3,4))
 
    ptsXYZproj = np.dot(pts3dht, Tproj.transpose())
    # to homogeneous coordinates
    ptsXYZproj[:,0] = ptsXYZproj[:,0] / ptsXYZproj[:,2]
    ptsXYZproj[:,1] = ptsXYZproj[:,1] / ptsXYZproj[:,2]
   
    return ((Tproj, Ttable), ptsXYZproj)

##%% draw all point in the image space based on angles and projection
#
#i = 0    
#rvlib.showImage(imgs[i], iTitle='image #%d' % i)
#plt.plot(p_0[:,0], p_0[:,1],'rx', markersize=3)
#plt.plot(p_1[:,0], p_1[:,1],'rx', markersize=3)
#plt.plot(p_2[:,0], p_2[:,1],'rx', markersize=3)
#plt.plot(p_3[:,0], p_3[:,1],'rx', markersize=3)
#plt.plot(p_4[:,0], p_4[:,1],'rx', markersize=3)
#plt.plot(p_5[:,0], p_5[:,1],'rx', markersize=3)
#plt.plot(p_6[:,0], p_6[:,1],'rx', markersize=3)
#plt.plot(p_7[:,0], p_7[:,1],'rx', markersize=3)
#
#ptsXYZproj = np.dot(pts_all, Tproj.transpose())
#ptsXYZproj[:,0] = ptsXYZproj[:,0] / ptsXYZproj[:,2]
#ptsXYZproj[:,1] = ptsXYZproj[:,1] / ptsXYZproj[:,2]
#plt.plot(ptsW[:,0], ptsW[:,1],'rx')
#plt.plot(ptsXYZproj[:,0], ptsXYZproj[:,1],'gx')

##%% draw point in the image space based on angle and projection
#plt.close('all')
##p_2d = [p_0, p_1, p_2, p_3, p_4, p_5, p_6, p_7];
#p_2d = [pts[it,:,:] for it in range(pts.shape[0]) ]
#p_3d = [np.dot(getPts3D(deg2rad(-angle)), Tproj.transpose()) for angle in angles]
#p_3d = [p_3d[it]/np.tile(p_3d[it][:,2],(3,1)).transpose() for it in range(len(p_3d))]
#
#i=15
#rvlib.showImage(imgs[i], iTitle='image #%d' % i)
#plt.plot(p_2d[i][:,0], p_2d[i][:,1],'rx',markersize=15)
##plt.plot(p_0[:,0], p_0[:,1],'rx')
#plt.plot(p_3d[i][:,0], p_3d[i][:,1],'gx',markersize=15)
#
#du = np.mean(p_2d[i][:,0]-p_3d[i][:,0])
#dv = np.mean(p_2d[i][:,1]-p_3d[i][:,1])
#Tcorr = np.array(((1,0,du),(0,1,dv),(0,0,1)))
#Tcorr = np.array(rvlib.mapAffineApprox2D(p_3d[i], p_2d[i]))
#
#p_3d_corr = np.dot(p_3d[i], Tcorr.transpose())
#
#plt.plot(p_3d_corr[:,0], p_3d_corr[:,1],'mx',markersize=15)
#
#err_before = np.mean(np.sqrt(np.sum((p_2d[i]-p_3d[i][:,:2])**2.0, axis=1)))
#err_after = np.mean(np.sqrt(np.sum((p_2d[i]-p_3d_corr[:,:2])**2.0, axis=1)))
#print('error before: %.2f' % err_before)
#print('error after: %.2f' % err_after)

##%% training of correction matrices to compensate errors
#p_2d = [pts[it,:,:] for it in range(pts.shape[0]) ]
#p_3d = [np.dot(getPts3D(deg2rad(-angle)), Tproj.transpose()) for angle in angles]
#p_3d = [p_3d[it]/np.tile(p_3d[it][:,2],(3,1)).transpose() for it in range(len(p_3d))]
#
#
## correct for in-plane translation
##get_Tcorr = lambda it: np.array(((1,0,np.mean(p_2d[it][:,0]-p_3d[it][:,0])),\
##                  (0,1,np.mean(p_2d[it][:,1]-p_3d[it][:,1])),\
##                  (0,0,1)))
## correct for 2d affine transformation
#get_Tcorr = lambda it: np.array(rvlib.mapAffineApprox2D(p_3d[it], p_2d[it]))
#
#Tcorr = [ get_Tcorr(it) for it in range(len(p_2d)) ]

#def project_points(pts, P)
#    pts = np.dot(pts, P.transpose())
#    pts[:,0] = pts[:,0] / pts[:,2]
#    pts[:,1] = pts[:,1] / pts[:,2]
#    return pts
#    
#Pop = lambda x: project_points(x, Tproj)
#
#plt.close("all")
#for i in range(pts):
#    plt.plot(ptsXYZproj[:,0], ptsXYZproj[:,1],'gx')


#%% filter projection images

def nextpow2(i):
    '''
    Find 2^n that is equal to or greater than
    
    :param i: arbitrary non-negative integer 
    :return: integer with power of two
    '''
    n = 1
    while n < i: n *= 2
    return n

def ramp_flat(n):
    '''
    Create 1D ramp filter in the spatial domain.
    
    :param n: Size of the filter, should be of power of 2. 
    :return: Ramp filter response vector.
    '''
    nn = np.arange(-(n/2),(n/2))
    h = np.zeros((nn.size,), dtype='float')
    h[n/2] = 1 / 4
    odd = np.mod(nn,2) == 1
    h[odd] = -1.0 / (np.pi * nn[odd])**2.0;
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
    f_kernel = np.abs(np.fft.fft(ramp_kernel)) * 2 # transform ramp filter to freqency domain
    filt = f_kernel[0:order//2+1].transpose()
    w = 2.0*np.pi*np.arange(0,filt.size)/order # frequency axis up to Nyquist 
    
    filter_type = filter_type.lower()   
    if filter_type == 'ram-lak':
        # do nothing
        None
    elif filter_type == 'shepp-logan':    
        # be careful not to divide by 0:
        filt[1:] = filt[1:] * (np.sin(w[1:]/(2*d))/(w[1:]/(2*d)))
    elif filter_type == 'cosine':
        filt[1:] = filt[1:] * np.cos(w[1:]/(2*d))
    elif filter_type == 'hamming':  
        filt[1:] = filt[1:] * (.54 + .46 * np.cos(w[1:]/d))
    elif filter_type == 'hann':
        filt[1:] = filt[1:] * (1+np.cos(w[1:]/d)) / 2
    else:
        raise ValueError('filter_type: invalid filter selected "{}"'.format(filter_type))
        
    filt[w>np.pi*d] = 0 # crop the frequency response
    filt = np.hstack((filt, filt[-2:0:-1])) # enforce symmetry of the filter
    
    return filt

def filter_projection(proj, filter_type='hann', cut_off=1, axis=0):
    '''
    Filter projection image using ramp-like filter
    
    :param proj: Projection image (u x v)
    :param filter_type: 
    :param cut_off: 
    :param axis: 
    :return: 
    '''
    """
    
    
    Inputs: 
        proj        - 
        filter_type - can be 'ram-lak', 'cosine', 'hamming' or 'hann'
        cut_off     - cut off frequency (0-1)
    Output:
        fproj       - filtered projection image
    
    """

    if filter_type == 'none':
        return proj

    if axis == 1:
        proj = proj.transpose()
    
    nu, nv = proj.shape
    filt_len = np.max([64, nextpow2(2*nu)])
    ramp_kernel, nn = ramp_flat(filt_len)
    
    filt = create_filter(filter_type, ramp_kernel, filt_len, cut_off)
    filt = np.tile(filt[:,np.newaxis],nv)
    # rvlib.showImage(filt, iCmap=cm.jet)
    
    # be careful to select the filter axis (corresponding to rotation axis)

    # append zeros    
    fproj = np.zeros((filt_len,nv), dtype='float')
    fproj[filt_len/2-nu/2:filt_len/2+nu/2,:] = proj
    
    # filter using fourier theorem
    fproj = np.fft.fft(fproj, axis=0)
    fproj = fproj * filt
    fproj = np.real(np.fft.ifft(fproj, axis=0))    
    fproj = fproj[filt_len/2-nu/2:filt_len/2+nu/2,:]    

    if axis == 1:
        fproj = fproj.transpose()
    
    return fproj

#%% perform backprojection
def fbp(imgs, angles, Tproj, filter_type='hann', filename='volume', sampling=2):
    
    def rgb2gray(img):
        """Convert RGB image to grayscale"""
        if img.ndim == 3:
            return np.mean(img.astype('float'), axis=-1)
        return img
    
    def get_volume(voldims = (100,100,100), sampling=1):
        """
        Define volume size and sampling points
        
        Inputs: 
            voldims     - define volume size in mm (for X x Y x Z axes)
            sampling    - define sampling step in mm
        Output:
            grid        - grid of points in the volume (in homogeneous coordinates)
            volsiz      - grid sizes (X x Y x Z)
        """    
        # get sampling points; the grid is axis aligned, and defined wrt to reference 
        # point on top of the caliber, which is determined by imaging the caliber
        xr = np.arange(-voldims[0]/2,voldims[0]/2, sampling);
        yr = np.arange(-voldims[1]/2,voldims[1]/2, sampling);
        zr = np.arange(-voldims[2]/2,voldims[2]/2, sampling);
        xg, yg, zg = np.meshgrid(xr,yr,zr,indexing='xy')
        # store volume shape
        grid_size = xg.shape
        # define matrix of homogeneous point coordinates
        grid = np.vstack((xg.flatten(), yg.flatten(), zg.flatten(), np.ones_like(xg.flatten())))
        grid = grid.transpose()
        # return 
        return grid, grid_size
    
    # debug: show result
    #i = 0; rvlib.showImage(imgs[i], iTitle='image #%d' % i)
    #plt.plot(grid[:,0], grid[:,1],'rx')
    
    angles = [-x for x in angles]      
    
    # get sampling points in homogeneous coordinates
    grid_raw, grid_size = get_volume(voldims=(200,200,300), sampling=sampling)
    
    # initialize volume
    vol = np.zeros(grid_size)
    xs,ys,zs = grid_size
    
    for i in range(len(angles)-1):       
        # display current status
        print("processing image #%d/%d" % (i+1, len(angles)))
        
        # normalize image
        img_t = rgb2gray(imgs[i]).astype('float')
    #    img_t = (img_t - np.min(img_t)) / (np.max(img_t) - np.min(img_t))    
        img_t = (img_t - np.mean(img_t)) / (np.std(img_t))
        
        # filter projection image     
        img_f = filter_projection(img_t , filter_type, cut_off=0.75, axis=1)
        img_f = (img_f - np.mean(img_f)) / (np.std(img_f))
        
        # define function to put points in reference space
        get_grid_at_angle = lambda ang: \
            np.dot(np.dot(Tproj[1], transRigid3D(trans=(0, 0, -100), rot=(0, 0, -np.pi + ang))), \
                   np.transpose(grid_raw)).transpose()
        
        # project points to imaging plane
        grid = np.dot(get_grid_at_angle(deg2rad(-angles[i])), Tproj[0].transpose())
        grid[:,0] = grid[:,0] / grid[:,2]
        grid[:,1] = grid[:,1] / grid[:,2]
        grid[:,2] = 1
    
        # correct in-plane errors due to incorrect geometry
    #    grid = np.dot(grid, Tcorr[i].transpose())    
    
    #    plt.close('all')
    #    rvlib.showImage(img_t, iTitle='original grid')
    #    plt.plot(grid[:,0], grid[:,1],'rx')
    #
    #rvlib.showImage(img_t, iTitle='corrected grid')
    #plt.plot(grid[:,0], grid2[:,1],'rx')
    
    #rvlib.showImage(img_f)
    
    
        # interpolate points to obtain backprojected volume
        us, vs = img_f.shape
        img_backprojected = interpn((np.arange(vs),np.arange(us)), img_f.transpose(),\
            grid[:,:2],method='linear', bounds_error=False)
        img_backprojected = img_backprojected.reshape((xs,ys,zs))
        vol = vol + img_backprojected
        vol[np.isnan(vol)]=0
    
    #    rvlib.showImage(img_resampled[:,:,0])    
    
    print('writing volume to file "%s.nrrd"...' % filename)
    nrrd.write('%s.nrrd' % filename, vol)

    return vol    
