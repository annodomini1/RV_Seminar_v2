# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 13:36:16 2015

@author: Ziga Spiclin

Image acquisition calibration
"""

from scipy.interpolate import interpn
from PIL import Image as im
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import rvlib
import nrrd

plt.ioff

#%% test nrrd
import nrrd
import numpy as np
nrrd.write('test.nrrd',np.zeros((100,100,100)))

#%% load images from folder
#folder = 'Z:\Work\Pedagogics\Poletna šola\ev3software\sw\python\\tree_test_r60_1st'
#folder = 'Z:\Work\Pedagogics\Poletna šola\ev3software\sw\python\\tree_test_r60_2nd'
#folder = 'Z:\Work\Pedagogics\Poletna šola\ev3software\sw\python\\tree_test_r60_2nd_gamma'
#folder = 'Z:\Work\Pedagogics\Poletna šola\ev3software\sw\python\pig_r60'
folder = 'Z:\\Work\\Pedagogics\\Poletni tabor\\software\\python\\data\\acquisitions\\phantomInCenter'
#folder = 'Z:\Work\Pedagogics\Poletna šola\ev3software\sw\python\cylinder'

def rgb2gray( img ):
    """Convert RGB image to grayscale"""
    if img.ndim == 3:
        return np.mean( img.astype('float'), axis=-1 )
    return img

#crop = (950, 900, 2250, 1800) # crop for cylinder
crop = (150, 800, 2250, 2000) # crop for cylinder
#crop = None
import os
imgs = []
angles = []
for file in os.listdir(folder):
    if file.endswith(".jpg"):
        print( 'loading file "%s"...' % file )
        img_t = rgb2gray( np.array( im.open( "%s\\%s" % (folder, file) ) ) )
        if crop != None:
            img_t = img_t[crop[0]:crop[2], crop[1]:crop[3]]
        imgs.append( img_t )
        angles.append( float(file.replace('.jpg','').split(sep='_')[-1]) )

#rvlib.showImage(imgs[15])

idx = np.argsort( angles )
imgs = [imgs[i] for i in idx]
angles = [angles[i] for i in idx]

#%% load images into workspace
imgs = []
for i in range(1,23):
    imgs.append( np.array( im.open( "./caliber/caliber-%d.jpg" % i ) ) )    

#for i in range(1,22):
#    imgs.append( np.array( im.open( "./phantom/phantom-%d.jpg" % i ) ) )

# show images
#rvlib.showImage( imgs[14] )
#rvlib.showImage( imgs[15] )
# check difference between first and last (which should be nearly the same)
#rvlib.showImage( np.abs( np.array( imgs[0], dtype='float' ) - np.array( imgs[-1], dtype='float' ) ), iCmap=cm.jet )

#%% check pairwise differences, remove duplicates and define rotation angles
#i = 0
#rvlib.plt.close('all')
#title = "displaying image pair %d-%d" % (i, i+1)
#rvlib.showImage( np.abs( np.array( imgs[i], dtype='float' ) - \
#    np.array( imgs[i+1], dtype='float' ) ), iCmap=cm.jet, iTitle=title )
#i = i+1

# 
# 16-17 are duplicated, thus remove image at index 16 or 17
imgs = imgs[:16] + imgs[17:]
#idx = list(range(0,22))
#idx.pop(16)

# define rotation angles with respect to first image
angle_step = 360 / len(imgs)
angles = angle_step * np.arange(len(imgs))

#%% define grid of points on the calibration object
def get_calibration_object():
    # define basic measurements
    d = 18.5 # ball diameter
    dz = 22.0 # ball height from base
    dh = np.round((131 - 12)/15) # distance between lego holes
    df = 20 + 1 # center to based of attachment frame + 1 mm gap
    
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
#    p_bottom_frame = p_top_frame + (0, 0, -15*dh-4-8-4)
#    p_bottom_ground = p_bottom_frame + (0, 0, -4-8-4)
    
    # create array of points
    return np.vstack((b0,b1,b2,b3,b4,b5,b6,b7))

pts = get_calibration_object()

# draw point in 3D
from mpl_toolkits.mplot3d import Axes3D
plt.close('all')
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

pts3d = np.copy( pts )

#%% draw complete geometry with camera and caliber



#%% find ball center coordinates in all images

#%% annotate all images
pts_ann = []
a = (0,)
for i in range(len(imgs)):
#for i in a:
    plt.close('all')
    rvlib.showImage( imgs[i], iTitle='image #%d' % i )
    pts_ann.append( plt.ginput( n=8, timeout=-1 ) )

# correct single entry
#    rvlib.showImage( imgs[2], iTitle='image #%d' % 2 )
#    pts_ann2 = plt.ginput( n=8, timeout=-1 )
#np.save('pts_ann_phantomInCenter-crop2', pts_ann)
np.save('..//data//pts_ann_phantomInCenter-crop%s' % 
    ('('+','.join([str(c) for c in crop])+')',),
    pts_ann)
#%% show all annotations
import time
#pts = np.load('pts_ann.npy')
#pts = np.load('pts_ann_phantomInCenter.npy')
#pts = np.load('..\\data\\pts_ann_phantomInCenter-crop2.npy')
#pts = np.load('..\\data\\pts_ann_phantomInCenter.npy')
#pts = np.load('..\\data\\pts_ann.npy')
pts = np.load('..//data//pts_ann_phantomInCenter-crop%s' % 
    ('('+','.join([str(c) for c in crop])+').npy',))
#for i in range(len(imgs)):
i = 0
plt.close('all')
rvlib.showImage( imgs[i], iTitle='image #%d' % i )
plt.plot(pts[i][:,0], pts[i][:,1],'mx')
plt.show()
#    time.sleep(2)
#    wait = input("PRESS ENTER TO CONTINUE.")

p_0 = np.squeeze( np.array([np.hstack( p[0,:] ) for p in pts]) )
p_1 = np.squeeze( np.array([np.hstack( p[1,:] ) for p in pts]) )
p_2 = np.squeeze( np.array([np.hstack( p[2,:] ) for p in pts]) )
p_3 = np.squeeze( np.array([np.hstack( p[3,:] ) for p in pts]) )
p_4 = np.squeeze( np.array([np.hstack( p[4,:] ) for p in pts]) )
p_5 = np.squeeze( np.array([np.hstack( p[5,:] ) for p in pts]) )
p_6 = np.squeeze( np.array([np.hstack( p[6,:] ) for p in pts]) )
p_7 = np.squeeze( np.array([np.hstack( p[7,:] ) for p in pts]) )

p_0_mean = np.mean(p_0, axis=0)
p_1_mean = np.mean(p_1, axis=0)
p_2_mean = np.mean(p_2, axis=0)
p_3_mean = np.mean(p_3, axis=0)
p_4_mean = np.mean(p_4, axis=0)
p_5_mean = np.mean(p_5, axis=0)
p_6_mean = np.mean(p_6, axis=0)
p_7_mean = np.mean(p_7, axis=0)


i = 0    
rvlib.showImage( imgs[i], iTitle='image #%d' % i )
plt.plot(p_0[:,0], p_0[:,1],'rx', markersize=3)
plt.plot(p_1[:,0], p_1[:,1],'rx', markersize=3)
plt.plot(p_2[:,0], p_2[:,1],'rx', markersize=3)
plt.plot(p_3[:,0], p_3[:,1],'rx', markersize=3)
plt.plot(p_4[:,0], p_4[:,1],'rx', markersize=3)
plt.plot(p_5[:,0], p_5[:,1],'rx', markersize=3)
plt.plot(p_6[:,0], p_6[:,1],'rx', markersize=3)
plt.plot(p_7[:,0], p_7[:,1],'rx', markersize=3)


plt.plot(p_0_mean[0], p_0_mean[1],'gx', markersize=5)
plt.plot(p_1_mean[0], p_1_mean[1],'gx', markersize=5)
plt.plot(p_2_mean[0], p_2_mean[1],'gx', markersize=5)
plt.plot(p_3_mean[0], p_3_mean[1],'gx', markersize=5)
plt.plot(p_4_mean[0], p_4_mean[1],'gx', markersize=5)
plt.plot(p_5_mean[0], p_5_mean[1],'gx', markersize=5)
plt.plot(p_6_mean[0], p_6_mean[1],'gx', markersize=5)
plt.plot(p_7_mean[0], p_7_mean[1],'gx', markersize=5)


#%% position points in space
def Trigid( trans=(0,0,0), rot=(0,0,0) ):
    """Rigid body transformation"""
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
    
#print(Trigid( trans=(1,2,3), rot=(-np.pi/2,0,0)))
dt = -d/2-(dz-d)-dh/2-15*dh-4 #-8-4 # slednje prinese podstavek
Ttable = Trigid( trans = (795, -10, -233 + 88 - dt) )
# 99 # visina mize s stekleno plosco
# 92 # visina mize brez steklene plosce
# 88 # visina pritrditve kalibra (4 mm razlike zaradi gumic)

#i = 0
#plt.close('all')
#rvlib.showImage( imgs[i], iTitle='image #%d' % i )
#plt.plot(pts[i][:,0], pts[i][:,1],'mx')
#plt.show()

#p0 = pts[i][:]

pts3dcam = np.array((0,0,0,1))
pts3dh = np.hstack( (pts3d, np.ones((np.size(pts3d,0),1))) )
pts3dht = np.dot( np.dot(Ttable, Trigid(rot=(0,0,-np.pi))), np.transpose(pts3dh) )
pts3dht = np.transpose( pts3dht )

# draw point in 3D
from mpl_toolkits.mplot3d import Axes3D
plt.close('all')
fig = plt.figure()    
ax = fig.add_subplot(111,projection='3d')
ax.scatter(pts3dcam[0],pts3dcam[1],pts3dcam[2], c='r', marker='o')
ax.scatter(pts3dht[:,0],pts3dht[:,1],pts3dht[:,2], c='m', marker='o')

X = np.hstack( (pts3dcam,pts3dht[:,0]) )
Y = np.hstack( (pts3dcam,pts3dht[:,1]) )
Z = np.hstack( (pts3dcam,pts3dht[:,2]) )

max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0

mean_x = X.mean()
mean_y = Y.mean()
mean_z = Z.mean()
ax.set_xlim(mean_x - max_range, mean_x + max_range)
ax.set_ylim(mean_y - max_range, mean_y + max_range)
ax.set_zlim(mean_z - max_range, mean_z + max_range)

plt.show()


#%% draw all points in space
getPts3D = lambda ang: np.dot( np.dot(Ttable, Trigid(rot=(0,0,-np.pi+ang))), np.transpose(pts3dh) ).transpose()

def deg2rad( ang ):
    return ang*np.pi/180.0

pts_all = np.zeros((0,4))
for angle in angles:
    pts_all = np.vstack( (pts_all, getPts3D( deg2rad( - angle ) ) ) )

# draw points in 3D
from mpl_toolkits.mplot3d import Axes3D
plt.close('all')
fig = plt.figure()    
ax = fig.add_subplot(111,projection='3d')
ax.scatter(pts3dcam[0],pts3dcam[1],pts3dcam[2], c='r', marker='o')    
ax.scatter(pts_all[:,0],pts_all[:,1],pts_all[:,2], c='m', marker='o')

X = np.hstack( (pts3dcam,pts_all[:,0]) )
Y = np.hstack( (pts3dcam,pts_all[:,1]) )
Z = np.hstack( (pts3dcam,pts_all[:,2]) )

max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0

mean_x = X.mean()
mean_y = Y.mean()
mean_z = Z.mean()
ax.set_xlim(mean_x - max_range, mean_x + max_range)
ax.set_ylim(mean_y - max_range, mean_y + max_range)
ax.set_zlim(mean_z - max_range, mean_z + max_range)

plt.show()

#%%
def dlt_calibration( ptsW, ptsU ):
    """Perform DLT camera calibration based on input points"""
    def get_mat_row( ptW, ptU ):
        row1 = np.array( (ptU[0],ptU[1],ptU[2],1,0,0,0,0,-ptU[0]*ptW[0],-ptU[1]*ptW[0],-ptU[2]*ptW[0]) )
        row2 = np.array( (0,0,0,0,ptU[0],ptU[1],ptU[2],1,-ptU[0]*ptW[1],-ptU[1]*ptW[1],-ptU[2]*ptW[1]) )        
        return np.vstack((row1, row2)), np.vstack((ptW[0], ptW[1]))
    dmat = np.zeros((0,11))
    dvec = np.zeros((0,1))
    for i in range(ptsW.shape[0]):
#        print(dmat.shape)
#        print(get_mat_row(ptsW[i,:], ptsU[i,:]).shape)
        dmatp, dvecp = get_mat_row(ptsW[i,:], ptsU[i,:])
        dmat = np.vstack((dmat,dmatp))
        dvec = np.vstack((dvec,dvecp))
    return dmat, dvec

#sensor_size = 25.4/4 # 1/4" is the sensor size
pixel_size = (25.4/4) / np.sqrt(float(imgs[0].shape[0]**2 + imgs[0].shape[1]**2))
ptsW = np.copy(pts[0])
#ptsW[:,0] = (ptsW[:,0] - imgs[0].shape[0]/2) * pixel_size
#ptsW[:,1] = (ptsW[:,1] - imgs[0].shape[1]/2) * pixel_size

a, b = dlt_calibration( ptsW, pts3dht )
#b = np.zeros( (a.shape[0], 1))
D = np.linalg.lstsq( a, b)
print(D[0])

Tproj = np.reshape(np.vstack((D[0],1.0)),(3,4))

ptsXYZproj = np.dot( pts3dht, Tproj.transpose() )
ptsXYZproj[:,0] = ptsXYZproj[:,0] / ptsXYZproj[:,2]
ptsXYZproj[:,1] = ptsXYZproj[:,1] / ptsXYZproj[:,2]

plt.plot(ptsW[:,0], ptsW[:,1],'rx')
plt.plot(ptsXYZproj[:,0], ptsXYZproj[:,1],'gx')

#%% draw all point in the image space based on angles and projection
i = 0    
rvlib.showImage( imgs[i], iTitle='image #%d' % i )
plt.plot(p_0[:,0], p_0[:,1],'rx', markersize=3)
plt.plot(p_1[:,0], p_1[:,1],'rx', markersize=3)
plt.plot(p_2[:,0], p_2[:,1],'rx', markersize=3)
plt.plot(p_3[:,0], p_3[:,1],'rx', markersize=3)
plt.plot(p_4[:,0], p_4[:,1],'rx', markersize=3)
plt.plot(p_5[:,0], p_5[:,1],'rx', markersize=3)
plt.plot(p_6[:,0], p_6[:,1],'rx', markersize=3)
plt.plot(p_7[:,0], p_7[:,1],'rx', markersize=3)

ptsXYZproj = np.dot( pts_all, Tproj2.transpose() )
ptsXYZproj[:,0] = ptsXYZproj[:,0] / ptsXYZproj[:,2]
ptsXYZproj[:,1] = ptsXYZproj[:,1] / ptsXYZproj[:,2]
plt.plot(ptsW[:,0], ptsW[:,1],'rx')
plt.plot(ptsXYZproj[:,0], ptsXYZproj[:,1],'gx')

#%% plot error wrt to angle
ptsCal3d = np.zeros((0,4))
for angle in angles:
    ptsCal3d = np.vstack( (ptsCal3d, getPts3D( deg2rad( - angle ) ) ) )
    
ptsMan2d = pts.reshape((168,2))
#a, b = dlt_calibration(ptsRef, ptsMov)
#D2 = np.linalg.lstsq(a, b)
#Tproj2 = np.reshape(np.vstack((D2[0],1.0)),(3,4))

Tproj = np.reshape(np.vstack((D[0],1.0)),(3,4))

def project_points_3dto2d(pts, Tproj):
    pts = np.dot( pts, Tproj.transpose() )
    pts[:,0] = pts[:,0] / pts[:,2]
    pts[:,1] = pts[:,1] / pts[:,2]
    pts[:,2] = np.zeros_like(pts[:,2])
    return pts[:,:3]

ptsCal2d = project_points_3dto2d(ptsCal3d, Tproj)

err = np.sqrt(np.sum((ptsMan2d - ptsCal2d[:,:2])**2.0, axis=1))


#%% refine calibration object position 
def dlt_calibration_T(ptsW, ptsU):
    a, b = dlt_calibration(ptsW, ptsU)
    D = np.linalg.lstsq(a, b)
    return np.reshape(np.vstack((D[0],1.0)),(3,4))

def err_calib_setup(p, Tproj=None):  
    Ttable = Trigid( trans = p )
    getPts3D = lambda ang: np.dot( np.dot(Ttable, Trigid(rot=(0,0,-np.pi+ang))), np.transpose(pts3dh) ).transpose()

    if Tproj == None:
        Tproj = dlt_calibration_T(ptsMan2d[:8,:], getPts3D(0))  
        
    ptsCal3d = np.zeros((0,4))
    for angle in angles:
        ptsCal3d = np.vstack( (ptsCal3d, getPts3D( deg2rad( - angle ) ) ) )
    
    ptsCal2d = project_points_3dto2d(ptsCal3d, Tproj)
    
    return np.sqrt(np.mean(np.sum((ptsMan2d - ptsCal2d[:,:2])**2.0, axis=1)))


from scipy.optimize import fmin
Tproj = dlt_calibration_T(ptsW, pts3dht)
p0 = np.array((795, -10, -233 + 88 - dt))
#err_calib_setup(p0, Tproj)
crit_fcn = lambda p : err_calib_setup(p, Tproj)
ret = fmin(crit_fcn, p0, full_output=1, xtol=1e-6, ftol=1e-6)

#%% refine calibration object position and transformation
def add_homogeneous_coord(pts):
    return np.hstack((pts, np.ones((np.size(pts,0),1))))
    
def collect_calib_pts(angles, getPts3D):
    pts = np.zeros((0,4))
    for angle in angles:
        pts = np.vstack( (pts, getPts3D( deg2rad( - angle ) ) ) )
    return pts

def err_calib_setup(p):    
    p_trans = p[:3]
    p_Tproj = p[3:]

    getPts3D = lambda ang: np.dot(np.dot(Trigid(trans = p_trans), 
                                  Trigid(rot=(0,0,-np.pi+ang))), 
                                  np.transpose(add_homogeneous_coord(
                                      get_calibration_object()))
                                 ).transpose()

    p_Tproj = np.concatenate((p_Tproj,np.array([1]))).reshape((3,4))

    ptsCal3d = collect_calib_pts(angles, getPts3D)
#    ptsCal3d = np.zeros((0,4))
#    for angle in angles:
#        ptsCal3d = np.vstack( (ptsCal3d, getPts3D( deg2rad( - angle ) ) ) )
    
    ptsCal2d = project_points_3dto2d(ptsCal3d, p_Tproj)
    
    return np.sqrt(np.mean(np.sum((ptsMan2d - ptsCal2d[:,:2])**2.0, axis=1)))

from scipy.optimize import fmin
# manually located points on the caliber
pts = np.load('..//data//pts_ann_phantomInCenter-crop%s' % 
    ('('+','.join([str(c) for c in crop])+').npy',))
ptsW = np.copy(pts[0])
# points on the caliber in 3d
trans0 = np.array((795, -10, -233 + 88 - dt))
getPts3D = lambda ang: np.dot(np.dot(Trigid(trans = trans0), 
                              Trigid(rot=(0,0,-np.pi+ang))), 
                              np.transpose(add_homogeneous_coord(
                                  get_calibration_object()))
                             ).transpose()
# initial transformation obtained by dlt
Tproj0 = dlt_calibration_T(ptsW, getPts3D(0))
p0 = np.concatenate((trans0, Tproj0.flatten()[:-1]))
pts3d0 = collect_calib_pts(angles, getPts3D)

#err_calib_setup(p0)
crit_fcn = lambda p : err_calib_setup(p)
ret = fmin(crit_fcn, p0, full_output=1, xtol=1e-6, ftol=1e-6)

# separate resulting parameters into caliber translation and projection matrix
p = ret[0]
transopt = p[:3]
Tprojopt = p[3:]
Tprojopt = np.concatenate((Tprojopt,np.array([1]))).reshape((3,4))
# recompute the positions of 3d points
getPts3D = lambda ang: np.dot(np.dot(Trigid(trans = transopt), 
                              Trigid(rot=(0,0,-np.pi+ang))), 
                              np.transpose(add_homogeneous_coord(
                                  get_calibration_object()))
                             ).transpose()  
ptsopt = collect_calib_pts(angles, getPts3D)
# plot the manually determined points
i = 0    
rvlib.showImage(imgs[i], iTitle='image #%d' % i)
plt.plot(p_0[:,0], p_0[:,1],'rx', markersize=3)
plt.plot(p_1[:,0], p_1[:,1],'rx', markersize=3)
plt.plot(p_2[:,0], p_2[:,1],'rx', markersize=3)
plt.plot(p_3[:,0], p_3[:,1],'rx', markersize=3)
plt.plot(p_4[:,0], p_4[:,1],'rx', markersize=3)
plt.plot(p_5[:,0], p_5[:,1],'rx', markersize=3)
plt.plot(p_6[:,0], p_6[:,1],'rx', markersize=3)
plt.plot(p_7[:,0], p_7[:,1],'rx', markersize=3)
# plot the initially mapped caliber points
ptsXYZproj = np.dot(pts3d0, Tproj0.transpose())
ptsXYZproj[:,0] = ptsXYZproj[:,0] / ptsXYZproj[:,2]
ptsXYZproj[:,1] = ptsXYZproj[:,1] / ptsXYZproj[:,2]
plt.plot(ptsXYZproj[:,0], ptsXYZproj[:,1],'mo')
# plot the optimally mapped caliber points
ptsXYZproj = np.dot(ptsopt, Tprojopt.transpose())
ptsXYZproj[:,0] = ptsXYZproj[:,0] / ptsXYZproj[:,2]
ptsXYZproj[:,1] = ptsXYZproj[:,1] / ptsXYZproj[:,2]
plt.plot(ptsXYZproj[:,0], ptsXYZproj[:,1],'gx')

#%% draw point in the image space based on angle and projection
plt.close('all')
#p_2d = [p_0, p_1, p_2, p_3, p_4, p_5, p_6, p_7];
p_2d = [pts[it,:,:] for it in range(pts.shape[0]) ]
p_3d = [np.dot( getPts3D( deg2rad( -angle ) ), Tproj.transpose()) for angle in angles]
p_3d = [p_3d[it]/np.tile(p_3d[it][:,2],(3,1)).transpose() for it in range(len(p_3d))]

i=15
rvlib.showImage( imgs[i], iTitle='image #%d' % i )
plt.plot(p_2d[i][:,0], p_2d[i][:,1],'rx',markersize=15)
#plt.plot(p_0[:,0], p_0[:,1],'rx')
plt.plot(p_3d[i][:,0], p_3d[i][:,1],'gx',markersize=15)

du = np.mean(p_2d[i][:,0]-p_3d[i][:,0])
dv = np.mean(p_2d[i][:,1]-p_3d[i][:,1])
Tcorr = np.array(((1,0,du),(0,1,dv),(0,0,1)))
Tcorr = np.array( rvlib.mapAffineApprox2D( p_3d[i], p_2d[i] ) )

p_3d_corr = np.dot( p_3d[i], Tcorr.transpose() )

plt.plot(p_3d_corr[:,0], p_3d_corr[:,1],'mx',markersize=15)

err_before = np.mean( np.sqrt( np.sum( (p_2d[i]-p_3d[i][:,:2])**2.0, axis=1 ) ) )
err_after = np.mean( np.sqrt( np.sum( (p_2d[i]-p_3d_corr[:,:2])**2.0, axis=1 ) ) )
print('error before: %.2f' % err_before)
print('error after: %.2f' % err_after)

#%% training of correction matrices to compensate errors
p_2d = [pts[it,:,:] for it in range(pts.shape[0]) ]
p_3d = [np.dot( getPts3D( deg2rad( -angle ) ), Tproj.transpose()) for angle in angles]
p_3d = [p_3d[it]/np.tile(p_3d[it][:,2],(3,1)).transpose() for it in range(len(p_3d))]


# correct for in-plane translation
#get_Tcorr = lambda it: np.array(((1,0,np.mean(p_2d[it][:,0]-p_3d[it][:,0])),\
#                  (0,1,np.mean(p_2d[it][:,1]-p_3d[it][:,1])),\
#                  (0,0,1)))
# correct for 2d affine transformation
get_Tcorr = lambda it: np.array( rvlib.mapAffineApprox2D( p_3d[it], p_2d[it] ) )

Tcorr = [ get_Tcorr(it) for it in range(len(p_2d)) ]

#def project_points( pts, P )
#    pts = np.dot( pts, P.transpose() )
#    pts[:,0] = pts[:,0] / pts[:,2]
#    pts[:,1] = pts[:,1] / pts[:,2]
#    return pts
#    
#Pop = lambda x: project_points( x, Tproj)
#
#plt.close("all")
#for i in range(pts):
#    plt.plot(ptsXYZproj[:,0], ptsXYZproj[:,1],'gx')


#%% filter projection images
def nextpow2(i):
    """Find 2^n that is equal to or greater than"""
    n = 1
    while n < i: n *= 2
    return n

def ramp_flat( n ):
    """Create 1D ramp filter"""
    nn = np.arange(-(n/2),(n/2))
    h = np.zeros( (nn.size, ), dtype='float')
    h[n/2] = 1 / 4
    odd = np.mod(nn,2) == 1
    h[odd] = -1.0 / (np.pi * nn[odd])**2.0;
    return h, nn

def create_filter( filter_type, kernel, order, d ):
    """Create 1d filter of selected type"""
    f_kernel = np.abs( np.fft.fft( kernel ) ) * 2
    filt = f_kernel[0:order/2+1].transpose()
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
        raise ValueError('filter_type: invalid filter selected "%s"' % filter_type)
        
    filt[w>np.pi*d] = 0 # Crop the frequency response
    filt = np.hstack( (filt, filt[-2:0:-1]) ) # Symmetry of the filter    
    
    return filt

def filter_projection( proj, filter_type='hann', cut_off=1, axis=0 ):
    """
    Filter projection image using ramp-like filter
    
    Inputs: 
        proj        - projection image (u x v)
        filter_type - can be 'ram-lak', 'cosine', 'hamming' or 'hann'
        cut_off     - cut off frequency (0-1)
    Output:
        fproj       - filtered projection image
    
    """

    if axis == 1:
        proj = proj.transpose()
    
    nu, nv = proj.shape
    filt_len = np.max( [64, nextpow2(2*nu)] )
    ramp_kernel, nn = ramp_flat( filt_len )
    
    filt = create_filter( filter_type, ramp_kernel, filt_len, cut_off )
    filt = np.tile(filt[:,np.newaxis],nv)
    # rvlib.showImage( filt, iCmap=cm.jet )
    
    # be careful to select the filter axis (corresponding to rotation axis)

    # append zeros    
    fproj = np.zeros( (filt_len,nv), dtype='float' )
    fproj[filt_len/2-nu/2:filt_len/2+nu/2,:] = proj
    
    # filter using fourier theorem
    fproj = np.fft.fft( fproj, axis=0 )
    fproj = fproj * filt
    fproj = np.real( np.fft.ifft( fproj, axis=0 ) )    
    fproj = fproj[filt_len/2-nu/2:filt_len/2+nu/2,:]    

    if axis == 1:
        fproj = fproj.transpose()
    
    return fproj
    
#nu = 256; nv = 200
#proj = np.ones((nu,nv))
#nu, nv, c = imgs[0].shape
proj = np.squeeze( imgs[0] )
rvlib.showImage( filter_projection( proj, 'ram-lak', cut_off=0.25), iCmap = cm.jet )
#rvlib.showImage( filter_projection( proj, 'hann', cut_off=0.75, axis=1), iCmap = cm.jet )
#rvlib.showImage( filter_projection( proj, 'hann'), iCmap = cm.jet )
#rvlib.showImage( proj )

#np.fft

#%% perform backprojection
def get_volume( voldims = (100,100,100), sampling=1 ):
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
    grid = np.vstack( (xg.flatten(), yg.flatten(), zg.flatten(), np.ones_like(xg.flatten())) )
    grid = grid.transpose()
    # return 
    return grid, grid_size

# debug: show result
#i = 0; rvlib.showImage( imgs[i], iTitle='image #%d' % i )
#plt.plot(grid[:,0], grid[:,1],'rx')

# get sampling points in homogeneous coordinates
grid_raw, grid_size = get_volume( voldims=(200,200,300), sampling=1 )

# initialize volume
vol = np.zeros( grid_size )
xs,ys,zs = grid_size

for i in range(len(angles)-1):
#for i in range(1):
#a = (40,)
#for i in a:
    
        # display current status
    print("processing image #%d" % i)
    
    # normalize image
    img_t = rgb2gray( imgs[i] ).astype('float')
#    img_t = (img_t - np.min(img_t)) / (np.max(img_t) - np.min(img_t))    
    img_t = (img_t - np.mean(img_t)) / (np.std(img_t))
    
    # filter projection image     
    img_f = filter_projection( img_t , 'hann', cut_off=0.75, axis=1 )
    img_f = (img_f - np.mean(img_f)) / (np.std(img_f))
#    img_f = img_t    
    
    # define function to put points in reference space
    get_grid_at_angle = lambda ang: \
        np.dot( np.dot(Ttable, Trigid(trans=(0,0,0),rot=(0,0,-np.pi + ang))), \
        np.transpose(grid_raw) ).transpose()
    
    # project points to imaging plane
    grid = np.dot( get_grid_at_angle( deg2rad( -angles[i] ) ), Tproj.transpose() )
    grid[:,0] = grid[:,0] / grid[:,2]
    grid[:,1] = grid[:,1] / grid[:,2]
    grid[:,2] = 1

    # correct in-plane errors due to incorrect geometry
#    grid = np.dot( grid, Tcorr[i].transpose() )    

#    plt.close('all')
#    rvlib.showImage( img_t, iTitle='original grid' )
#    plt.plot(grid[:,0], grid[:,1],'rx')
#
#rvlib.showImage( img_t, iTitle='corrected grid' )
#plt.plot(grid[:,0], grid2[:,1],'rx')

#rvlib.showImage( img_f )


    # interpolate points to obtain backprojected volume
    us, vs = img_f.shape
    img_backprojected = interpn( (np.arange(vs),np.arange(us)), img_f.transpose(),\
        grid[:,:2],method='linear', bounds_error=False)
    img_backprojected = img_backprojected.reshape( (xs,ys,zs) )
    vol = vol + img_backprojected
    vol[np.isnan(vol)]=0

#    rvlib.showImage( img_resampled[:,:,0] )

print('writing volume to file...')
nrrd.write('volume.nrrd',vol)
#nrrd.write('backprojection.nrrd',img_backprojected)
