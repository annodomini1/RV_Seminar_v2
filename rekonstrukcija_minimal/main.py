import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import PIL.Image as im
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from os.path import join

import reconlib as rl

# ---------- NALOZI SLIKE IZ MAPE ----------
# pth = 'C:/Users/PTIT/Desktop/PTIT/data'
pth = '/home/martin/Desktop/RV_Seminar_v2/rekonstrukcija_minimal'

acquisition_data_pth = join(pth, 'acquisitions', 'klovn30')
calibration_image_fname = join(pth, 'calibration', 'Aneja je pro.jpg')
calibration_data_fname = join(pth, 'calibration', 'tocke_kalibra_aneja.npy')
out_volume_fname = join(pth, 'reconstructions', 'klovn3d.nrrd')

slike, koti = rl.load_images(acquisition_data_pth, proc=rl.rgb2gray)

# ---------- DOLOCI 3D KOORDINATE TOCK NA KALIBRU ----------
pts3d = rl.IRCT_CALIBRATION_OBJECT()
# plt.close('all')
# r3d.show_points_in_3d(pts3d)

# ---------- OZNACI 8 TOCK NA KALIBRU, KI NAJ OZNACUJEJO SREDISCE KROGEL ----------
if not os.path.exists(calibration_data_fname):
    calibration_image = np.array(im.open(calibration_image_fname))
    pts2d =  rl.annotate_caliber_image(calibration_image, calibration_data_fname, n=8)

    plt.close('all')
    pts2d = np.load(calibration_data_fname)[0]
    rl.showImage(slike[0], iTitle='Oznacena sredisca krogel na kalibru.')
    plt.plot(pts2d[:,0], pts2d[:,1],'mx',markersize=15)

pts2d = np.load(calibration_data_fname)[0]

# ---------- KALIBRIRAJ SISTEM ZA ZAJEM SLIK ----------
Tproj, pts3dproj = rl.calibrate_irct(pts2d, pts3d)

# plt.close('all')
# imlib.showImage(slike[0], iTitle='Oznacena sredisca krogel na kalibru.')
plt.plot(pts2d[:,0], pts2d[:,1],'rx', markersize=15)
plt.plot(pts3dproj[:,0], pts3dproj[:,1],'gx', markersize=15)

# ---------- FILTRIRANJE 2D SLIK PRED POVRATNO PROJEKCIJO ----------
slika = np.squeeze(slike[0])
tip_filtra = 'hann'  # none, ram-lak, cosine, hann, hamming
slika_f = rl.filter_projection(slika, tip_filtra, cut_off=0.75)
# imlib.showImage(slika_f, iCmap=cm.jet)

# ---------- REKONSTRUKCIJA 3D SLIKE ----------
# FBP = Filtered BackProjection
vol = rl.fbp(slike[::1], koti[::1], Tproj,
              filter_type='hann', sampling_mm=3,
              out_fname=out_volume_fname)



def linScale(iImage, oMax):
    k = oMax/np.max(iImage)
    oImage = k*iImage
    return oImage

def thresholdImage(iImage, iThreshold):
    oImage = 255 * np.array(iImage > iThreshold, dtype='uint8')
    return oImage

#Thres = 100
Thres = 15
Deci = 5
endHeightShare = 0.9
startHeightShare = 0.1

# generacija thresholdanega volumna
# for z in range(dz):
#     dImage = vol[:,:,z]
#     dImage = linScale(dImage, 255)
#     dImage = thresholdImage(dImage, Thres)
#     #dvol = np.append(dvol, dImage, axis=2)
#     for x in range(dx):
#         for y in range(dy):
#             dvol[x,y,z] = dImage[x,y]


pointCoorX = []
pointCoorY = []
pointCoorZ = []
dvol = np.ones_like(vol)

# [dX, dY, dZ] = vol.shape
dZ = len(vol[0,0,:])
endZ = int(np.round(dZ*endHeightShare))
startZ = int(np.round(dZ*startHeightShare))

vol = vol[:,:,startZ:endZ]
[dx, dy, dz] = vol.shape

# for z in range(endZ):
for z in range(dz):
    dImage = vol[:,:,z]
    #dImage = linScale(dImage, 255)
    dImage = thresholdImage(dImage, Thres)

    for x in range(dx):
        for y in range(dy):
            dvol[x,y,z] = dImage[x,y]

            if (dImage[x, y] <= Thres):
                pointCoorX.append(x)
                pointCoorY.append(y)
                pointCoorZ.append(z)

# imlib.showImage(dvol[:,:,endZ-1])
# plt.show()

#redcenje tock
pointCoorX = pointCoorX[::Deci]
pointCoorY = pointCoorY[::Deci]
pointCoorZ = pointCoorZ[::Deci]

# #3D (ne naredi equal osi :s)
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.set_aspect('equal')
# ax.scatter(pointCoorX, pointCoorY, pointCoorZ)
# plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_aspect('equal')

X = pointCoorX
Y = pointCoorY
Z = pointCoorZ

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

