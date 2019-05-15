import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import PIL.Image as im
import numpy as np
# import nrrd

from os.path import join

import imgproc.imlib as imlib
import imgproc.reconstruction as r3d
import kalibracija as calibration
# import calibration

# ---------- NALOZI SLIKE IZ MAPE ----------
# pth = 'C:/Users/PTIT/Desktop/PTIT/data'
pth = '/home/martin/Desktop/RV_Seminar/rekonstrukcija'

acquisition_data_pth = join(pth, 'acquisitions', 'klovn30')
calibration_image_fname = join(pth, 'calibration', 'Aneja je pro.jpg')
calibration_data_fname = join(pth, 'calibration', 'tocke_kalibra_aneja.npy')
out_volume_fname = join(pth, 'reconstructions', 'klovn3d.nrrd')

slike, koti = r3d.load_images(acquisition_data_pth, proc=imlib.rgb2gray)

# ---------- DOLOCI 3D KOORDINATE TOCK NA KALIBRU ----------
pts3d = calibration.IRCT_CALIBRATION_OBJECT()
# plt.close('all')
# r3d.show_points_in_3d(pts3d)

# ---------- OZNACI 8 TOCK NA KALIBRU, KI NAJ OZNACUJEJO SREDISCE KROGEL ----------
if not os.path.exists(calibration_data_fname):
    calibration_image = np.array(im.open(calibration_image_fname))
    pts2d =  r3d.annotate_caliber_image(calibration_image, calibration_data_fname, n=8)

    plt.close('all')
    pts2d = np.load(calibration_data_fname)[0]
    imlib.showImage(slike[0], iTitle='Oznacena sredisca krogel na kalibru.')
    plt.plot(pts2d[:,0], pts2d[:,1],'mx',markersize=15)

pts2d = np.load(calibration_data_fname)[0]

# ---------- KALIBRIRAJ SISTEM ZA ZAJEM SLIK ----------
Tproj, pts3dproj = r3d.calibrate_irct(pts2d, pts3d)

# plt.close('all')
# imlib.showImage(slike[0], iTitle='Oznacena sredisca krogel na kalibru.')
plt.plot(pts2d[:,0], pts2d[:,1],'rx', markersize=15)
plt.plot(pts3dproj[:,0], pts3dproj[:,1],'gx', markersize=15)

# ---------- FILTRIRANJE 2D SLIK PRED POVRATNO PROJEKCIJO ----------
slika = np.squeeze(slike[0])
tip_filtra = 'hann'  # none, ram-lak, cosine, hann, hamming
slika_f = r3d.filter_projection(slika, tip_filtra, cut_off=0.75)
# imlib.showImage(slika_f, iCmap=cm.jet)

# ---------- REKONSTRUKCIJA 3D SLIKE ----------
# FBP = Filtered BackProjection
vol = r3d.fbp(slike[::1], koti[::1], Tproj,
              filter_type='hann', sampling_mm=3,
              out_fname=out_volume_fname)

[dx, dy, dz] = vol.shape

def linScale(iImage, oMax):
    k = oMax/np.max(iImage)
    oImage = k*iImage
    return oImage

def thresholdImage(iImage, iThreshold):
    oImage = 255 * np.array(iImage > iThreshold, dtype='uint8')
    return oImage

Thres = 100
Deci = 10

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

endZ = int(np.round(dz*0.90))

for z in range(endZ):
    dImage = vol[:,:,z]
    dImage = linScale(dImage, 255)
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

# # 3D izris
pointCoorX = pointCoorX[::Deci]
pointCoorY = pointCoorY[::Deci]
pointCoorZ = pointCoorZ[::Deci]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pointCoorX, pointCoorY, pointCoorZ)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()