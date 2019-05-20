import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import PIL.Image as im
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import cv2 as cv

from os.path import join

import reconlib as rl

# ---------- NALOZI SLIKE IZ MAPE ----------
pth = '/home/martin/Desktop/RV_Seminar_v2/rekonstrukcija_minimal'

#acquisition_data_pth = join(pth, 'acquisitions', 'klovn30')
acquisition_data_pth = join(pth, 'acquisitions', 'kocka')
calibration_image_fname = join(pth, 'calibration', 'kalibr.jpg')
calibration_data_fname = join(pth, 'calibration', 'tocke_kalibra_aneja.npy')
# out_volume_fname = join(pth, 'reconstructions', 'klovn3d.nrrd')
out_volume_fname = join(pth, 'reconstructions', 'ozilje.nrrd')

slike, koti = rl.load_images(acquisition_data_pth, proc=rl.rgb2gray)

# # obrezovanje
# ce potrebujes -> (ni se preverjena)funkcija crop_image

# rl.showImage(slike[10])
# plt.show()

# ---------- DOLOCI 3D KOORDINATE TOCK NA KALIBRU ----------
pts3d = rl.IRCT_CALIBRATION_OBJECT()
# plt.close('all')
# r3d.show_points_in_3d(pts3d)

# ---------- OZNACI 8 TOCK NA KALIBRU, KI NAJ OZNACUJEJO SREDISCE KROGEL ----------
if not os.path.exists(calibration_data_fname):
    calibration_image = np.array(im.open(calibration_image_fname))
    # calibration_image = calibration_image[200:slika_x-100, 300:slika_y-300]
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
#tip_filtra = 'hann'  # none, ram-lak, cosine, hann, hamming
tip_filtra = 'hann'
slika_f = rl.filter_projection(slika, tip_filtra, cut_off=0.9)
# rl.showImage(slika_f, iCmap=cm.jet)
# plt.show()

# ---------- REKONSTRUKCIJA 3D SLIKE ----------
# FBP = Filtered BackProjection
vol = rl.fbp(slike[::1], koti[::1], Tproj,
              filter_type='hann', sampling_mm=3,
              out_fname=out_volume_fname, cut_off=0.75)

# ---------- VOL -> POINT CLOUD ----------
pointCoorX, pointCoorY, pointCoorZ = rl.get_point_cloud(vol, 0.9, 5, 0, 1, 40)

# ---------- IZRIS POINT CLOUD ----------
rl.plot_point_cloud(pointCoorX, pointCoorY, pointCoorZ)

#TODO: stestiraj drug threshold: https://docs.opencv.org/3.1.0/d7/d4d/tutorial_py_thresholding.html

# import cv2
# dimage = vol[:,:,60]
# dimage = dimage + abs(np.min(dimage))

# def scaleImage(iImage):
#         oImage = (255/np.max(iImage))*iImage
#         return oImage
# oimage = scaleImage(dimage)
# oimage = cv2.medianBlure(oimage, 5)
# print(np.min(oimage), np.max(oimage))
# rl.showImage(oimage)
# plt.show()

print("konc")