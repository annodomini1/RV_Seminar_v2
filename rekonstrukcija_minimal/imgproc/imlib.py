'''
IMLIB: knjiznica funkcij za obdelavo in analizo slik

'''
import numpy as np
import PIL.Image as im
import scipy as sp
import colorsys
import SimpleITK as itk

import matplotlib.pyplot as plt
import matplotlib.cm as cm  # uvozi barvne lestvice
import matplotlib.colors as colors

from scipy.ndimage import convolve
from scipy.interpolate import interpn
from scipy.spatial import Delaunay
import mpl_toolkits.mplot3d as a3


def loadImageRaw(iPath, iSize, iFormat):
    '''
    Nalozi sliko iz raw datoteke

    Parameters
    ----------
    iPath : str 
        Pot do datoteke
    iSize : tuple 
        Velikost slike
    iFormat : str
        Tip vhodnih podatkov

    Returns
    ---------
    oImage : numpy array
        Izhodna slika

    '''
    oImage = np.fromfile(iPath, dtype=iFormat)  # nalozi raw datoteko
    oImage = np.reshape(oImage, iSize)  # uredi v matriko

    return oImage


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


def saveImageRaw(iImage, iPath, iFormat):
    '''
    Shrani sliko na disk

    Parameters
    ----------
    iImage : numpy.ndarray
        Vhodna slika za shranjevanje
    iPath : str
        Pot in ime datoteke, v katero zelimo sliko shraniti
    iFormat : str
        Tip podatkov v matriki slike

    Returns
    ---------
    None
    '''
    iImage = iImage.astype(iFormat)
    iImage.tofile(iPath)  # zapisi v datoteko


def loadImage(iPath):
    '''
    Nalozi sliko v standardnih formatih (bmp, jpg, png, tif, gif, idr.)
    in jo vrni kot matriko

    Parameters
    ----------
    iPath - str
        Pot do slike skupaj z imenom

    Returns
    ----------
    oImage - numpy.ndarray
        Vrnjena matricna predstavitev slike
    '''
    oImage = np.array(im.open(iPath))
    if oImage.ndim == 3:
        oImage = np.transpose(oImage, [2, 0, 1])
    elif oImage.ndim == 2:
        oImage = np.transpose(oImage, [1, 0])
    return oImage


def saveImage(iPath, iImage, iFormat):
    '''
    Shrani sliko v standardnem formatu (bmp, jpg, png, tif, gif, idr.)

    Parameters
    ----------
    iPath : str
        Pot do slike z zeljenim imenom slike
    iImage : numpy.ndarray
        Matricna predstavitev slike
    iFormat : str
        zeljena koncnica za sliko (npr. 'bmp')

    Returns
    ---------
    None

    '''
    if iImage.ndim == 3:
        iImage = np.transpose(iImage, [1, 2, 0])
    elif iImage.ndim == 2:
        iImage = np.transpose(iImage, [1, 0])
    img = im.fromarray(iImage)  # ustvari slikovni objekt iz matrike
    img.save(iPath.split('.')[0] + '.' + iFormat)


def drawLine(iImage, iValue, x1, y1, x2, y2):
    ''' Narisi digitalno daljico v sliko

        Parameters
        ----------
        iImage : numpy.ndarray
            Vhodna slika
        iValue : tuple, int
            Vrednost za vrisavanje (barva daljice).
            Uporabi tuple treh elementov za barvno sliko in int za sivinsko sliko
        x1 : int
            Zacetna x koordinata daljice
        y1 : int
            Zacetna y koordinata daljice
        x2 : int
            Koncna x koordinata daljice
        y2 : int
            Koncna y koordinata daljice
    '''

    oImage = iImage

    if iImage.ndim == 3:
        assert type(iValue) == tuple, 'Za barvno sliko bi paramter iValue moral biti tuple treh elementov'
        for rgb in range(3):
            drawLine(iImage[rgb, :, :], iValue[rgb], x1, y1, x2, y2)

    elif iImage.ndim == 2:
        assert type(iValue) == int, 'Za sivinsko sliko bi paramter iValue moral biti int'

        dx = np.abs(x2 - x1)
        dy = np.abs(y2 - y1)
        if x1 < x2:
            sx = 1
        else:
            sx = -1
        if y1 < y2:
            sy = 1
        else:
            sy = -1
        napaka = dx - dy

        x = x1
        y = y1

        while True:
            oImage[y - 1, x - 1] = iValue
            if x == x2 and y == y2:
                break
            e2 = 2 * napaka
            if e2 > -dy:
                napaka = napaka - dy
                x = x + sx
            if e2 < dx:
                napaka = napaka + dx
                y = y + sy

    return oImage


def crop_image(img, region):
    '''
    Crop image.

    :param img: Crop input grayscale image. 
    :param region: Region to crop defined by (uy,lx,by,rx). 
    :return: Cropped image.
    '''
    return np.array(img[region[1]:region[3], region[0]:region[2]])


def rgb2gray(img):
    '''
    Convert RGB image to grayscale.

    :param img: Input RGB image. 
    :return: Grayscale image.
    '''
    if img.ndim == 3:
        return np.mean(img.astype('float'), axis=-1)
    return img


def colorToGray(iImage):
    '''
    Pretvori barvno sliko v sivinsko.

    Parameters
    ---------
    iImage : numpy.ndarray
        Vhodna barvna slika

    Returns
    -------
    oImage : numpy.ndarray
        Sivinska slika
    '''
    dtype = iImage.dtype
    r = iImage[0, :, :].astype('float')
    g = iImage[1, :, :].astype('float')
    b = iImage[2, :, :].astype('float')

    return (r * 0.299 + g * 0.587 + b * 0.114).astype(dtype)


def computeHistogram(iImage, iNumBins, iRange=[], iDisplay=False, iTitle=''):
    '''
    Izracunaj histogram sivinske slike

    Parameters
    ---------
    iImage : numpy.ndarray
        Vhodna slika, katere histogram zelimo izracunati

    iNumBins : int
        stevilo predalov histograma

    iRange : tuple, list
        Minimalna in maksimalna sivinska vrednost 

    iDisplay : bool
        Vklopi/izklopi prikaz histograma v novem oknu

    iTitle : str
        Naslov prikaznega okna

    Returns
    -------
    oHist : numpy.ndarray
        Histogram sivinske slike
    oEdges: numpy.ndarray
        Robovi predalov histograma
    '''
    iImage = np.asarray(iImage)
    iRange = np.asarray(iRange)
    if iRange.size == 2:
        iMin, iMax = iRange
    else:
        iMin, iMax = np.min(iImage), np.max(iImage)
    oEdges = np.linspace(iMin, iMax + 1, iNumBins + 1)
    oHist = np.zeros([iNumBins, ])
    for i in range(iNumBins):
        idx = np.where((iImage >= oEdges[i]) * (iImage < oEdges[i + 1]))
        if idx[0].size > 0:
            oHist[i] = idx[0].size
    if iDisplay:
        plt.figure()
        plt.bar(oEdges[:-1], oHist)
        plt.suptitle(iTitle)

    return oHist, oEdges


def computeContrast(iImages):
    '''
    Izracunaj kontrast slik

    Parameters
    ---------
    iImages : list of numpy.ndarray
        Vhodne slike, na katerih zelimo izracunati kontrast

    Returns : list
        Seznam kontrastov za vsako vhodno sliko
    '''
    oM = np.zeros((len(iImages),))
    for i in range(len(iImages)):
        fmin = np.percentile(iImages[i].flatten(), 5)
        fmax = np.percentile(iImages[i].flatten(), 95)
        oM[i] = (fmax - fmin) / (fmax + fmin)
    return oM


def computeEffDynRange(iImages):
    '''
    Izracunaj efektivno dinamicno obmocje

    Parameters
    ----------
    iImages : numpy.ndarray
        Vhodne slike

    Returns
    --------
    oEDR : float
        Vrednost efektivnega dinamicnega obmocja
    '''
    L = np.zeros((len(iImages, )))
    sig = np.zeros((len(iImages),))
    for i in range(len(iImages)):
        L[i] = np.mean(iImages[i].flatten())
        sig[i] = np.std(iImages[i].flatten())
    oEDR = np.log2((L.max() - L.min()) / sig.mean())
    return oEDR


def computeSNR(iImage1, iImage2):
    '''
    Vrne razmerje signal/sum

    Paramters
    ---------
    iImage1, iImage2 : np.ndarray
        Sliki podrocij zanimanja, med katerima racunamo SNR

    Returns
    ---------
    oSNR : float
        Vrednost razmerja signal/sum
    '''
    mu1 = np.mean(iImage1.flatten())
    mu2 = np.mean(iImage2.flatten())

    sig1 = np.std(iImage1.flatten())
    sig2 = np.std(iImage2.flatten())

    oSNR = np.abs(mu1 - mu2) / np.sqrt(sig1 ** 2 + sig2 ** 2)

    return oSNR


def scaleImage(iImage, iSlopeA, iIntersectionB):
    '''
    Linearna sivinska preslikava y = a*x + b

    Parameters
    ----------
    iImage : numpy.ndarray
        Vhodna slika

    iSlopeA : float
        Linearni koeficient (a) v sivinski preslikavi

    iIntersectionB : float
        Konstantna vrednost (b) v sivinski preslikavi

    Returns
    --------
    oImage : numpy.ndarray
        Linearno preslikava sivinska slika
    '''
    iImageType = iImage.dtype
    iImage = np.array(iImage, dtype='float')
    oImage = iSlopeA * iImage + iIntersectionB
    # zaokrozevanje vrednosti
    if iImageType.kind in ('u', 'i'):
        oImage[oImage < np.iinfo(iImageType).min] = np.iinfo(iImageType).min
        oImage[oImage > np.iinfo(iImageType).max] = np.iinfo(iImageType).max
    return np.array(oImage, dtype=iImageType)


def windowImage(iImage, iCenter, iWidth):
    '''
    Linearno oknjenje y = (Ls-1)/w*(x-(c-w/2)

    Parameters
    ----------
    iImage : numpy.ndarray
        Vhodna slika

    iCenter : float
        Sivinska vrednost, ki doloca polozaj centra okna

    iWidth : float
        sirina okna, ki doloca razpon linearno preslikavnih vrednosti

    Returns
    --------
    oImage : numpy.ndarray
        Oknjena sivinska slika
    '''
    iImageType = iImage.dtype
    if iImageType.kind in ('u', 'i'):
        iMaxValue = np.iinfo(iImageType).max
        iMinValue = np.iinfo(iImageType).min
        iRange = iMaxValue - iMinValue
    else:
        iMaxValue = np.max(iImage)
        iMinValue = np.max(iImage)
        iRange = iMaxValue - iMinValue

    iSlopeA = iRange / float(iWidth)
    iInterceptB = - iSlopeA * (float(iCenter) - iWidth / 2.0)

    return scaleImage(iImage, iSlopeA, iInterceptB)


def thresholdImage(iImage, iThreshold):
    '''
    Upragovljanje y = x > t

    Parameters
    ----------
    iImage : numpy.ndarray
        Vhodna slika

    iThreshold : float
        Sivinska vrednost, ki doloca prag

    Returns
    --------
    oImage : numpy.ndarray
        Upragovljena binarna slika
    '''
    iImage = np.asarray(iImage)
    oImage = 255 * np.array(iImage > iThreshold, dtype='uint8')
    return oImage


def gammaImage(iImage, iGamma):
    '''
    Upragovljanje y = (Ls-1)(x/(Lr-1))^gamma

    Parameters
    ----------
    iImage : numpy.ndarray
        Vhodna slika

    iGamma : float
        Vrednost gama

    Returns
    --------
    oImage : numpy.ndarray
        Gama preslikana slika
    '''
    iImage = np.asarray(iImage)
    iImageType = iImage.dtype
    iImage = np.array(iImage, dtype='float')
    # preberi mejne vrednosti in obmocje vrednosti
    if iImageType.kind in ('u', 'i'):
        iMaxValue = np.iinfo(iImageType).max
        iMinValue = np.iinfo(iImageType).min
        iRange = iMaxValue - iMinValue
    else:
        iMaxValue = np.max(iImage)
        iMinValue = np.max(iImage)
        iRange = iMaxValue - iMinValue
    # izvedi gamma preslikavo
    iImage = (iImage - iMinValue) / float(iRange)
    oImage = iImage ** iGamma
    oImage = float(iRange) * oImage + iMinValue
    # zaokrozevanje vrednosti
    if iImageType.kind in ('u', 'i'):
        oImage[oImage < np.iinfo(iImageType).min] = np.iinfo(iImageType).min
        oImage[oImage > np.iinfo(iImageType).max] = np.iinfo(iImageType).max
    # vrni sliko v originalnem formatu
    return np.array(oImage, dtype=iImageType)


def convertImageColorSpace(iImage, iConversionType):
    '''
    Pretvorba barvne slike med barvnima prostoroma RGB in HSV

    Parameters
    ----------
    iImage : numpy.ndarray
        Vhodna RGB ali HSV slika

    iConversionType : str
        'RGBtoHSV' ali 'HSVtoRGB'

    Returns
    --------
    oImage : numpy.ndarray
        Preslikana RGB ali HSV slika
    '''
    iImage = np.asarray(iImage)
    colIdx = [iImage.shape[i] == 3 for i in range(len(iImage.shape))]
    iImage = np.array(iImage, dtype='float')

    if iConversionType == 'RGBtoHSV':
        if colIdx.index(True) == 0:
            r = iImage[0, :, :];
            g = iImage[1, :, :];
            b = iImage[2, :, :];
        elif colIdx.index(True) == 1:
            r = iImage[:, 0, :];
            g = iImage[:, 1, :];
            b = iImage[:, 2, :];
        elif colIdx.index(True) == 2:
            r = iImage[:, :, 0];
            g = iImage[:, :, 1];
            b = iImage[:, :, 2];

        r = r / 255.0
        g = g / 255.0
        b = b / 255.0

        h = np.zeros_like(r)
        s = np.zeros_like(r)
        v = np.zeros_like(r)

        Cmax = np.maximum(r, np.maximum(g, b))
        Cmin = np.minimum(r, np.minimum(g, b))
        delta = Cmax - Cmin + 1e-7

        h[Cmax == r] = 60.0 * ((g[Cmax == r] - b[Cmax == r]) / delta[Cmax == r] % 6.0)
        h[Cmax == g] = 60.0 * ((b[Cmax == g] - r[Cmax == g]) / delta[Cmax == g] + 2.0)
        h[Cmax == b] = 60.0 * ((r[Cmax == b] - g[Cmax == b]) / delta[Cmax == b] + 4.0)

        s[delta != 0.0] = delta[delta != 0.0] / (Cmax[delta != 0.0] + 1e-7)

        v = Cmax

        # ustvari izhodno sliko
        oImage = np.zeros_like(iImage)
        if colIdx.index(True) == 0:
            oImage[0, :, :] = h;
            oImage[1, :, :] = s;
            oImage[2, :, :] = v;
        elif colIdx.index(True) == 1:
            oImage[:, 0, :] = h;
            oImage[:, 1, :] = s;
            oImage[:, 2, :] = v;
        elif colIdx.index(True) == 2:
            oImage[:, :, 0] = h;
            oImage[:, :, 1] = s;
            oImage[:, :, 2] = v;

        return oImage

    elif iConversionType == 'HSVtoRGB':
        if colIdx.index(True) == 0:
            h = iImage[0, :, :];
            s = iImage[1, :, :];
            v = iImage[2, :, :];
        elif colIdx.index(True) == 1:
            h = iImage[:, 0, :];
            s = iImage[:, 1, :];
            v = iImage[:, 2, :];
        elif colIdx.index(True) == 2:
            h = iImage[:, :, 0];
            s = iImage[:, :, 1];
            v = iImage[:, :, 2];

        C = v * s
        X = C * (1.0 - np.abs(((h / 60.0) % 2.0) - 1))
        m = v - C

        r = np.zeros_like(h)
        g = np.zeros_like(h)
        b = np.zeros_like(h)

        r[(h >= 0.0) * (h < 60.0)] = C[(h >= 0.0) * (h < 60.0)]
        g[(h >= 0.0) * (h < 60.0)] = X[(h >= 0.0) * (h < 60.0)]

        r[(h >= 60.0) * (h < 120.0)] = X[(h >= 60.0) * (h < 120.0)]
        g[(h >= 60.0) * (h < 120.0)] = C[(h >= 60.0) * (h < 120.0)]

        g[(h >= 120.0) * (h < 180.0)] = C[(h >= 120.0) * (h < 180.0)]
        b[(h >= 120.0) * (h < 180.0)] = X[(h >= 120.0) * (h < 180.0)]

        g[(h >= 180.0) * (h < 240.0)] = X[(h >= 180.0) * (h < 240.0)]
        b[(h >= 180.0) * (h < 240.0)] = C[(h >= 180.0) * (h < 240.0)]

        r[(h >= 240.0) * (h < 300.0)] = X[(h >= 240.0) * (h < 300.0)]
        b[(h >= 240.0) * (h < 300.0)] = C[(h >= 240.0) * (h < 300.0)]

        r[(h >= 300.0) * (h < 360.0)] = C[(h >= 300.0) * (h < 360.0)]
        b[(h >= 300.0) * (h < 360.0)] = X[(h >= 300.0) * (h < 360.0)]

        r = r + m
        g = g + m
        b = b + m

        # ustvari izhodno sliko
        oImage = np.zeros_like(iImage)
        print(oImage.dtype)
        if colIdx.index(True) == 0:
            oImage[0, :, :] = r;
            oImage[1, :, :] = g;
            oImage[2, :, :] = b;
        elif colIdx.index(True) == 1:
            oImage[:, 0, :] = r;
            oImage[:, 1, :] = g;
            oImage[:, 2, :] = b;
        elif colIdx.index(True) == 2:
            oImage[:, :, 0] = r;
            oImage[:, :, 1] = g;
            oImage[:, :, 2] = b;

        # zaokrozevanje vrednosti
        oImage = 255.0 * oImage
        oImage[oImage > 255.0] = 255.0
        oImage[oImage < 0.0] = 0.0

        oImage = np.array(oImage, dtype='uint8')

        return oImage


def csRGBToHSV(iImage):
    '''
    Pretvorba barvne slike iz RGB v HSV barvni prostor

    Parameters
    ----------
    iImage : numpy.ndarray
        Vhodna RGB slika

    Returns
    --------
    oImage : numpy.ndarray
        Preslikana HSV slika
    '''
    iImage = np.array(iImage, dtype='float')
    oImage = np.zeros_like(iImage)
    for y in range(iImage.shape[0]):
        for x in range(iImage.shape[1]):
            r, g, b = iImage[y, x, :]
            oImage[y, x, :] = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
    oImage[:, :, 0] = oImage[:, :, 0] * 360.0
    return oImage


def csHSVToRGB(iImage):
    '''
    Pretvorba barvne slike iz RGB v HSV barvni prostor

    Parameters
    ----------
    iImage : numpy.ndarray
        Vhodna HSV slika

    Returns
    --------
    oImage : numpy.ndarray
        Preslikana RGB slika
    '''
    iImage = np.array(iImage, dtype='float')
    oImage = np.zeros_like(iImage)
    for y in range(iImage.shape[0]):
        for x in range(iImage.shape[1]):
            h, s, v = iImage[y, x, :]
            oImage[y, x, :] = colorsys.hsv_to_rgb(h / 360.0, s, v)
    oImage = oImage * 255.0
    oImage[oImage > 255.0] = 255.0
    oImage[oImage < 0.0] = 0.0
    oImage = np.array(oImage, dtype='uint8')
    return oImage


def discreteConvolution2D(iImage, iKernel):
    '''
    Diskretna 2D konvolucija slike s poljubnim jedrom

    Parameters
    ----------
    iImage : numpy.ndarray
        Vhodna sivinska slika

    iKernel : numpy.ndarray
        Jedro ali matrika za konvolucijo

    Returns
    --------
    oImage : numpy.ndarray
        Z jedrom konvolirana vhodna slika
    '''
    # pretvori vhodne spremenljivke v np polje in
    # inicializiraj izhodno np polje
    iImage = np.asarray(iImage)
    iKernel = np.asarray(iKernel)
    return convolve(iImage, iKernel, mode='nearest')
    # DIREKTNA IMPLEMENTACIJA
    oImage = np.zeros_like(iImage).astype('float')
    # preberi velikost slike in jedra
    dy, dx = iImage.shape
    dv, du = iKernel.shape
    # izracunaj konvolucijo
    for y in range(dy):
        for x in range(dx):
            for v in range(dv):
                for u in range(du):
                    tx = x - u + du / 2
                    ty = y - v + dv / 2
                    if tx >= 0 and tx < dx and ty >= 0 and ty < dy:
                        oImage[y, x] = oImage[y, x] + \
                                       float(iImage[ty, tx]) * float(iKernel[v, u])
    if iImage.dtype.kind in ('u', 'i'):
        oImage[oImage < np.iinfo(iImage.dtype).min] = np.iinfo(iImage.dtype).min
        oImage[oImage > np.iinfo(iImage.dtype).max] = np.iinfo(iImage.dtype).max
    return np.array(oImage, dtype=iImage.dtype)


def discreteGaussian2D(iSigma):
    '''
    Diskretno 2D Gaussovo jedro za glajenje slik

    Parameters
    ----------
    iSigma : float
        Standardna deviacija simetricnega 2D Gaussovega jedra

    Returns
    --------
    oKernel : numpy.ndarray
        2D Gaussovo jedro
    '''
    iKernelSize = int(2 * np.ceil(3 * iSigma) + 1)
    oKernel = np.zeros([iKernelSize, iKernelSize])
    k2 = np.floor(iKernelSize / 2);
    s2 = iSigma ** 2.0
    for y in range(oKernel.shape[1]):
        for x in range(oKernel.shape[0]):
            oKernel[y, x] = np.exp(-((x - k2) ** 2 + (y - k2) ** 2) / 2.0 / s2) / s2 / 2.0 / np.pi
    return oKernel


def interpolate0Image2D(iImage, iCoorX, iCoorY, use_builtin=True):
    '''
    Funkcija za interpolacijo nictega reda

    Parameters
    ----------
    iImage : numpy.ndarray
        Vhodna sivinska slika

    iCoorX : numpy.ndarray
        Polje X koordinat za interpolacijo

    iCoorY : numpy.ndarray
        Polje Y koordinat za interpolacijo

    Returns
    --------
    oImage : numpy.ndarray
        Interpolirane vrednosti v vhodnih koordinatah X in Y
    '''
    # pretvori vhodne spremenljivke v np polje
    iImage = np.asarray(iImage)
    iCoorX = np.asarray(iCoorX)
    iCoorY = np.asarray(iCoorY)
    # preberi velikost slike in jedra
    dy, dx = iImage.shape
    # ustvari 2d polje koordinat iz 1d vhodnih koordinat (!!!)
    if np.size(iCoorX) != np.size(iCoorY):
        print('Stevilo X in Y koordinat je razlicno!')
        iCoorX, iCoorY = np.meshgrid(iCoorX, iCoorY, sparse=False, indexing='xy')

    if use_builtin:
        return interpn((np.arange(dy), np.arange(dx)), iImage.astype('float'),
                       np.dstack((iCoorY, iCoorX)), method='nearest', bounds_error=False).astype(iImage.dtype)

    # DIREKTNA IMPLEMENTACIJA
    # zaokrozi na najblizjo celostevilsko vrednost (predstavlja indeks!)
    oShape = iCoorX.shape
    iCoorX = np.round(iCoorX);
    iCoorX = iCoorX.flatten()
    iCoorY = np.round(iCoorY);
    iCoorY = iCoorY.flatten()
    # ustvari izhodno polje
    oImage = np.zeros(oShape);
    oImage = oImage.flatten()
    oImage = np.array(oImage, dtype=iImage.dtype)
    print(iCoorX.shape)
    print(iCoorY.shape)
    # priredi vrednosti
    for idx in range(oImage.size):
        tx = iCoorX[idx]
        ty = iCoorY[idx]
        if tx >= 0 and tx < dx and ty >= 0 and ty < dy:
            oImage[idx] = iImage[ty, tx]
    # vrni izhodno sliko
    return np.reshape(oImage, oShape)


def interpolate1Image2D(iImage, iCoorX, iCoorY, use_builtin=True):
    '''
    Funkcija za interpolacijo prvega reda

    Parameters
    ----------
    iImage : numpy.ndarray
        Vhodna sivinska slika

    iCoorX : numpy.ndarray
        Polje X koordinat za interpolacijo

    iCoorY : numpy.ndarray
        Polje Y koordinat za interpolacijo
        
    use_builtin : bool

    Returns
    --------
    oImage : numpy.ndarray
        Interpolirane vrednosti v vhodnih koordinatah X in Y
    '''
    # pretvori vhodne spremenljivke v np polje
    iImage = np.asarray(iImage)
    iCoorX = np.asarray(iCoorX)
    iCoorY = np.asarray(iCoorY)
    # preberi velikost slike in jedra
    dy, dx = iImage.shape
    # ustvari 2d polje koordinat iz 1d vhodnih koordinat (!!!)
    if np.size(iCoorX) != np.size(iCoorY):
        print('Stevilo X in Y koordinat je razlicno!')
        iCoorX, iCoorY = np.meshgrid(iCoorX, iCoorY, sparse=False, indexing='xy')

    if use_builtin:
        return interpn((np.arange(dy), np.arange(dx)), iImage.astype('float'),
                       np.dstack((iCoorY, iCoorX)), method='linear', bounds_error=False).astype(iImage.dtype)

    # DIREKTNA IMPLEMENTACIJA
    # pretvori v linearno polje
    oShape = iCoorX.shape
    iCoorX = iCoorX.flatten()
    iCoorY = iCoorY.flatten()
    # ustvari izhodno polje, pretvori v linearno polje
    oImage = np.zeros(oShape);
    oImage = oImage.flatten()
    oImage = np.array(oImage, dtype='float')
    print(iCoorX.shape)
    print(iCoorY.shape)
    # priredi vrednosti
    for idx in range(oImage.size):
        lx = np.floor(iCoorX[idx])
        ly = np.floor(iCoorY[idx])
        sx = float(iCoorX[idx]) - lx
        sy = float(iCoorY[idx]) - ly
        if lx >= 0 and lx < (dx - 1) and ly >= 0 and ly < (dy - 1):
            # izracunaj utezi
            a = (1 - sx) * (1 - sy)
            b = sx * (1 - sy)
            c = (1 - sx) * sy
            d = sx * sy
            # izracunaj izhodno vrednost
            oImage[idx] = a * iImage[ly, lx] + \
                          b * iImage[ly, lx + 1] + \
                          c * iImage[ly + 1, lx] + \
                          d * iImage[ly + 1, lx + 1]
    if iImage.dtype.kind in ('u', 'i'):
        oImage[oImage < np.iinfo(iImage.dtype).min] = np.iinfo(iImage.dtype).min
        oImage[oImage > np.iinfo(iImage.dtype).max] = np.iinfo(iImage.dtype).max
    return np.array(np.reshape(oImage, oShape), dtype=iImage.dtype)


def decimateImage2D(iImage, iLevel):
    '''
    Funkcija za piramidno decimacijo

    Parameters
    ----------
    iImage : numpy.ndarray
        Vhodna sivinska slika

    iLevel : int
        stevilo decimacij s faktorjem 2

    Returns
    --------
    oImage : numpy.ndarray
        Decimirana slika
    '''
    print('Decimacija pri iLevel = ', iLevel)
    # pretvori vhodne spremenljivke v np polje
    iImage = np.asarray(iImage)
    iImageType = iImage.dtype
    # gaussovo jedro za glajenje
    iKernel = np.array(((1 / 16, 1 / 8, 1 / 16), (1 / 8, 1 / 4, 1 / 8), (1 / 16, 1 / 8, 1 / 16)))
    # glajenje slike pred decimacijo
    # iImage = discreteConvolution2D( iImage, iKernel )
    # hitrejsa verzija glajenja
    iImage = convolve(iImage, iKernel, mode='nearest')
    # decimacija s faktorjem 2
    iImage = iImage[::2, ::2]
    # vrni sliko oz. nadaljuj po piramidi
    if iLevel <= 1:
        return np.array(iImage, dtype=iImageType)
    else:
        return decimateImage2D(iImage, iLevel - 1)


def imageGradient(iImage):
    """
    Gradient slike s Sobelovim operatorjem
    
    Parameters
    ----------
    iImage : numpy.ndarray
        Vhodna sivinska slika    
        
    Returns
    --------
    oGx : numpy.ndarray
        Sivinska slika gradienta vzdolz x osi
        
    oGy : numpy.ndarray
        Sivinska slika gradienta vzdolz y osi                
        
    """
    iImage = np.array(iImage, dtype='float')
    iSobel = np.array(((-1, 0, 1), (-2, 0, 2), (-1, 0, 1)))
    oGx = convolve(iImage, iSobel, mode='nearest')
    oGy = convolve(iImage, np.transpose(iSobel), mode='nearest')
    return oGx, oGy


def responseHarris(iImage, iKappa, iSigma):
    """
    Odziv Harrisovega detektorja kotov
    
    Parameters
    ----------
    iImage : numpy.ndarray
        Vhodna sivinska slika
            
    iKappa : float
        Parameter obcutljivost Harris detektorja oglisc, vrednosti med 0 in 1/4
                    
    iSigma : float
        Stopnja glajenja                    
        
    Returns
    --------
    oQH : numpy.ndarray
        Sivinska slika polja odziva Harris funkcije za detekcijo oglisc
        
    """
    iImage = np.array(iImage, dtype='float')
    # odvod slike
    oGx, oGy = imageGradient(iImage)
    # komponente matrike M
    A = oGx ** 2
    B = oGy ** 2
    C = oGx * oGy
    # Gaussovo jedro za iSigma
    iGaussKernel = discreteGaussian2D(iSigma)
    # glajenje komponent matrike M
    A = convolve(A, iGaussKernel, mode='nearest')
    B = convolve(B, iGaussKernel, mode='nearest')
    C = convolve(C, iGaussKernel, mode='nearest')
    # odziv Harrisovega detektorja oglisc
    trM = A + B
    detM = A * B - C ** 2
    oQH = detM - iKappa * (trM ** 2)
    return oQH


def findLocalMax(iArray, iThreshold=None):
    """
    Poisci lokalne optimume v danem polju
    
    Parameters
    ----------
    iImage : numpy.ndarray
        Vhodna sivinska slika
            
    iThreshold : float
        Vrednost v polju, ki mora biti presezena v lokalnem maksimumu
                           
    Returns
    --------
    oLocalMax : numpy.ndarray
        Polje Mx2 koordinat (x,y) lokalnih maksimumov     
    """
    iArray = np.asarray(iArray)
    dy, dx = iArray.shape
    oLocalMax = []
    # find local max
    for y in range(dy):
        for x in range(dx):
            isMax = True
            cval = iArray[y, x]
            # preskoci element ce je cval manjse od iThreshold
            if iThreshold is not None:
                if cval < iThreshold:
                    continue
            # preveri vse elemente v 3x3 sosescini
            gx, gy = np.meshgrid((x - 1, x, x + 1), (y - 1, y, y + 1), sparse=False)
            gx = gx.flatten();
            gy = gy.flatten()
            for i in range(gx.size):
                isWithinImage = gx[i] >= 0 and gx[i] < dx and gy[i] >= 0 and gy[i] < dy
                isNotCenter = gx[i] != x or gy[i] != y
                if isNotCenter and isWithinImage:
                    if cval <= iArray[gy[i], gx[i]]:
                        isMax = False
                        break
            if isMax:
                oLocalMax.append((x, y))
                # return list of local max
    return np.array(oLocalMax)


def cornerHarris(iImage, iKappa, iTmin, iSigma = 3.0):
    """
    Izloci robne tocke s Harris detektorjem oglisc
    
    
    Parameters
    ----------
    iImage : numpy.ndarray
        Vhodna sivinska slika
            
    iKappa : float
        Parameter obcutljivost Harris detektorja oglisc, vrednosti med 0 in 1/4
                    
    iSigma : float
        Stopnja glajenja                    
        
    Returns
    --------
    oCorners : numpy.ndarray
        Polje Mx2 koordinat (x,y) najdenih oglisc    
    """
    oQHt = responseHarris(iImage, iKappa, iSigma)
    oCorners = findLocalMax(oQHt, iTmin)
    return oCorners


def emphasizeLinear(iImage, iSigma, iBeta):
    """
    Poudarjanje podolgovatih struktur
    
    Parameters
    ----------
    iImage : numpy.ndarray
        Vhodna sivinska slika
                               
    iSigma : float
        Stopnja glajenja
                            
    iBeta : float
        Parameter obcutljivosti na linearne strukture                            
        
    Returns
    --------
    oQLA : numpy.ndarray
        Sivinska slika polja odziva funkcije za detekcijo linearnih segmentov    
    """
    iImage = np.array(iImage, dtype='float')
    # odvod slike
    oGx, oGy = imageGradient(iImage)
    # komponente matrike M
    A = oGx ** 2
    B = oGy ** 2
    C = oGx * oGy
    # Gaussovo jedro za iSigma
    iGaussKernel = discreteGaussian2D(iSigma)
    # glajenje komponent matrike M
    A = convolve(A, iGaussKernel, mode='nearest')
    B = convolve(B, iGaussKernel, mode='nearest')
    C = convolve(C, iGaussKernel, mode='nearest')
    trM = A + B
    detM = A * B - C ** 2
    # izracun lastnih vrednosti
    D = (trM / 2.0) ** 2 - detM
    D[D < 0] = 0;
    D = np.sqrt(D)
    l1 = trM / 2.0 + D
    l2 = trM / 2.0 - D
    # linearna anizotropicnost
    oQLA = (l1 - l2) / (l1 + l2 + iBeta)
    return oQLA


def smoothImageGradientAbs(iImage, iSigma):
    iImage = iImage.astype('float')

    # 1. dusenje suma z glajenjem
    if iSigma > 0:
        iGaussKernel = discreteGaussian2D(iSigma)
        oImage = convolve(iImage, iGaussKernel, mode='nearest')
    else:
        oImage = iImage

    # 2. Izracun gradienta slike (magnituda + orientacija)
    oGx, oGy = imageGradient(oImage)

    oGAbs = np.sqrt(oGx ** 2 + oGy ** 2)
    oGPhi = np.arctan2(oGy, oGx)

    return oGAbs, oGPhi


def nonMaximaSuppression(iGAbs, iGPhi):
    # 2. gradient

    iGPhi = iGPhi.flatten();  # flatten for further processing
    iGPhi[iGPhi < 0] = iGPhi[iGPhi < 0] + np.pi;

    distPi0 = np.minimum(iGPhi, np.abs(iGPhi - np.pi));
    distPi45 = np.abs(iGPhi - np.pi / 4);
    distPi90 = np.abs(iGPhi - np.pi / 2);
    distPi135 = np.abs(iGPhi - 3 * np.pi / 4);

    minIdx = np.argmin(np.vstack((distPi0, distPi45, distPi90, distPi135)),
                       axis=0)  # indices of the minimum along an axis

    iGPhi = np.reshape(minIdx, iGAbs.shape)

    # 3. Iskanje maksimalnih vrednosti gradienta
    oEdge = np.zeros_like(iGAbs)

    dy, dx = oEdge.shape

    for y in range(1, dy - 1):
        for x in range(1, dx - 1):

            if iGPhi[y, x] == 0:
                if iGAbs[y, x] > iGAbs[y, x - 1] and iGAbs[y, x] > iGAbs[y, x + 1]:
                    oEdge[y, x] = 1;
            elif iGPhi[y, x] == 1:
                if iGAbs[y, x] > iGAbs[y - 1, x - 1] and iGAbs[y, x] > iGAbs[y + 1, x + 1]:
                    oEdge[y, x] = 1;
            elif iGPhi[y, x] == 2:
                if iGAbs[y, x] > iGAbs[y - 1, x] and iGAbs[y, x] > iGAbs[y + 1, x]:
                    oEdge[y, x] = 1;
            elif iGPhi[y, x] == 3:
                if iGAbs[y, x] > iGAbs[y - 1, x + 1] and iGAbs[y, x] > iGAbs[y + 1, x - 1]:
                    oEdge[y, x] = 1;

    return oEdge


def connectEdge(iEdge, iGAbs, iThreshold):
    # 4. Povezovanje robov
    iGAbs = iGAbs / np.max(iGAbs[iEdge > 0])  # normaliziramo med 0 in 1
    iGAbs[iGAbs > 1] = 1

    edgeWeakIdx = np.nonzero(iEdge * (iGAbs > iThreshold[0]) * (iGAbs <= iThreshold[1]))
    edgeStrongIdx = np.nonzero(iEdge * (iGAbs > iThreshold[1]))

    # connect edges
    oEdge = np.zeros_like(iEdge)
    oEdge[edgeStrongIdx] = 1;

    yxl = np.asarray(edgeWeakIdx)

    oEdgeP = np.zeros_like(oEdge)

    iterN = 0

    while np.sum(np.abs(oEdge - oEdgeP)) > 0:

        iterN = iterN + 1
        print(iterN)

        oEdgeP = np.copy(oEdge)

        for i in range(yxl.shape[1]):
            x = yxl[1, i]
            y = yxl[0, i]

            if oEdge[y, x] > 0:
                continue

            if x > 0 and y > 0 and y < oEdge.shape[0] - 1 and x < oEdge.shape[1] - 1:

                gx, gy = np.meshgrid((x - 1, x, x + 1), (y - 1, y, y + 1), sparse=False)
                gx = gx.flatten()
                gy = gy.flatten()

                if np.sum(oEdge[gy, gx] == 1) > 0:
                    oEdge[y, x] = 1;

    return oEdge


def edgeCanny(iImage, iThreshold, iSigma):
    oGAbs, oGPhi = smoothImageGradientAbs(iImage, iSigma)
    oEdge = nonMaximaSuppression(oGAbs, oGPhi)
    oEdge = connectEdge(oEdge, oGAbs, iThreshold)

    # enojno upragavljanje
    #    oGAbs = oGAbs / np.max(oGAbs[oEdge>0]); # normaliziramo med 0 in 1
    #    oGAbs[oGAbs>1] = 1;
    #    oEdge[oGAbs < iThreshold[0]] = 0;
    #
    return oEdge


def transAffine2D(iScale=(1, 1), iTrans=(0, 0), iRot=0, iShear=(0, 0)):
    '''
    Ustvari poljubno 2D afino preslikavo v obliki 3x3 homogene matrike

    Parameters
    ----------
    iScale : tuple, list
        Skaliranje vzdolz x in y

    iTrans : tuple, list
        Translacija vzdolz x in y

    iRot : float
        Kot rotacije

    iShear : tuple, list
        Strig vzdolz x in y

    Returns
    --------
    oMat2D : numpy.ndarray
        Homogena 3x3 transformacijska matrika

    '''
    iRot = iRot * np.pi / 180
    oMatScale = np.array(((iScale[0], 0, 0), (0, iScale[1], 0), (0, 0, 1)))
    oMatTrans = np.array(((1, 0, iTrans[0]), (0, 1, iTrans[1]), (0, 0, 1)))
    oMatRot = np.array(((np.cos(iRot), -np.sin(iRot), 0), \
                        (np.sin(iRot), np.cos(iRot), 0),
                        (0, 0, 1)))
    oMatShear = np.array(((1, iShear[0], 0), (iShear[1], 1, 0), (0, 0, 1)))
    # ustvari izhodno matriko
    oMat2D = np.dot(oMatTrans, np.dot(oMatShear, np.dot(oMatRot, oMatScale)))
    return oMat2D


def transRigid3D(trans=(0, 0, 0), rot=(0, 0, 0)):
    '''
    Rigid body transformation

    :param trans: Translation 3-vector (tx,ty,tz).
    :param rot: Rotation vector (rx,ry,rz).
    :return: Rigid-body 4x4 transformation matrix.
    '''
    Trotx = np.array(((1, 0, 0, 0), \
                      (0, np.cos(rot[0]), -np.sin(rot[0]), 0), \
                      (0, np.sin(rot[0]), np.cos(rot[0]), 0), \
                      (0, 0, 0, 1)))
    Troty = np.array(((np.cos(rot[1]), 0, np.sin(rot[1]), 0), \
                      (0, 1, 0, 0), \
                      (-np.sin(rot[1]), 0, np.cos(rot[1]), 0), \
                      (0, 0, 0, 1)))
    Trotz = np.array(((np.cos(rot[2]), -np.sin(rot[2]), 0, 0), \
                      (np.sin(rot[2]), np.cos(rot[2]), 0, 0), \
                      (0, 0, 1, 0), \
                      (0, 0, 0, 1)))
    Ttrans = np.array(((1, 0, 0, trans[0]), \
                       (0, 1, 0, trans[1]), \
                       (0, 0, 1, trans[2]), \
                       (0, 0, 0, 1)))
    return np.dot(np.dot(np.dot(Trotx, Troty), Trotz), Ttrans)


def addHomogCoord2D(iPts):
    '''
    Seznamu 2D koordinat dodaj homogeno koordinato

    Parameters
    ----------
    iPts : numpy.ndarray
        Polje Nx2 koordinat x in y

    Returns
    --------
    oPts : numpy.ndarray
        Polje Nx3 homogenih koordinat x in y

    '''
    if iPts.shape[-1] == 3:
        return iPts
    iPts = np.hstack((iPts, np.ones((iPts.shape[0], 1))))
    return iPts


def mapAffineInterp2D(iPtsRef, iPtsMov):
    '''
    Afina interpolacijska poravnava na osnovi 3 pripadajocih parov tock

    Parameters
    ----------
    iPtsRef : numpy.ndarray
        Polje 3x2 koordinat x in y (lahko tudi v homogeni obliki)

    iPtsMov : numpy.ndarray
        Polje 3x2 koordinat x in y (lahko tudi v homogeni obliki)

    Returns
    --------
    oMat2D : numpy.ndarray
        Homogena 3x3 transformacijska matrika
    '''
    # dodaj homogeno koordinato
    iPtsRef = addHomogCoord2D(iPtsRef)
    iPtsMov = addHomogCoord2D(iPtsMov)
    # afina interpolacija
    iPtsRef = iPtsRef.transpose()
    iPtsMov = iPtsMov.transpose()
    oMat2D = np.dot(iPtsRef, np.linalg.inv(iPtsMov))
    return oMat2D


def mapAffineApprox2D(iPtsRef, iPtsMov, iUsePseudoInv=False):
    '''
    Afina aproksimacijska poravnava na osnovi N pripadajocih parov tock

    Parameters
    ----------
    iPtsRef : numpy.ndarray
        Polje Nx2 koordinat x in y (lahko tudi v homogeni obliki)

    iPtsMov : numpy.ndarray
        Polje Nx2 koordinat x in y (lahko tudi v homogeni obliki)

    Returns
    --------
    oMat2D : numpy.ndarray
        Homogena 3x3 transformacijska matrika
    '''
    if iUsePseudoInv:
        # po potrebi dodaj homogeno koordinato
        iPtsRef = addHomogCoord2D(iPtsRef)
        iPtsMov = addHomogCoord2D(iPtsMov)
        # afina aproksimacija (s psevdoinverzom)
        iPtsRef = iPtsRef.transpose()
        iPtsMov = iPtsMov.transpose()
        # psevdoinverz
        oMat2D = np.dot(iPtsRef, np.linalg.pinv(iPtsMov))
        # psevdoinverz na dolgo in siroko:
        # oMat2D = iPtsMov * iPtsRef.transpose() * \
        # np.linalg.inv( iPtsRef * iPtsRef.transpose() )
    else:
        # izloci koordinate
        x = np.array(iPtsMov[:, 0], dtype='float')
        y = np.array(iPtsMov[:, 1], dtype='float')

        u = np.array(iPtsRef[:, 0], dtype='float')
        v = np.array(iPtsRef[:, 1], dtype='float')

        # doloci povprecja
        uxm = np.mean(u * x)
        uym = np.mean(u * y)
        vxm = np.mean(v * x)
        vym = np.mean(v * y)
        um = np.mean(u)
        vm = np.mean(v)
        xxm = np.mean(x * x)
        xym = np.mean(x * y)
        yym = np.mean(y * y)
        xm = np.mean(x)
        ym = np.mean(y)
        # sestavi vektor in matriko linearnega sistema
        pv = np.array((uxm, uym, um, vxm, vym, vm))
        Pm = np.array(((xxm, xym, xm, 0, 0, 0), \
                       (xym, yym, ym, 0, 0, 0), \
                       (xm, ym, 1, 0, 0, 0), \
                       (0, 0, 0, xxm, xym, xm), \
                       (0, 0, 0, xym, yym, ym), \
                       (0, 0, 0, xm, ym, 1)))
        t = np.dot(np.linalg.inv(Pm), pv)
        oMat2D = np.array(((t[0], t[1], t[2]), \
                           (t[3], t[4], t[5]), \
                           (0, 0, 1)))
    return oMat2D


def findCorrespondingPoints(iPtsRef, iPtsMov):
    '''
    Iskanje pripadajocih parov tock kot paroma najblizje tocke

    Parameters
    ----------
    iPtsRef : numpy.ndarray
        Polje Mx2 koordinat x in y (lahko tudi v homogeni obliki)

    iPtsMov : numpy.ndarray
        Polje Nx2 koordinat x in y (lahko tudi v homogeni obliki)

    Returns
    --------
    oPtsRef : numpy.ndarray
        Polje Kx3 homogenih koordinat x in y (ali v homogeni obliki), ki pripadajo oPtsMov (K=min(M,N))

    oPtsMov : numpy.ndarray
        Polje Kx3 homogenih koordinat x in y (ali v homogeni obliki), ki pripadajo oPtsRef (K=min(M,N))
    '''
    # inicializiraj polje indeksov
    idxPair = -np.ones((iPtsRef.shape[0], 1)).astype('int32')
    idxDist = np.ones((iPtsRef.shape[0], iPtsMov.shape[0]))
    for i in range(iPtsRef.shape[0]):
        for j in range(iPtsMov.shape[0]):
            idxDist[i, j] = np.sum((iPtsRef[i, :2] - iPtsMov[j, :2]) ** 2)
    # doloci bijektivno preslikavo
    while not np.all(idxDist == np.inf):
        i, j = np.where(idxDist == np.min(idxDist))
        idxPair[i[0]] = j[0]
        idxDist[i[0], :] = np.inf
        idxDist[:, j[0]] = np.inf
        # doloci pare tock
    idxValid, idxNotValid = np.where(idxPair >= 0)
    idxValid = np.array(idxValid)
    iPtsRef_t = iPtsRef[idxValid, :]
    iPtsMov_t = iPtsMov[idxPair[idxValid].flatten(), :]
    #        iPtsMov_t = np.squeeze(iPtsMov[idxPair[idxValid],:])
    return iPtsRef_t, iPtsMov_t


def alignICP(iPtsRef, iPtsMov, iEps=1e-6, iMaxIter=50):
    '''
    Postopek iterativno najblizje tocke

    Parameters
    ----------
    iPtsRef : numpy.ndarray
        Polje Mx2 koordinat x in y (lahko tudi v homogeni obliki)

    iPtsMov : numpy.ndarray
        Polje Nx2 koordinat x in y (lahko tudi v homogeni obliki)

    iEps : float
        Najvecja absolutna razlika do homogene matrike preslikave identitete, ki zaustavi postopek

    iMaxIter : int
        Maksimalno stevilo iteracij

    Returns
    --------
    oMat2D : numpy.ndarray
        Homogena 3x3 transformacijska matrika med setoma vhodnih tock

    oErr : list
        Srednja Evklidska razdalja med pripadajocimi pari tock preko iteracij
    '''
    # inicializiraj izhodne parametre
    curMat = [];
    oErr = [];
    iCurIter = 0
    # zacni iterativni postopek
    while True:
        # poisci korespondencne pare tock
        iPtsRef_t, iPtsMov_t = findCorrespondingPoints(iPtsRef, iPtsMov)
        # doloci afino aproksimacijsko preslikavo
        oMat2D = mapAffineApprox2D(iPtsRef_t, iPtsMov_t)
        # posodobi premicne tocke
        iPtsMov = np.dot(addHomogCoord2D(iPtsMov), oMat2D.transpose())
        # izracunaj napako
        curMat.append(oMat2D)
        oErr.append(np.sqrt(np.sum((iPtsRef_t[:, :2] - iPtsMov_t[:, :2]) ** 2)))
        iCurIter = iCurIter + 1
        # preveri kontrolne parametre
        dMat = np.abs(oMat2D - transAffine2D())
        if iCurIter > iMaxIter or np.all(dMat < iEps):
            break
    # doloci kompozitum preslikav
    oMat2D = transAffine2D()
    for i in range(len(curMat)):
        oMat2D = np.dot(curMat[i], oMat2D)
    return oMat2D, oErr


def renderSurface(iCoorX, iCoorY, iCoorZ, iTitle='', nSamples=2000):
    """
    Izris 3D objekta s trikotnisko mrezo
    
    Parameters
    ----------
    iCoorX : numpy.ndarray
        Polje koordinat x

    iCoorY : numpy.ndarray
        Polje koordinat y

    iCoorZ : numpy.ndarray
        Polje koordinat z

    iTitle : str
        Naslov prikaznega okna

    nSamples : int
        stevilo tock za ustvarjanje trikotniske mreze

    Returns
    --------
    None
    """
    # pretvori koordinate v 1d polje
    iCoorX = iCoorX.flatten()
    iCoorY = iCoorY.flatten()
    iCoorZ = iCoorZ.flatten()
    # zdruzi tocke v vektor
    pts = np.concatenate((iCoorX, iCoorY, iCoorZ), axis=1)
    pts = pts.reshape((3, iCoorX.size))
    # zmanjsaj stevilo tock [nakljucno izberi cca. 2000 tock]
    idx = np.unique(np.floor(pts.shape[1] * sp.rand(nSamples)).astype('uint32'))
    pts = pts[:, idx]

    # PRETVORI TOCKE V SFERICNI KOORDINATNI SISTEM
    # izracunaj sredisce in centriraj
    ptsM = np.mean(pts, axis=1)
    ptsC = pts - np.tile(ptsM, (pts.shape[1], 1)).transpose()
    # doloci radij od sredisca in sfericna kota [theta, phi]
    r = np.sqrt(ptsC[0, :] ** 2 + ptsC[1, :] ** 2 + ptsC[2, :] ** 2)
    sphpts = np.zeros((2, ptsC.shape[1]))
    sphpts[0, :] = np.arccos(ptsC[2, :] / r)  # theta
    sphpts[1, :] = np.arctan2(ptsC[1, :], ptsC[0, :])  # phi

    # POVEZI TOCKE V MREZO Z DELAUNAY TRIANGULACIJO
    dl = Delaunay(sphpts.transpose())

    # IZRISI MREZO S KVAZI BARVO KOZE
    ax = a3.Axes3D(plt.figure())
    minAx = np.min(ptsC)
    maxAx = np.max(ptsC)
    ax.set_aspect('equal')
    ax.set_zlim(bottom=minAx, top=maxAx)
    ax.set_ylim(bottom=minAx, top=maxAx)
    ax.set_xlim(left=minAx, right=maxAx)
    skincol = colors.rgb2hex(np.array((1, .75, .65)))
    for i in range(0, dl.simplices.shape[0]):
        vtx = ptsC[:, dl.simplices[i, :]].transpose()
        tri = a3.art3d.Poly3DCollection([vtx])
        tri.set_color(skincol)
        # tri.set_edgecolor(skincol)
        tri.set_edgecolor('k')
        tri.set_linewidth(.1)
        ax.add_collection3d(tri)
    plt.suptitle(iTitle)
    plt.show()

def nonRigidBSplineRegistration(fixed, moving):
    # inicializacija postopka
    R = itk.ImageRegistrationMethod()

    # inicializacija preslikave z B-zlepki
    bsplineGrid = 8
    bTr = itk.BSplineTransformInitializer(fixed, [bsplineGrid] * 2)
    R.SetInitialTransform(bTr, inPlace=True)

    # inicializacija mere podobnosti
    R.SetMetricAsMattesMutualInformation(50)
    R.SetMetricSamplingPercentage(0.10)
    R.SetMetricSamplingStrategy(R.RANDOM)

    # inicializacija optimizacije
    R.SetOptimizerAsGradientDescentLineSearch(learningRate=5.0,
                                              numberOfIterations=100,
                                              convergenceMinimumValue=1e-5,
                                              convergenceWindowSize=5)
    R.SetOptimizerScalesFromPhysicalShift()

    # zagon poravnave
    outTx = R.Execute(fixed, moving)

    # ustvarjanje izhodne slike
    S = itk.ResampleImageFilter()
    S.SetReferenceImage(fixed)
    S.SetInterpolator(itk.sitkLinear)
    S.SetDefaultPixelValue(0)
    S.SetTransform(outTx)
    outImage = S.Execute(moving)
    return outImage

CSF, GM, WM, LESIONS = 1, 2, 3, 10

def kMeansInit(y, k, type='k++'):
    """Dolocanje zacetnih sredisc"""
    y = np.asarray(y, dtype='float64')
    ylength = np.max(y.shape)

    if type == 'random':
        mi = [y[:, np.random.random_integers(ylength)],
              y[:, np.random.random_integers(ylength)],
              y[:, np.random.random_integers(ylength)]]

    elif type == 'k++':
        yj = y[:, np.random.random_integers(ylength)]  # nakljucno izberemo eno izmed znacilnic
        mi = [yj]

        for i in range(k - 1):
            d = []
            for i in range(len(mi)):
                di = np.linalg.norm(np.transpose(y) - mi[i], axis=1)  # evklidske razdalje znacilnic do sredisc
                d.append(di)

            dmin = np.min(d, axis=0)  # evklidske razdalje znacilnic do najblizjih sredisc

            mii = np.sum(dmin ** 2 * y, axis=1) / np.sum(dmin ** 2)  # dolocimo novo sredisce
            mi.append(mii)

    return mi

def findClosest(y, mi):
    """Analiza gruc s postopkom k-povprecij"""
    d = []
    for i in range(len(mi)):
        di = np.linalg.norm(np.transpose(y) - mi[i], axis=1)  # evklidske razdalje do sredisc
        d.append(di)

    z = np.argmin(d, axis=0)  # najblizja sredisca
    dmin = np.min(d, axis=0)  # razdalja do najblizjega sredisca

    return z, dmin

def kMeans(y, k, maxIter=100):
    # dolocanje zacetnih sredisc
    mi = kMeansInit(y, k)

    eps = None
    Iter = 0

    while (eps != 0 and Iter < maxIter):
        # razvrscanje znacilnic
        z, _ = findClosest(y, mi)

        # posodabljanje sredisc
        miPP = []
        for i in range(len(mi)):
            miPP.append(np.mean(y[:, z == i], axis=1))  # povprecja znacilnic v mnozici

        eps = np.sum(np.abs(np.asarray(mi) - np.asarray(miPP)))  # razlika med polozaji novih in starih sredisc
        mi = miPP  # posodobimo sredisca
        Iter += 1

    return mi

def nonparClassification(y, mi):
    """Neparametricno razvrscanje znacilnic"""
    z, dmin = findClosest(y, mi)
    return z, dmin

def mrBrainSegmentation(t1, t2, pd, mask, threshold=None):
    """Razgradnja MR slik glave (iz slik T1, T2 in PD)"""
    y = []

    if pd is not None:
        y.append(pd[mask>0])
    if t2 is not None:
        y.append(t2[mask>0])
    if t1 is not None:
        y.append(t1[mask>0])
    else:
        print('razvrscanje v razrede CSF, GM, WM ne bo delovalo!')

    y = np.asarray(y, dtype='float64')

    k = 3
    mi = kMeans(y[:, ::10], k)  # analiza gruc
    z, dmin = nonparClassification(y, mi)  # neparametricno razvrscanje znacilnic

    yk, ym = [], []
    # izracunamo povprecja sivinskih vrednosti v posamezni gruci
    for i in range(k):
        yki = y[-1][z == i]
        yk.append(yki)
        ym.append(np.mean(yki))  # povprecja

    # glede na povprecja, dolocimo posamezne strukture
    icsf, igm, iwm = np.argsort(ym)

    # posameznim grucam dolocimo oznake
    L = np.zeros_like(z)
    L[z == icsf] = CSF
    L[z == igm] = GM
    L[z == iwm] = WM

    # poiscemo lezije
    if threshold is not None:
        L[dmin > threshold] = 10

    # zgradimo izhodno sliko oznak
    S = np.zeros_like(mask).astype('float')
    S[mask > 0] = L

    return S
