import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from scipy import signal

def computeC2(sigma, theta, filterRadius):
    numerater = 0
    denominator = 0
    e = [np.cos(theta), np.sin(theta)]
    num = (np.pi/(2*sigma))
    for x in range(-filterRadius, filterRadius+1):
        for y in range(-filterRadius, filterRadius+1):
            u = [x,y]
            u2 = (x**2) + (y**2)
            denominator += np.exp(-u2/(2*(sigma**2)))
            numerater += complex(np.cos(num*np.dot(u, e)),np.sin(num*np.dot(u,e)))*np.exp(-u2/(2*(sigma**2)))
    c2 = numerater / denominator
    return c2

def computeC1(sigma, theta, c2, filterRadius):
    z=0
    e = [np.cos(theta), np.sin(theta)]
    num = np.pi / (2*sigma)
    for x in range(-filterRadius, filterRadius+1):
        for y in range(-filterRadius, filterRadius+1):
            u = [x,y]
            u2 = (x**2)+(y**2)
            z += ((1 - 2 * c2 * np.cos(num*np.dot(u,e))+(c2**2))*np.exp(-u2/(sigma**2)))
    c1 = sigma/(z**0.5)
    return c1

def psiFunc(x,y,c1,c2,sigma,theta):
    num = np.pi/(2*sigma)
    u2 = (x**2)+(y**2)
    u = [x,y]
    e = [np.cos(theta), np.sin(theta)]
    psi = (c1/sigma) * (complex(np.cos(num*np.dot(u,e)), np.sin(num*np.dot(u,e))) - c2)* np.exp(-u2/(2*(sigma**2)))
    return psi

def makeWavelet(sigma, theta, filterRadius, morletReal, morletImaginary):
    c2 = computeC2(sigma, theta, filterRadius)
    c1 = computeC1(sigma, theta, c2, filterRadius)
    for x in range(-filterRadius, filterRadius+ 1):
        for y in range(-filterRadius, filterRadius+ 1):
            morlet = psiFunc(x*1., y*1., c1, c2, sigma, theta)
            morletReal[x+filterRadius][y+filterRadius] = morlet.real
            morletImaginary[x+filterRadius][y+filterRadius] = morlet.imag
    return

def makeWaveletList(sigma, Theta, filterRadius):
    morletReal = np.zeros((filterRadius*2+1,filterRadius*2+1))
    morletImaginary = np.zeros((filterRadius*2+1,filterRadius*2+1))
    for theta in Theta:
        makeWavelet(sigma, theta, filterRadius, morletReal, morletImaginary)
        rList.append(np.matrix.copy(morletReal))
        iList.append(np.matrix.copy(morletImaginary))
    return

def rescale(matrix):
    scaled = ((matrix - matrix.min()) * (255 / (matrix.max() - matrix.min()))).astype(np.uint8)
    return scaled

def plotWavelets():
    plt.suptitle("Real Wavelets")
    for i in range (0,len(rList)):
        realScaled = rescale(rList[i]) #(255.0 / (rList[i].max() - rList[i].min()) * (rList[i] - rList[i].min())).astype(np.uint8)
        realImage = Image.fromarray(realScaled)
        plt.subplot(2,2, i+1)
        plt.imshow(realImage, cmap='gray')
    plt.show()

    plt.suptitle('Imaginary Wavelets')
    for i in range (0,len(iList)):
        imaginaryScaled = rescale(iList[i]) #(255.0 / (iList[i].max() - iList[i].min()) * (iList[i] - iList[i].min())).astype(np.uint8)
        imaginaryImage = Image.fromarray(imaginaryScaled)
        plt.subplot(2,2, i+1)
        plt.imshow(imaginaryImage, cmap='gray')
    plt.show()
    return

def convolve(leftPic, rightPic):
    for i in range (0,len(rList)):
        #left real
        leftreal = signal.convolve2d(leftPic, rList[i])
        realLeft.append(np.matrix.copy(leftreal))
        #right real
        rightreal = signal.convolve2d(rightPic, rList[i])
        realRight.append(np.matrix.copy(rightreal))
        #left imaginary
        leftimaginary = signal.convolve2d(leftPic, iList[i])
        imagLeft.append(np.matrix.copy(leftimaginary))
        #right imaginary
        rightimaginary = signal.convolve2d(rightPic, iList[i])
        imagRight.append(np.matrix.copy(rightimaginary))
    return

def computeW(imagLeft, realLeft, imagRight, realRight):
    [r,c] = imagLeft[0].shape
    for i in range (0, r):
        for j in range(0, c):
            for n in range (0, len(imagLeft)):
                if Wleft[i][j] < abs(imagLeft[n][i][j] - realLeft[n][i][j]):
                    Wleft[i][j] = abs(imagLeft[n][i][j] - realLeft[n][i][j])
                if Wright[i][j] < abs(imagRight[n][i][j] - realRight[n][i][j]):
                    Wright[i][j] = abs(imagRight[n][i][j] - realRight[n][i][j])
    return

def plotEdge(Wleft, Wright):
    plt.suptitle("Edge Map")
    rescaledLeft = rescale(Wleft)
    rescaledRight = rescale(Wright)
    plt.subplot(1,2,1)
    plt.title("Left Pentagon")
    plt.imshow(rescaledLeft, cmap = 'gray')
    plt.subplot(1, 2, 2)
    plt.title("Right Pentagon")
    plt.imshow(rescaledRight, cmap = 'gray')
    plt.show()
    return

def computeDisp(Wleft, Wright, Delta):
    e = 0.001
    for delta in Delta:
        dispM = np.zeros(Wleft.shape) + 1000.
        [r,c] = dispM.shape
        for y in range (30, r - 30):
            for x in range(30, c - 30):
                minError = 999999.
                for disparity in range (-5, 16):
                    error = 0
                    for xd in range (x-delta, x+delta+1):
                        #error += (Wleft[y][xd] + e)/(Wright[y][xd-disparity]+e) + (Wright[y][xd-disparity] + e) / (Wleft[y][xd] +e)
                        error += abs(Wleft[y][xd] - Wright[y][xd - disparity]) ** 2
                    if error < minError:
                        dispM[y][x] = disparity
                        minError = error
        #   deduct 1000.
        for y in range(0, r):
            for x in range(0, c):
                if dispM[x][y] == 1000.:
                    dispM[x][y] = 0
        disp.append(np.matrix.copy(dispM))
    return

def plotDisp():
    plt.suptitle("Pentagon Images Disparity Solutions")
    rescaled1 = rescale(disp[0])
    rescaled2 = rescale(disp[1])
    plt.subplot(1,2,1)
    plt.title("delta = 2")
    plt.imshow(rescaled1, cmap = 'gray')
    plt.subplot(1, 2, 2)
    plt.title("delta = 4")
    plt.imshow(rescaled2, cmap = 'gray')
    plt.show()
    return

def computeDispDP(Wleft, Wright, Occ):

    return

#   inputs
sigma = 2
Theta = [0, np.pi/4, np.pi/2, np.pi*3/4]
filterRadius = 6
rList = []
iList = []

#   create wavelets and plot
makeWaveletList(sigma, Theta, filterRadius)
#plotWavelets()

#   import pentagons
pentagonLeft = cv2.imread('Pentagonleft.png',0)
pentagonRight = cv2.imread('Pentagonright.png',0)

#   convolve
realLeft = []
realRight = []
imagLeft = []
imagRight = []
convolve(pentagonLeft, pentagonRight)

#   compute Wleft Wright
Wleft = np.zeros(realLeft[0].shape)
Wright = np.zeros(realRight[0].shape)
computeW(imagLeft, realLeft, imagRight, realRight)

#   plot Wleft Wright
plotEdge(Wleft, Wright)

#   compute disparity
disp = []
Delta = [2, 4]
#computeDisp(Wleft, Wright, Delta)
#plotDisp()

#   compute disparity w dynamic programming
#computeDispDP(Wleft, Wright, Occ)
