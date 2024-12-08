import cv2
import numpy as np
import matplotlib.pyplot as plt

def BuildLutImage(cmap, height):
    lut = BuildLut(cmap)
    image = np.repeat(lut[np.newaxis, ...], height, axis=0)
    return image

def BuildLut(cmap):
    lut = np.empty(shape=(256, 3), dtype=np.uint8)
    max = 256
    lastval, lastcol = cmap[0]
    for step, col in cmap[1:]:
        val = int(step * max)
        for i in range(3):
            lut[lastval:val, i] = np.linspace(lastcol[i], col[i], val - lastval)
        lastcol = col
        lastval = val
    return lut

def ApplyColorMap1(gray, cmap):
    lut = BuildLut(cmap)
    s0, s1 = gray.shape
    out = np.empty(shape=(s0, s1, 3), dtype=np.uint8)
    for i in range(3):
        out[..., i] = cv2.LUT(gray, lut[:, i])
    return out

def ApplyColorMap2(gray, cmap):
    lut = BuildLut(cmap)
    lut2 = np.reshape(lut, (256, 1, 3))
    imColor = cv2.applyColorMap(gray, lut2)
    return imColor

def ShowWithMatplotlib(color_img, title, pos):
    imgRGB = color_img[:, :, ::-1]
    ax = plt.subplot(2, 2, pos)
    plt.imshow(imgRGB)
    plt.title(title)
    plt.axis('off')

grayImg = cv2.imread('images/lenna.png', cv2.IMREAD_GRAYSCALE)
plt.figure(figsize=(14, 6))
plt.suptitle("Custom color maps based on key colors and legend", fontsize=14, fontweight='bold')

custom1 = ApplyColorMap1(grayImg, ((0, (255, 0, 255)), (0.25, (255, 0, 180)), (0.5, (255, 0, 120)),
                         (0.75, (255, 0, 60)), (1.0, (255, 0, 0))))
custom2 = ApplyColorMap2(grayImg, ((0, (0, 255, 128)), (0.25, (128, 184, 64)), (0.5, (255, 128, 0)),
                         (0.75, (64, 128, 224)), (1.0, (0, 128, 255))))
legend1 = BuildLutImage(((0, (255, 0, 255)), (0.25, (255, 0, 180)), (0.5, (255, 0, 120)),
                        (0.75, (255, 0, 60)), (1.0, (255, 0, 0))), 20)
legend2 = BuildLutImage(((0, (0, 255, 128)), (0.25, (128, 184, 64)), (0.5, (255, 128, 0)),
                        (0.75, (64, 128, 224)), (1.0, (0, 128, 255))), 20)

ShowWithMatplotlib(legend1, "", 1)
ShowWithMatplotlib(custom1, "", 3)
ShowWithMatplotlib(legend2, "", 2)
ShowWithMatplotlib(custom2, "", 4)

plt.show()