import cv2
import numpy as np
import matplotlib.pyplot as plt

dictColor = {0: "blue", 1: "green", 2: "red"}

def BuildLut(cmap):
    lut = np.empty(shape=(256, 3), dtype=np.uint8)
    print("----------")
    print(cmap)
    print("-----")
    max = 256
    lastval, lastcol = cmap[0]
    for step, col in cmap[1:]:
        val = int(step * max)
        for i in range(3):
            print("{} : np.linspace('{}', '{}', '{}' - '{}' = '{}')".format(dictColor[i], lastcol[i], col[i], val, lastval, val - lastval))
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
    lut_reshape = np.reshape(lut, (256, 1, 3))
    imColor = cv2.applyColorMap(gray, lut_reshape)
    return imColor

def ShowWithMatplotlib(colorImg, title, pos):
    imgRGB = colorImg[:, :, ::-1]
    ax = plt.subplot(2, 3, pos)
    plt.imshow(imgRGB)
    plt.title(title)
    plt.axis('off')

grayImg = cv2.imread('images/shades.png', cv2.IMREAD_GRAYSCALE)
plt.figure(figsize=(14, 3))
plt.suptitle("Custom color maps based on key colors", fontsize=14, fontweight='bold')
ShowWithMatplotlib(cv2.cvtColor(grayImg, cv2.COLOR_GRAY2BGR), "gray", 1)

custom1 = ApplyColorMap1(grayImg, ((0, (255, 0, 255)), (0.25, (255, 0, 180)), (0.5, (255, 0, 120)),
                          (0.75, (255, 0, 60)), (1.0, (255, 0, 0))))
custom2 = ApplyColorMap1(grayImg, ((0, (0, 255, 128)), (0.25, (128, 184, 64)), (0.5, (255, 128, 0)),
                          (0.75, (64, 128, 224)), (1.0, (0, 128, 255))))
custom3 = ApplyColorMap2(grayImg, ((0, (255, 0, 255)), (0.25, (255, 0, 180)), (0.5, (255, 0, 120)),
                          (0.75, (255, 0, 60)), (1.0, (255, 0, 0))))
custom4 = ApplyColorMap2(grayImg, ((0, (0, 255, 128)), (0.25, (128, 184, 64)), (0.5, (255, 128, 0)),
                          (0.75, (64, 128, 224)), (1.0, (0, 128, 255))))

ShowWithMatplotlib(custom1, "custom 1 using cv2.LUT()", 2)
ShowWithMatplotlib(custom2, "custom 2 using cv2.LUT()", 3)
ShowWithMatplotlib(custom3, "custom 3 using cv2.applyColorMap()", 5)
ShowWithMatplotlib(custom4, "custom 4 using using cv2.applyColorMap()", 6)

plt.show()