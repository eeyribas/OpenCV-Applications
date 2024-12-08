import cv2
import numpy as np
import matplotlib.pyplot as plt

def ShowWithMatplotlib(img, title):
    imgRGB = img[:, :, ::-1]
    plt.imshow(imgRGB)
    plt.title(title)
    plt.show()

def DrawFloatCircle(img, center, radius, color, thickness=1, lineType=8, shift=4):
    factor = 2 ** shift
    center = (int(round(center[0] * factor)), int(round(center[1] * factor)))
    radius = int(round(radius * factor))
    cv2.circle(img, center, radius, color, thickness, lineType, shift)

colors = {'blue': (255, 0, 0), 'green': (0, 255, 0), 'red': (0, 0, 255), 'yellow': (0, 255, 255),
          'magenta': (255, 0, 255), 'cyan': (255, 255, 0), 'white': (255, 255, 255), 'black': (0, 0, 0),
          'gray': (125, 125, 125), 'rand': np.random.randint(0, high=256, size=(3,)).tolist(),
          'dark_gray': (50, 50, 50), 'light_gray': (220, 220, 220)}

image = np.zeros((800, 800, 3), dtype="uint8")
image[:] = colors['light_gray']
DrawFloatCircle(image, (299, 299), 300, colors['red'], 1, 8, 0)
DrawFloatCircle(image, (299.9, 299.9), 300, colors['green'], 1, 8, 1)
DrawFloatCircle(image, (299.99, 299.99), 300, colors['blue'], 1, 8, 2)
DrawFloatCircle(image, (299.999, 299.999), 300, colors['yellow'], 1, 8, 3)
ShowWithMatplotlib(image, 'cv2.circle()')