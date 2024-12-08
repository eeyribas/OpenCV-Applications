import cv2
import numpy as np
import matplotlib.pyplot as plt

def ShowWithMatplotlib(img, title):
    imgRGB = img[:, :, ::-1]
    plt.imshow(imgRGB)
    plt.title(title)
    plt.show()

colors = {'blue': (255, 0, 0), 'green': (0, 255, 0), 'red': (0, 0, 255), 'yellow': (0, 255, 255),
          'magenta': (255, 0, 255), 'cyan': (255, 255, 0), 'white': (255, 255, 255), 'black': (0, 0, 0),
          'gray': (125, 125, 125), 'rand': np.random.randint(0, high=256, size=(3,)).tolist(),
          'dark_gray': (50, 50, 50), 'light_gray': (220, 220, 220)}

image = np.zeros((20, 20, 3), dtype="uint8")
image[:] = colors['light_gray']
cv2.line(image, (5, 0), (20, 15), colors['yellow'], 1, cv2.LINE_4)
cv2.line(image, (0, 0), (20, 20), colors['red'], 1, cv2.LINE_AA)
cv2.line(image, (0, 5), (15, 20), colors['green'], 1, cv2.LINE_8)
ShowWithMatplotlib(image, 'LINE_4    LINE_AA    LINE_8')