import cv2
import numpy as np
import matplotlib.pyplot as plt

def ShowWithMatplotlib(img, title):
    imgRGB=img[:, :, ::-1]
    plt.imshow(imgRGB)
    plt.title(title)
    plt.show()

colors={'blue':(255, 0, 0), 'green':(0, 255, 0), 'red':(0, 0, 255), 'yellow':(0, 255, 255),
        'magenta':(255, 0, 255), 'cyan':(255, 255, 0), 'white':(255, 255, 255), 'black':(0, 0, 0),
        'gray':(125, 125, 125), 'rand':np.random.randint(0, high=256, size=(3,)).tolist(),
        'dark_gray':(50, 50, 50), 'light_gray':(220, 220, 220)}
fonts={0:"FONT HERSHEY SIMPLEX", 1:"FONT HERSHEY PLAIN", 2:"FONT HERSHEY DUPLEX", 3:"FONT HERSHEY COMPLEX",
       4:"FONT HERSHEY TRIPLEX", 5:"FONT HERSHEY COMPLEX SMALL ", 6:"FONT HERSHEY SCRIPT SIMPLEX",
       7:"FONT HERSHEY SCRIPT COMPLEX"}

indexColors={0:'blue', 1:'green', 2:'red', 3:'yellow', 4:'magenta', 5:'cyan', 6:'black', 7:'dark_gray'}
image=np.zeros((650, 650, 3), dtype="uint8")
image[:]=colors['white']
position=(10, 30)

for i in range(0, 8):
    print("i index value: '{}' text: '{}' + color: '{}' = '{}'".format(i, fonts[i].lower(), indexColors[i], colors[indexColors[i]]))
    cv2.putText(image, fonts[i], position, i, 1.1, colors[indexColors[i]], 2, cv2.LINE_4)
    position=(position[0], position[1]+40)
    cv2.putText(image, fonts[i].lower(), position, i, 1.1, colors[indexColors[i]], 2, cv2.LINE_4)
    position=(position[0], position[1]+40)

ShowWithMatplotlib(image, 'cv2.putText() using all OpenCV fonts')