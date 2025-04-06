import cv2
import numpy as np
import matplotlib.pyplot as plt

colors={'blue':(255, 0, 0), 'green':(0, 255, 0), 'red':(0, 0, 255), 'yellow':(0, 255, 255),
        'magenta':(255, 0, 255), 'cyan':(255, 255, 0), 'white':(255, 255, 255), 'black':(0, 0, 0),
        'gray':(125, 125, 125), 'rand':np.random.randint(0, high=256, size=(3,)).tolist(),
        'dark_gray':(50, 50, 50), 'light_gray':(220, 220, 220)}

image=np.zeros((400, 400, 3), dtype="uint8")
image[:]=colors['light_gray']

def UpdateImgWithMatplotlib():
    imgRGB=image[:, :, ::-1]
    plt.imshow(imgRGB)
    figure.canvas.draw()

def ClickMouseEvent(event):
    if event.dblclick and event.button == 1:
        cv2.rectangle(image, (int(round(event.xdata)), int(round(event.ydata))),
                      (int(round(event.xdata))+100, int(round(event.ydata))+50),
                      colors['blue'], cv2.FILLED)
    UpdateImgWithMatplotlib()

figure=plt.figure()
figure.add_subplot(111)
UpdateImgWithMatplotlib()
figure.canvas.mpl_connect('button_press_event', ClickMouseEvent)
plt.show()