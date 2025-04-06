import cv2
import numpy as np

colors={'blue':(255, 0, 0), 'green':(0, 255, 0), 'red':(0, 0, 255), 'yellow':(0, 255, 255),
        'magenta':(255, 0, 255), 'cyan':(255, 255, 0), 'white':(255, 255, 255), 'black':(0, 0, 0),
        'gray':(125, 125, 125), 'rand':np.random.randint(0, high=256, size=(3,)).tolist(),
        'dark_gray':(50, 50, 50), 'light_gray':(220, 220, 220)}

def DrawCircle(event, x, y, flags, param):
    if event==cv2.EVENT_LBUTTONDBLCLK:
        print("event: EVENT_LBUTTONDBLCLK")
        cv2.circle(image, (x, y), 10, colors['magenta'], -1)
    if event==cv2.EVENT_MOUSEMOVE:
        print("event: EVENT_MOUSEMOVE")
    if event==cv2.EVENT_LBUTTONUP:
        print("event: EVENT_LBUTTONUP")
    if event==cv2.EVENT_LBUTTONDOWN:
        print("event: EVENT_LBUTTONDOWN")

image=np.zeros((600, 600, 3), dtype="uint8")
cv2.namedWindow('Image mouse')
cv2.setMouseCallback('Image mouse', DrawCircle)

while True:
    cv2.imshow('Image mouse', image)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()