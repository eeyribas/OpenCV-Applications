import cv2
import numpy as np

colors={'blue':(255, 0, 0), 'green':(0, 255, 0), 'red':(0, 0, 255), 'yellow':(0, 255, 255),
        'magenta':(255, 0, 255), 'cyan':(255, 255, 0), 'white':(255, 255, 255), 'black':(0, 0, 0),
        'gray':(125, 125, 125), 'rand':np.random.randint(0, high=256, size=(3,)).tolist(),
        'dark_gray':(50, 50, 50), 'light_gray':(220, 220, 220)}

def DrawText():
    menuPos1=(10, 500)
    menuPos2=(10, 525)
    menuPos3=(10, 550)
    menuPos4=(10, 575)
    cv2.putText(image, 'Double left click: add a circle', menuPos1, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255))
    cv2.putText(image, 'Simple right click: delete last circle', menuPos2, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255))
    cv2.putText(image, 'Double right click: delete all circle', menuPos3, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255))
    cv2.putText(image, 'Press \'q\' to exit', menuPos4, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255))

def DrawCircle(event, x, y, flags, param):
    global circles
    if event==cv2.EVENT_LBUTTONDBLCLK:
        print("event: EVENT_LBUTTONDBLCLK")
        circles.append((x, y))
    if event==cv2.EVENT_RBUTTONDBLCLK:
        print("event: EVENT_RBUTTONDBLCLK")
        circles[:]=[]
    elif event==cv2.EVENT_RBUTTONDOWN:
        print("event: EVENT_RBUTTONDOWN")
        try:
            circles.pop()
        except (IndexError):
            print("no circles to delete")
    if event==cv2.EVENT_MOUSEMOVE:
        print("event: EVENT_MOUSEMOVE")
    if event==cv2.EVENT_LBUTTONUP:
        print("event: EVENT_LBUTTONUP")
    if event==cv2.EVENT_LBUTTONDOWN:
        print("event: EVENT_LBUTTONDOWN")

circles=[]
image=np.zeros((600, 600, 3), dtype="uint8")
cv2.namedWindow('Image mouse')
cv2.setMouseCallback('Image mouse', DrawCircle)
DrawText()
clone=image.copy()

while True:
    image=clone.copy()
    for pos in circles:
        cv2.circle(image, pos, 30, colors['blue'], -1)
    cv2.imshow('Image mouse', image)
    if cv2.waitKey(400) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()