import cv2
import numpy as np
import datetime
import math

def ArrayToTuple(arr):
    return tuple(arr.reshape(1, -1)[0])

colors={'blue':(255, 0, 0), 'green':(0, 255, 0), 'red':(0, 0, 255), 'yellow':(0, 255, 255),
        'magenta':(255, 0, 255), 'cyan':(255, 255, 0), 'white':(255, 255, 255), 'black':(0, 0, 0),
        'gray':(125, 125, 125), 'rand':np.random.randint(0, high=256, size=(3,)).tolist(),
        'dark_gray':(50, 50, 50), 'light_gray':(220, 220, 220)}

image=np.zeros((640, 640, 3), dtype="uint8")
image[:]=colors['light_gray']

hoursOrig=np.array(
    [(620, 320), (580, 470), (470, 580), (320, 620), (170, 580), (60, 470), (20, 320), (60, 170), (169, 61), (319, 20),
     (469, 60), (579, 169)])
hoursDest=np.array(
    [(600, 320), (563, 460), (460, 562), (320, 600), (180, 563), (78, 460), (40, 320), (77, 180), (179, 78), (319, 40),
     (459, 77), (562, 179)])

for i in range(0, 12):
    cv2.line(image, ArrayToTuple(hoursOrig[i]), ArrayToTuple(hoursDest[i]), colors['black'], 3)

cv2.circle(image, (320, 320), 310, colors['dark_gray'], 8)
cv2.rectangle(image, (150, 175), (490, 270), colors['dark_gray'], -1)
cv2.putText(image, "Mastering OpenCV 4", (150, 200), 1, 2, colors['light_gray'], 1, cv2.LINE_AA)
cv2.putText(image, "with Python", (210, 250), 1, 2, colors['light_gray'], 1, cv2.LINE_AA)
imageOriginal=image.copy()

while True:
    dateTimeNow=datetime.datetime.now()
    timeNow=dateTimeNow.time()
    hour=math.fmod(timeNow.hour, 12)
    minute=timeNow.minute
    second=timeNow.second
    print("hour:'{}' minute:'{}' second: '{}'".format(hour, minute, second))

    secondAngle=math.fmod(second*6+270, 360)
    minuteAngle=math.fmod(minute*6+270, 360)
    hourAngle=math.fmod((hour*30)+(minute/2)+270, 360)
    print("hour_angle:'{}' minute_angle:'{}' second_angle: '{}'".format(hourAngle, minuteAngle, secondAngle))

    secondX=round(320+310*math.cos(secondAngle*3.14/180))
    secondY=round(320+310*math.sin(secondAngle*3.14/180))
    cv2.line(image, (320, 320), (secondX, secondY), colors['blue'], 2)

    minuteX=round(320+260*math.cos(minuteAngle*3.14/180))
    minuteY=round(320+260*math.sin(minuteAngle*3.14/180))
    cv2.line(image, (320, 320), (minuteX, minuteY), colors['blue'], 8)

    hourX=round(320+220*math.cos(hourAngle*3.14/180))
    hourY=round(320+220*math.sin(hourAngle*3.14/180))
    cv2.line(image, (320, 320), (hourX, hourY), colors['blue'], 10)

    cv2.circle(image, (320, 320), 10, colors['dark_gray'], -1)
    cv2.imshow("clock", image)
    image=imageOriginal.copy()
    if cv2.waitKey(500) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()