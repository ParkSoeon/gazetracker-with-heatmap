import cv2
import pyautogui
from matplotlib import pyplot as plt
import seaborn as sns
import pyscreenshot
import PIL
from PIL import Image
import numpy as np

ESCAPE_KEY = 27
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)

def transform_video_coordinates_to_screen(eye_x_pos, eye_y_pos):
    if not video_resolution:
        return (eye_x_pos, eye_y_pos)

    return (
        eye_x_pos / video_resolution[0] * screen_resolution[0],
        eye_y_pos / video_resolution[1] * screen_resolution[1],
    )

def update_mouse_position(hough_circles, eye_x_pos, eye_y_pos, roi_color2):
    try:
        for circle in hough_circles[0, :]:
            circle_center = (circle[0], circle[1])
            cv2.circle(
                img=roi_color2,
                center=circle_center,
                radius=circle[2],
                color=WHITE,
                thickness=2
            )
            cv2.circle(
                img=roi_color2,
                center=circle_center,
                radius=2,
                color=WHITE,
                thickness=3
            )

            x_pos, y_pos = transform_video_coordinates_to_screen(eye_x_pos, eye_y_pos)
            pyautogui.moveTo(x_pos, y_pos)
    except Exception as e:
        print('Exception:', e)

face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_righteye_2splits.xml')

video_capture = cv2.VideoCapture(0)
eye_x_positions = []
eye_y_positions = []

screen_resolution = pyautogui.size()
print("Screen resolution:", screen_resolution)

if video_capture.isOpened():
    video_resolution = (
        video_capture.get(cv2.CAP_PROP_FRAME_WIDTH),
        video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT),
    )
    print("Camera FPS:", video_capture.get(cv2.CAP_PROP_FPS))
else:
    video_resolution = None

fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter("output.avi", fourcc, 30.0, (screen_resolution[0], screen_resolution[1]))

while True:
    success, image = video_capture.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray)

    for (eye_x, eye_y, eye_width, eye_height) in eyes:
        cv2.rectangle(image, (eye_x, eye_y), (eye_x + eye_width, eye_y + eye_height), GREEN, 2)
        roi_gray2 = gray[eye_y: eye_y + eye_height, eye_x: eye_x + eye_width]
        roi_color2 = image[eye_y: eye_y + eye_height, eye_x: eye_x + eye_width]

        hough_circles = cv2.HoughCircles(
            roi_gray2, cv2.HOUGH_GRADIENT, 1, 200, param1=200, param2=1, minRadius=0, maxRadius=0
        )

        eye_x_pos = (eye_x + eye_width) / 2
        eye_y_pos = (eye_y + eye_height) / 2
        print(eye_x_pos, eye_y_pos)
        eye_x_positions.append(eye_x_pos)
        eye_y_positions.append(eye_y_pos)

        update_mouse_position(hough_circles, eye_x_pos, eye_y_pos, roi_color2)

    background_screenshot = pyautogui.screenshot()
    frame = np.array(background_screenshot)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    my_dpi = 118
    plt.scatter(eye_x_positions, eye_y_positions, alpha=0.8)
    plt.axis('off')
    plt.savefig('scatter.png', dpi=my_dpi * 2)
    plt.clf()
    
    scatter = PIL.Image.open("scatter.png")
    scatter = scatter.resize(screen_resolution)
    scatter.save("new_scatter.png")

    img2 = cv2.imread('new_scatter.png')
    final_frame = cv2.addWeighted(frame, 0.7, img2, 0.3, 0)
    cv2.imshow('img', final_frame)
    out.write(final_frame)
    
    key_pressed = cv2.waitKey(30) & 0xff
    if key_pressed == ESCAPE_KEY or key_pressed == ord('q'):
        break

video_capture.release()
out.release()
cv2.destroyAllWindows()
