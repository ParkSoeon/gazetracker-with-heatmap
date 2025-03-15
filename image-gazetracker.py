import cv2
import pyautogui
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from PIL import ImageGrab, Image

ESCAPE_KEY = 27
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)


def transform_video_coordinates_to_screen(
    eye_x_pos, eye_y_pos, video_resolution, screen_resolution
):
    if not video_resolution:
        return (eye_x_pos, eye_y_pos)

    return (
        eye_x_pos / video_resolution[0] * screen_resolution[0],
        eye_y_pos / video_resolution[1] * screen_resolution[1],
    )


def update_mouse_position(
    hough_circles, eye_x_pos, eye_y_pos, roi_color2, screen_resolution, video_resolution
):
    try:
        if hough_circles is not None:
            for circle in hough_circles[0, :]:
                circle_center = (circle[0], circle[1])

                cv2.circle(
                    roi_color2,
                    center=circle_center,
                    radius=circle[2],
                    color=WHITE,
                    thickness=2,
                )
                cv2.circle(
                    roi_color2, center=circle_center, radius=2, color=WHITE, thickness=3
                )

                x_pos, y_pos = transform_video_coordinates_to_screen(
                    eye_x_pos, eye_y_pos, video_resolution, screen_resolution
                )
                pyautogui.moveTo(int(x_pos), int(y_pos))
    except Exception as e:
        print("Exception:", e)


# Load Haar cascades
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_righteye_2splits.xml"
)

# Initialize video capture
video_capture = cv2.VideoCapture(0)
eye_x_positions, eye_y_positions = [], []

screen_resolution = pyautogui.size()
print("Screen resolution:", screen_resolution)

video_resolution = (
    (
        int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    if video_capture.isOpened()
    else None
)

while True:
    success, image = video_capture.read()
    if not success:
        break

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray)

    for eye_x, eye_y, eye_width, eye_height in eyes:
        cv2.rectangle(
            image, (eye_x, eye_y), (eye_x + eye_width, eye_y + eye_height), GREEN, 2
        )

        roi_gray2 = gray[eye_y : eye_y + eye_height, eye_x : eye_x + eye_width]
        roi_color2 = image[eye_y : eye_y + eye_height, eye_x : eye_x + eye_width]

        hough_circles = cv2.HoughCircles(
            roi_gray2,
            cv2.HOUGH_GRADIENT,
            1,
            200,
            param1=200,
            param2=1,
            minRadius=0,
            maxRadius=0,
        )

        eye_x_pos = eye_x + eye_width // 2
        eye_y_pos = eye_y + eye_height // 2
        print(eye_x_pos, eye_y_pos)

        eye_x_positions.append(eye_x_pos)
        eye_y_positions.append(eye_y_pos)

        update_mouse_position(
            hough_circles,
            eye_x_pos,
            eye_y_pos,
            roi_color2,
            screen_resolution,
            video_resolution,
        )

    cv2.imshow("Eye Tracking", image)

    key_pressed = cv2.waitKey(30) & 0xFF
    if key_pressed == ESCAPE_KEY:
        break

video_capture.release()
cv2.destroyAllWindows()

data = list(zip(eye_x_positions, eye_y_positions))
print("Eye tracking data:", data)

# Capture screenshot using Pillow (instead of pyscreenshot)
screenshot = ImageGrab.grab()
screenshot.save("screenshot.png")

background_image = Image.open("screenshot.png")
width, height = background_image.size
print("Screenshot size:", width, height)

# Set DPI for the saved image
dpi = 118

# Scatter plot
plt.scatter(eye_x_positions, eye_y_positions, c="red", alpha=0.5)
plt.tight_layout()
plt.axis("off")
plt.savefig("scatter.png", dpi=dpi * 2, bbox_inches="tight")

scatter = Image.open("scatter.png").resize((width, height))
scatter.save("new_scatter.png")

# Overlay scatter plot on screenshot
img1 = cv2.imread("screenshot.png")
img2 = cv2.imread("new_scatter.png")
dst = cv2.addWeighted(img1, 0.7, img2, 0.3, 0)
cv2.imshow("Scatter Plot", dst)
cv2.imwrite("Output_Scatter.png", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Heatmap plot
plt.clf()
sns.kdeplot(
    x=eye_x_positions, y=eye_y_positions, fill=True, cmap="rocket_r", thresh=0.01
)
plt.axis("off")
plt.savefig("heatmap.png", dpi=dpi * 2, bbox_inches="tight")

heatmap = Image.open("heatmap.png").resize((width, height))
heatmap.save("new_heatmap.png")

# Overlay heatmap on screenshot
img1 = cv2.imread("screenshot.png")
img2 = cv2.imread("new_heatmap.png")
dst = cv2.addWeighted(img1, 0.6, img2, 0.4, 0)
cv2.imshow("Heatmap Plot", dst)
cv2.imwrite("Output_heatmap.png", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
