import cv2
from run_pose_detector import *


def show_webcam(overlay_pose=False):
    interpreter = load_model()
    cam = cv2.VideoCapture(0)
    while True:
        ret_val, img = cam.read()
        if overlay_pose:
            img = annotate_img(img, interpreter)
        cv2.imshow('my webcam', img)
        if cv2.waitKey(1) == 27:
            break  # esc to quit
    cv2.destroyAllWindows()


def main():
    show_webcam(overlay_pose=True)


if __name__ == '__main__':
    main()