import cv2
import numpy as np
import pyautogui
from collections import defaultdict
from handsDetect import HandDetector

trajectory = defaultdict(list)
max_frames = 50
fade_time = 15

mouse_speed = 5
previous_position = None

camera = cv2.VideoCapture(0)
hand_detector = HandDetector()

threshold_distance = 15

while True:
    success, img = camera.read()
    if success:
        img = cv2.flip(img, 1)
        hand_detector.process(img, draw=False)
        position = hand_detector.find_position(img)
        left_finger = position['Right'].get(8, None)
        thumb = position['Right'].get(4, None)

        if previous_position is None:
            previous_position = left_finger

        if left_finger is not None:
            finger_movement_x = left_finger[0] - previous_position[0]
            finger_movement_y = left_finger[1] - previous_position[1]
            if abs(finger_movement_x) > 10 or abs(finger_movement_y) > 0:
                mouse_delta_x = int(finger_movement_x * mouse_speed)
                mouse_delta_y = int(finger_movement_y * mouse_speed)
                pyautogui.moveRel(mouse_delta_x, mouse_delta_y)
            previous_position = left_finger

        if left_finger is not None and thumb is not None:
            distance = np.sqrt((left_finger[0] - thumb[0]) ** 2 + (left_finger[1] - thumb[1]) ** 2)
            if distance < threshold_distance:
                pyautogui.click()
              
        if left_finger is not None:
            cv2.circle(img, (left_finger[0], left_finger[1]), 10, (0, 0, 255), cv2.FILLED)
            trajectory['Left_Finger'].append(left_finger)
            if len(trajectory['Left_Finger']) > max_frames:
                trajectory['Left_Finger'].pop(0)
        
        if thumb is not None:
            cv2.circle(img, (thumb[0], thumb[1]), 10, (255, 0, 0), cv2.FILLED)  # 标记大拇指位置为红色
            trajectory['Thumb'].append(thumb)
            if len(trajectory['Thumb']) > max_frames:
                trajectory['Thumb'].pop(0)

        cv2.imshow("camera", img)

    k = cv2.waitKey(1)
    if k == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
