import cv2
import numpy as np
from getKeypointsFile import getKeypoints
from functionLightG import matchFrame
from lightglue import viz2d
import matplotlib.pyplot as plt
from ultralytics import YOLO

def createMaskGroup(keypoints):
    x_values = keypoints[:, :, 0].flatten()
    y_values = keypoints[:, :, 1].flatten()

    # Exclude zero values
    x_values = x_values[x_values != 0]
    y_values = y_values[y_values != 0]
    
    min_x = np.min(x_values)
    max_x = np.max(x_values)
    min_y = np.min(y_values)
    max_y = np.max(y_values)

    return min_x, max_x, min_y, max_y

model_pose = YOLO('yolov8x-pose-p6.pt')
model_seg = YOLO('yolov8x-seg.pt')

nameBase = "16_35_42_26_02_2024_VID"

# VIDEOS ---------------------------
capL=cv2.VideoCapture('./database/waiter/integradora/' + nameBase + '_LEFT_calibrated.avi')
# capL=cv2.VideoCapture('./database/waiter/YOLO/' + nameBase + '_LEFT.avi')
capR=cv2.VideoCapture('./database/waiter/integradora/' + nameBase + '_RIGHT_calibrated.avi')
# capR=cv2.VideoCapture('./database/waiter/YOLO/' + nameBase + '_RIGHT.avi')

frame_num = 0
step_frames = 256
conf = 0.65

# 260 marca 0
# 253 en izquierdo no es el mismo del derecho
# 255 marca 0



while(capR.isOpened() and capL.isOpened()):
    frame_num += 1

    if step_frames > frame_num:
        capL.set(cv2.CAP_PROP_POS_FRAMES, step_frames)
        capR.set(cv2.CAP_PROP_POS_FRAMES, step_frames)
        frame_num = step_frames
    
    ret,frameL = capL.read()
    retR,frameR = capR.read()

    h = frameR.shape[0]
    w = frameR.shape[1]

    if(not ret or not retR):
        print("Failed to read frames")
        break
    
    """
    resultL = model_pose.predict(frameL, conf=conf)
    resultR = model_pose.predict(frameR, conf=conf)
    """

    # resultL_seg = model_seg.predict(frameL, conf=conf, classes=[0])
    resultR_seg = model_seg.predict(frameR, conf=conf, classes=[0])

    """
    keypointsL = np.array(resultL[0].keypoints.xy.cpu())#[:, [2, 5, 9, 12], :]
    keypointsR_filtered = np.array(resultR[0].keypoints.xy.cpu())#[:, [2, 5, 9, 12], :]
    """

    # rectangle = np.zeros((h, w, 3), dtype='uint8')
    rectangle = np.zeros((h, w), dtype='uint8')

    """
    # todos
    if len(keypointsL) == 0 or len(keypointsR_filtered) == 0:
        print("No keypoints found", "Frame", frame_num)
        continue
    """

    """
    keypoints = np.array(resultR[0].boxes.data.cpu())
    for person in keypoints:
        box = person.astype(int)
        rectangle[box[1]: box[3], box[0]: box[2]] = 255
    """
    
    for person in resultR_seg[0].masks.xy:
        person = np.int32([person])
        cv2.fillPoly(rectangle, person, [255, 255, 255])
    

    # cv2.imshow('Rectangle',rectangle)
    cv2.imshow('LEFT',frameL)
    # cv2.imshow('RIGHT 2', frameR * rectangle)
    newFrameR = cv2.bitwise_and(frameR, frameR, mask=rectangle)
    cv2.imshow('RIGHT 2', newFrameR)
    imgResult, pts0, pts1, points0, points1, matches01 = matchFrame(frameL, newFrameR)
    # cv2.imshow('Correspondencia', imgResult)
    
    
    archivo_f_l = open("results/" + nameBase + "_LEFT/" + nameBase + "_LEFT_" + str(frame_num) +  ".txt", "w")
    archivo_f_r = open("results/" + nameBase + "_RIGHT/" + nameBase + "_RIGHT_" + str(frame_num) +  ".txt", "w")
    archivo_f_l.write(str(pts0))
    archivo_f_r.write(str(pts1))
    archivo_f_l.close()
    archivo_f_r.close() 
    
    axes = viz2d.plot_images([cv2.cvtColor(frameL, cv2.COLOR_BGR2RGB), cv2.cvtColor(newFrameR, cv2.COLOR_BGR2RGB)])
    viz2d.plot_matches(points0, points1, color="lime", lw=0.2)
    viz2d.add_text(0, f'Stop after {matches01["stop"]} layers', fs=20)
    plt.show()

    key = cv2.waitKey(0)
    if key == ord('q'):
        # Close
        break

    print("Frame", frame_num)

capL.release()
capR.release()
cv2.destroyAllWindows()
