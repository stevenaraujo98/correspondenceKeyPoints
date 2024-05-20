import cv2
import numpy as np
from getKeypointsFile import getKeypoints
from functionLightG import matchFrame
from lightglue import viz2d
import matplotlib.pyplot as plt

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

def createMaskPerson(keypoints):
    x_values = keypoints[:, 0].flatten()
    y_values = keypoints[:, 1].flatten()

    # Exclude zero values
    x_values = x_values[x_values != 0]
    y_values = y_values[y_values != 0]
    
    min_x = np.min(x_values)
    max_x = np.max(x_values)
    min_y = np.min(y_values)
    max_y = np.max(y_values)

    return min_x, max_x, min_y, max_y

nameBase = "16_35_42_26_02_2024_VID"

capL=cv2.VideoCapture('./database/waiter/' + nameBase + '_LEFT_calibrated.avi')
# capL=cv2.VideoCapture('./database/waiter/YOLO/' + nameBase + '_LEFT.avi')
path_file_left = './database/waiter/YOLO/' + nameBase + '_LEFT/' + nameBase + '_LEFT_'
# path_file_left = './database/waiter/YOLO/' + nameBase + '_LEFT/frame_' OP
capR=cv2.VideoCapture('./database/waiter/' + nameBase + '_RIGHT_calibrated.avi')
# capR=cv2.VideoCapture('./database/waiter/YOLO/' + nameBase + '_RIGHT.avi')
path_file_right = './database/waiter/YOLO/' + nameBase + '_RIGHT/' + nameBase + '_RIGHT_'
# path_file_right = './database/waiter/YOLO/' + nameBase + '_RIGHT/frame_' OP

frame_num = 0
step_frames = 256

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
    
    

    keypointsL = np.array(getKeypoints(path_file_left + str(frame_num) + '.txt'))
    keypointsR_sorted = np.array(getKeypoints(path_file_right + str(frame_num) + '.txt'))

    # keypointsL_filtered = keypointsL[:, [5, 6, 11, 12], :]
    # keypointsR_filtered = keypointsR_sorted[:, [5, 6, 11, 12], :]
    keypointsL_filtered = keypointsL#[:, [2, 5, 9, 12], :]
    keypointsR_filtered = keypointsR_sorted#[:, [2, 5, 9, 12], :]

    # rectangle = np.zeros((h, w, 3), dtype='uint8')
    rectangle = np.zeros((h, w), dtype='uint8')

    # filtro de ceros x y y
    # keypointsL_filtered = keypointsL_filtered[np.logical_and(keypointsL_filtered[:, :, 0] != 0, keypointsL_filtered[:, :, 1] != 0)]
    # keypointsR_filtered = keypointsR_filtered[np.logical_and(keypointsR_filtered[:, :, 0] != 0, keypointsR_filtered[:, :, 1] != 0)]
    # Por persona
    # x_values = keypointsR_filtered[:, :, 0][0].flatten()
    # y_values = keypointsR_filtered[:, :, 1][0].flatten()
    # todos
    if len(keypointsL) == 0 or len(keypointsR_sorted) == 0:
        print("No keypoints found", "Frame", frame_num)
        continue
    
    """
    # En grupo
    min_x, max_x, min_y, max_y = createMaskGroup(keypointsR_filtered)
    

    # rectangle[int(min_y)-50: int(max_y)+50, int(min_x)-50: int(max_x)+50] = 1
    rectangle[int(min_y)-50: int(max_y)+50, int(min_x)-50: int(max_x)+50] = 255
    cv2.imshow('RECTANGULO', rectangle)


    cv2.imshow('LEFT',frameL)
    # cv2.imshow('RIGHT 2', frameR * rectangle)
    newFrameR = cv2.bitwise_and(frameR, frameR, mask=rectangle)
    cv2.imshow('RIGHT 2', newFrameR)
    imgResult, pts0, pts1 = matchFrame(frameL, newFrameR)
    cv2.imshow('Correspondencia', imgResult)

    indices = [i for i, kp in enumerate(keypointsL[0][:,1]) if kp in pts0]
    print("Indices:", indices, keypointsL[0][:,1][-1])
    """

    # Individual
    countPerson = 0
    for person in keypointsR_filtered:
        min_x, max_x, min_y, max_y = createMaskPerson(person)    
        rectangle[int(min_y)- 30: int(max_y)+40, int(min_x)-50: int(max_x)+50] = 255

    cv2.imshow('Rectangle',rectangle)
    cv2.imshow('LEFT',frameL)
    # cv2.imshow('RIGHT 2', frameR * rectangle)
    newFrameR = cv2.bitwise_and(frameR, frameR, mask=rectangle)
    cv2.imshow('RIGHT 2', newFrameR)
    imgResult, pts0, pts1, points0, points1, matches01 = matchFrame(frameL, newFrameR)
    cv2.imshow('Correspondencia', imgResult)





    print("tipo", type(pts0), type(pts1))
    
    archivo_f_l = open("results/" + nameBase + "_LEFT/" + nameBase + "_LEFT_" + str(frame_num) +  ".txt", "w")
    archivo_f_r = open("results/" + nameBase + "_RIGHT/" + nameBase + "_RIGHT_" + str(frame_num) +  ".txt", "w")
    archivo_f_l.write(str(pts0))
    archivo_f_r.write(str(pts1))
    archivo_f_l.close()
    archivo_f_r.close() 
    
    axes = viz2d.plot_images([frameL, newFrameR])
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
