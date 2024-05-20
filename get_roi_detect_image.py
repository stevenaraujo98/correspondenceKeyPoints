import cv2
import numpy as np
from getKeypointsFile import getKeypoints
from functionLightG import matchFrame
from lightglue import viz2d
import matplotlib.pyplot as plt
from ultralytics import YOLO
import glob
import os

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

conf = 0.5
robot_selected = "rosmasterx3plus"
# path_prefered = "200/calibrated/integradora/"
path_prefered = "200/calibrated/re_calibration2/"
# path_prefered = "200/YOLO/integradora/"
# path_prefered = "200/YOLO/re_calibration2/"
list_files = glob.glob('./database/rosmasterx3plus/' + path_prefered + '*_LEFT_CALIB.jpg')

for file in list_files:
    # nameBase = file[54:-15]
    nameBase = file[58:-15]


    # IMAGENES ---------------------------
    frameL=cv2.imread('./database/' + robot_selected + '/' + path_prefered + nameBase + '_LEFT_CALIB.jpg')
    # frameL=cv2.imread('./database/' + robot_selected + '/' + path_prefered + nameBase + '_LEFT_CALIB.avi')
    # frameL=cv2.imread('./database/' + robot_selected + '/' + path_prefered + nameBase + '_LEFT.avi200/YOLO/re_calibration2/' + nameBase + '_LEFT.avi')
    frameR=cv2.imread('./database/' + robot_selected + '/' + path_prefered + nameBase + '_RIGHT_CALIB.jpg')
    # frameR=cv2.imread('./database/' + robot_selected + '/' + path_prefered + nameBase + '_RIGHT_CALIB.avi')
    # frameR=cv2.imread('./database/' + robot_selected + '/' + path_prefered + nameBase + '_RIGHT.avi')
    # frameR=cv2.VideoimreadCapture('./database/' + robot_selected + '/' + path_prefered + nameBase + '_RIGHT.avi')


    h = frameR.shape[0]
    w = frameR.shape[1]

    # Modelos
    """
    resultL = model_pose.predict(frameL, conf=conf)
    resultR = model_pose.predict(frameR, conf=conf)
    keypointsL = np.array(resultL[0].keypoints.xy.cpu())#[:, [2, 5, 9, 12], :]
    keypointsR_filtered = np.array(resultR[0].keypoints.xy.cpu())#[:, [2, 5, 9, 12], :]
    """

    # resultL_seg = model_seg.predict(frameL, conf=conf, classes=[0])
    resultR_seg = model_seg.predict(frameR, conf=conf, classes=[0])

    #  Mascara
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
    # cv2.imshow('LEFT',frameL)
    # cv2.imshow('RIGHT 2', frameR * rectangle)
    newFrameR = cv2.bitwise_and(frameR, frameR, mask=rectangle)
    # cv2.imshow('RIGHT 2', newFrameR)
    imgResult, pts0, pts1, points0, points1, matches01 = matchFrame(frameL, newFrameR)
    # cv2.imshow('Correspondencia', imgResult)
    
    
    
    # Crear las carpetas para guardar los keypoints
    os.mkdir("results/" + robot_selected + "/" + path_prefered + nameBase + "_LEFT")
    os.mkdir("results/" + robot_selected + "/" + path_prefered + nameBase + "_RIGHT")
    # Gurdar los keypoints en un archivo de texto
    archivo_f_l = open("results/" + robot_selected + "/" + path_prefered + nameBase + "_LEFT/" + nameBase + "_LEFT_1.txt", "w")
    archivo_f_r = open("results/" + robot_selected + "/" + path_prefered + nameBase + "_RIGHT/" + nameBase + "_RIGHT_1.txt", "w")
    # archivo_f = open("results/" + name_video + "/frame_" + str(frame_num) +  ".txt", "w")
    archivo_f_l.write(str(pts0))
    archivo_f_r.write(str(pts1))
    archivo_f_l.close()
    archivo_f_r.close()
    
    """
    axes = viz2d.plot_images([cv2.cvtColor(frameL, cv2.COLOR_BGR2RGB), cv2.cvtColor(newFrameR, cv2.COLOR_BGR2RGB)])
    viz2d.plot_matches(points0, points1, color="lime", lw=0.2)
    viz2d.add_text(0, f'Stop after {matches01["stop"]} layers', fs=20)
    plt.show()
    """
    # Guardar imagenes con keypoints
    cv2.imwrite("results/" + robot_selected + "/" + path_prefered + nameBase + "_LEFT.jpg", frameL)
    cv2.imwrite("results/" + robot_selected + "/" + path_prefered + nameBase + "_RIGHT.jpg", newFrameR)
    cv2.imwrite("results/" + robot_selected + "/" + path_prefered + nameBase + "_CORRESP.jpg", imgResult)

    # key = cv2.waitKey(0)
    # cv2.destroyAllWindows()
