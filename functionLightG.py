from torchvision.transforms.functional import resize, to_pil_image
from lightglue import LightGlue, SuperPoint, DISK, viz2d
import torchvision.transforms as transforms
from lightglue.utils import rbd

import numpy as np
import torch
import cv2

def get_kps(kps_fps, w=1920, h=1080):
    list_pts = []
    for person in kps_fps:
        numItem = 5
        for i in range(17):
            pos_x = float(person[numItem])*w
            pos_y = float(person[numItem+1])*h
            numItem += 3
            list_pts.append([pos_x, pos_y])
    return list_pts

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'mps', 'cpu'

extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)  # load the extractor
# or DISK+LightGlue, ALIKED+LightGlue or SIFT+LightGlue
# extractor = DISK(max_num_keypoints=2048).eval().cuda()  # load the extractor
matcher = LightGlue(features="superpoint").eval().to(device)

def matchFrame(image0, image1):
    image0 = cv2.cvtColor(image0, cv2.COLOR_BGR2RGB)
    transform = transforms.ToTensor()
    image0 = transform(image0)

    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    transform = transforms.ToTensor()
    image1 = transform(image1)

    feats0 = extractor.extract(image0.to(device))
    feats1 = extractor.extract(image1.to(device))

    
    matches01 = matcher({"image0": feats0, "image1": feats1})
    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # remove batch dimension

    #matches = matches01['matches']  # indices with shape (K,2)
    matches, scores = matches01["matches"], matches01["scores"]
    points0 = feats0['keypoints'][matches[..., 0]]  # coordinates in image #0, shape (K,2)
    points1 = feats1['keypoints'][matches[..., 1]]  # coordinates in image #1, shape (K,2)

    pts0 = points0.cpu().numpy()
    pts1 = points1.cpu().numpy()

    # Find the homography matrix that maps all points from pts0 to pts1
    M1, _ = cv2.findHomography(pts0, pts1)
    # Convert the PyTorch image tensors to PIL images for warpPerspective
    image0_pil = to_pil_image(image0)

    # Apply perspective transformation using warpPerspective
    image0_warped = cv2.warpPerspective(np.array(image0_pil), M1, (image1.shape[2], image1.shape[1]))
    warped_image0 = torch.tensor(image0_warped).permute(2, 0, 1)

    result_image_np = np.transpose(warped_image0.cpu().detach().numpy(), (1, 2, 0)) 

    return cv2.cvtColor(result_image_np, cv2.COLOR_BGR2RGB), pts0, pts1, points0, points1, matches01

