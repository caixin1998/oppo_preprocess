import math, shutil, os, time, argparse, json, re, sys
import numpy as np
# import scipy.io as sio
from PIL import Image
import cv2

parser = argparse.ArgumentParser(description='iTracker-pytorch-PrepareDataset.')
parser.add_argument('--dataset_path', help="Path to extracted files. It should have folders called '%%05d' in it.")
parser.add_argument('--output_path', default=None, help="Where to write the output. Can be the same as dataset_path if you wish (=default).")
args = parser.parse_args()


def cropImage(img, bbox):
    bbox = np.array(bbox, int)

    aSrc = np.maximum(bbox[:2], 0)
    bSrc = np.minimum(bbox[:2] + bbox[2:], (img.shape[1], img.shape[0]))

    aDst = aSrc - bbox[:2]
    bDst = aDst + (bSrc - aSrc)

    res = np.zeros((bbox[3], bbox[2], img.shape[2]), img.dtype)    
    res[aDst[1]:bDst[1],aDst[0]:bDst[0],:] = img[aSrc[1]:bSrc[1],aSrc[0]:bSrc[0],:]

    return res

def parse_faceinfo(faceinfo):
    # if faceinfo["score"] > 0.5:
    
    w = int((faceinfo["facial_area"][2] + faceinfo["facial_area"][3]) / 2)
    h = w
    x = int(faceinfo["facial_area"][0])
    y = int(faceinfo["facial_area"][1])
    
    face_bbox = [x,y,w,h]
    
    return face_bbox

def readJson(filename):
    if not os.path.isfile(filename):
        logError('Warning: No such file %s!' % filename)
        return None

    with open(filename) as f:
        try:
            data = json.load(f)
        except:
            data = None

    if data is None:
        logError('Warning: Could not read file %s!' % filename)
        return None

    return data
    
        # leye_x = int(faceinfo["landmarks"]["left_eye"][0] - w / 6)
        # leye_y = int(faceinfo["landmarks"]["left_eye"][1] - h / 6)
        # leye_w = int(w/3)
        # leye_h = leye_w

        # leye_bbox = [leye_x, leye_y, leye_w, leye_h]

        # reye_x = int(faceinfo["landmarks"]["right_eye"][0] - w / 6)
        # reye_y = int(faceinfo["landmarks"]["right_eye"][1] - h / 6)
        # reye_w = int(w/3)
        # reye_h = leye_w

        # reye_bbox = [reye_x, reye_y, reye_w, reye_h]

        # return (face_bbox, leye_bbox, reye_bbox)
 
faceInfo = readJson(os.path.join(args.dataset_path, "facedect.json"))
dotInfo = readJson(os.path.join(args.dataset_path, 'dotInfo.json'))
os.makedirs(os.path.join(args.output_path, "Label"), exist_ok = True)
os.makedirs(os.path.join(args.output_path, "Image"), exist_ok = True)

label_outpath = os.path.join(args.output_path, "Label", os.path.basename(args.dataset_path) + ".txt")
outfile = open(label_outpath, 'w')
outfile.write("Face FaceGrid PoG\n")

# print(len(faceInfo), len(dotInfo['XCam']))
faceInfo.sort(key=lambda faceInfo:faceInfo["image_path"])
assert len(faceInfo) == len(dotInfo['XCam'])
for i, frameinfo in enumerate(faceInfo): 
    image_path = frameinfo["image_path"]
    if frameinfo["faceinfo"] is None:
        continue
    faceinfo = frameinfo["faceinfo"][0]
    bbox = parse_faceinfo(faceinfo)
    image = cv2.imread(os.path.join(os.path.dirname(args.dataset_path),image_path))
    image_width, image_height = image.shape[1], image.shape[0]
    if image_width >= image_height:
        grid_x = bbox[0] / image_width
        grid_y = bbox[1] / image_width + (image_width - image_height) / 2 / image_width
        grid_w = bbox[2] / image_width
    else:
        grid_x = bbox[0] / image_height + (image_height - image_width) / 2 / image_height
        grid_y = bbox[1] / image_height 
        grid_w = bbox[2] / image_height
    grid_h = grid_w

    # face = cropImage(image, bbox)
    face_path = image_path.replace("frames", "faces").replace("jpg", "png")
    os.makedirs(os.path.dirname(os.path.join(args.output_path, "Image", face_path)), exist_ok = True)
    # cv2.imwrite(os.path.join(args.output_path, "Image", face_path), face)
    save_name_face = face_path
    save_gaze = "%s,%s"%(dotInfo['XCam'][i], dotInfo['YCam'][i])
    save_grid = "%s,%s,%s,%s"%(grid_x, grid_y, grid_w, grid_h)
    save_str =  " ".join([save_name_face,save_grid,save_gaze]) 

    outfile.write(save_str + "\n")

print("")
outfile.close()
    # if bboxs is not None:

