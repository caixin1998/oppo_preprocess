import os 
# from retinaface import RetinaFace
import mediapipe as mp
import json
import numpy as np
import argparse
import cv2
import math
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int,
    image_height: int):
  """Converts normalized value pair to pixel coordinates."""

  x_px = min(math.floor(normalized_x * image_width), image_width - 1)
  y_px = min(math.floor(normalized_y * image_height), image_height - 1)
  return x_px, y_px

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str, default='inputs', help='Input image or folder')
args = parser.parse_args()
# datasets_path = ["/home1/caixin/GazeData/GazeCapture"]
# for dataset_path in datasets_path:
    # output_path = dataset_path
    # # output_path = os.path.join(os.path.dirname(dataset_path),os.path.basename(dataset_path) + "_faceinfo")
    # # os.makedirs(output_path, exist_ok=True)
    # folders = os.listdir(dataset_path)
input_path = args.input

# for folder in folders:
# if not os.path.isdir(os.path.join(output_path, folder))
#     break

output_file = os.path.join(input_path, "facedect.json")
# if not os.path.isdir(input_path) or os.path.exists(output_file):
#       exit() 
faceinfo_dicts = []
folder_path = os.path.join(input_path, "frames")
print(folder_path)
images = os.listdir(folder_path).sort()
os.makedirs("tmp/%s"%os.path.basename(input_path), exist_ok = True)

with mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.9) as face_detection:
    for i, image in enumerate(images):
        image_path = os.path.join(folder_path,image)
        image = cv2.imread(image_path)
        results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        rel_image_path = image_path.split("GazeCapture/")[1]
        if not results.detections:
            faceinfo_dicts.append({"image_path": rel_image_path, "faceinfo": None})
            print(rel_image_path, " : None")
            continue 

        image_width, image_height = image.shape[1], image.shape[0]
        dect_results = []
        for j, detection in enumerate(results.detections):
            normalized_keypoints = detection.location_data.relative_keypoints
            normalized_bbox = detection.location_data.relative_bounding_box

            x,y = normalized_to_pixel_coordinates(normalized_bbox.xmin, normalized_bbox.ymin, image_width, image_height)
            w,h = normalized_to_pixel_coordinates(normalized_bbox.width, normalized_bbox.height, image_width, image_height)
            landmarks = []
            for idx, landmark in enumerate(normalized_keypoints):
                landmark_px = normalized_to_pixel_coordinates(landmark.x, landmark.y,
                                                   image_width, image_height)
                landmarks.append(landmark_px)   

            dect_results.append({"landmarks": landmarks, "facial_area": [x,y,w,h]})

        if i % 100 == 0:

            rect_start_point = normalized_to_pixel_coordinates(
                normalized_bbox.xmin, normalized_bbox.ymin, image_width,
                image_height)
            rect_end_point = normalized_to_pixel_coordinates(
                normalized_bbox.xmin + normalized_bbox.width,
                normalized_bbox.ymin + normalized_bbox.height, image_width,
                image_height)

            annotated_image = image.copy()
            cv2.rectangle(annotated_image, rect_start_point, rect_end_point,
                (224, 224, 224), 2)
            
            for landmark_px in landmarks:
                cv2.circle(annotated_image, landmark_px, 2,
               (0, 0, 224), 2)        

            cv2.imwrite('tmp/%s/annotated_image_'%os.path.basename(input_path) + str(i) + '.png', annotated_image)
        
        # print(rel_image_path)
        faceinfo_dicts.append({"image_path": rel_image_path, "faceinfo": dect_results})
    
with open(output_file, 'w') as f:
    json.dump(faceinfo_dicts, f, cls = NpEncoder)