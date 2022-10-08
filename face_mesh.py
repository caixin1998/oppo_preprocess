import os 
# from retinaface import RetinaFace
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

import json
import numpy as np
import argparse
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

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

output_file = os.path.join(input_path, "faceinfo.json")
if not os.path.isdir(input_path) or os.path.exists(output_file):
      exit() 
faceinfo_dicts = []
folder_path = os.path.join(input_path, "frames")
print(folder_path)
images = os.listdir(folder_path)

with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5) as face_mesh:
    for image in images:
        image_path = os.path.join(folder_path,image)
        print(image_path)
        image = cv2.imread(image_path)
        results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        faceinfo_dicts.append({"image_path": image_path, "faceinfo": results})
    
with open(output_file, 'w') as f:
    json.dump(faceinfo_dicts, f, cls = NpEncoder)