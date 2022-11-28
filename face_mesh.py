import os 
# from retinaface import RetinaFace
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
import cv2
import json
import math
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

def normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int,
    image_height: int):
  """Converts normalized value pair to pixel coordinates."""

  x_px = min(math.floor(normalized_x * image_width), image_width - 1)
  y_px = min(math.floor(normalized_y * image_height), image_height - 1)
  return [x_px, y_px]


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

output_file = os.path.join(input_path, "facemesh.json")
# if not os.path.isdir(input_path) or os.path.exists(output_file):
#       exit() 
faceinfo_dicts = []
folder_path = os.path.join(input_path, "frames")
# print(folder_path)
images = os.listdir(folder_path)
images.sort()
# print(images)
os.makedirs("tmp/%s"%os.path.basename(input_path), exist_ok = True)

with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5) as face_mesh:
        for i, image in enumerate(images):
                image_path = os.path.join(folder_path,image)
                image = cv2.imread(image_path)
                results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                rel_image_path = image_path.split("GazeCapture/")[1]
                if not results.multi_face_landmarks:
                    faceinfo_dicts.append({"image_path": rel_image_path, "faceinfo": None})
                    print(rel_image_path, " : None")
                    continue 

                image_width, image_height = image.shape[1], image.shape[0]
                dect_results = []

                for j, face_landmarks in enumerate(results.multi_face_landmarks):
                    landmarks = []
                    for idx, landmark in enumerate(face_landmarks.landmark):
                        landmark_px  = normalized_to_pixel_coordinates(landmark.x, landmark.y, image_width, image_height)    
                        landmarks.append(landmark_px) 
                    landmarks_np = np.array(landmarks)

                    rightEyeUpper0 = [246, 161, 160, 159, 158, 157, 173] 
                    rightEyeLower0 = [33, 7, 163, 144, 145, 153, 154, 155, 133]

                    leftEyeUpper0 = [466, 388, 387, 386, 385, 384, 398]
                    leftEyeLower0 = [263, 249, 390, 373, 374, 380, 381, 382, 362]
                    rightEye_idx = rightEyeUpper0 + rightEyeLower0
                    leftEye_idx = leftEyeUpper0 + leftEyeLower0
                    right_eye_center = np.mean(landmarks_np[rightEye_idx], axis = 0)
                    left_eye_center = np.mean(landmarks_np[leftEye_idx], axis = 0)
                    eye_center = (left_eye_center + right_eye_center) / 2
                    distance = (left_eye_center - right_eye_center)[0]
                    # print(distance)
                    gamma = 1.2
                    beta = 0.5
                    # x = eye_center[0] - distance  * gamma 
                    y = math.floor(eye_center[1] - distance * beta)
                    # w = distance * gamma * 2 
                    # h = w
                    w = math.floor(np.max(landmarks_np[:,0]) - np.min(landmarks_np[:,0])) 
                    x = math.floor(np.min(landmarks_np[:,0]))
                    h = w
                    dect_results.append({"landmarks": landmarks, "facial_area": [x,y,w,h]})

                if i % 100 == 0:

                    # rect_start_point = normalized_to_pixel_coordinates(
                    #     normalized_bbox.xmin, normalized_bbox.ymin, image_width,
                    #     image_height)
                    # rect_end_point = normalized_to_pixel_coordinates(
                    #     normalized_bbox.xmin + normalized_bbox.width,
                    #     normalized_bbox.ymin + normalized_bbox.height, image_width,
                    #     image_height)

                    rect_start_point = np.array([x, y], dtype = np.int32)
                    rect_end_point = np.array([x + w, y + h], dtype = np.int32)
                    # print(rect_start_point, rect_end_point)
                    annotated_image = image.copy()
                    # face_image = image.copy()
                    for face_landmarks in results.multi_face_landmarks:
                        mp_drawing.draw_landmarks(
                            image=annotated_image,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles
                            .get_default_face_mesh_tesselation_style())
                        mp_drawing.draw_landmarks(
                            image=annotated_image,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_CONTOURS,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles
                            .get_default_face_mesh_contours_style())
                        mp_drawing.draw_landmarks(
                            image=annotated_image,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_IRISES,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles
                            .get_default_face_mesh_iris_connections_style())

                    cv2.rectangle(annotated_image, rect_start_point, rect_end_point,
                    (224, 224, 224), 2)
                    
                    cv2.imwrite('tmp/%s/mesh_image_'%os.path.basename(input_path) + str(i) + '.png', annotated_image)
                    # cv2.imwrite('tmp/%s/face_image_'%os.path.basename(input_path) + str(i) + '.png', face_image)
                # if i == 500:
                #     break
                # print(rel_image_path)
                faceinfo_dicts.append({"image_path": rel_image_path, "faceinfo": dect_results})
    
with open(output_file, 'w') as f:
    json.dump(faceinfo_dicts, f, cls = NpEncoder)