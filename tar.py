# import os 
# from tqdm import tqdm
# data_path = "/data1/GazeData/CaptureFace/Image"
# out_path = "/data1/GazeData/Image"
# os.makedirs(out_path, exist_ok = True)
# persons = os.listdir(data_path)
# persons.sort()
# for person in tqdm(persons):
#     if not os.path.isdir(os.path.join(data_path, person)):
#         continue
#     os.system("tar -zvcf %s/%s.tar.gz -C %s %s"%(out_path, person, data_path,  person))
    # break

import os 
from tqdm import tqdm
data_path = "/home1/caixin/GazeData/MPKU_Kid/B"
out_path = "/data1/GazeData/Image"
os.makedirs(out_path, exist_ok = True)
persons = os.listdir(data_path)
persons.sort()
for person in tqdm(persons):
    if not os.path.isdir(os.path.join(data_path, person)):
        continue
    os.system("tar -zvcf %s/%s.tar.gz -C %s %s"%(out_path, person, data_path,  person))