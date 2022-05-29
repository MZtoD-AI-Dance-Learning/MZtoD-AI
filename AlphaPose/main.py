import os
import boto3 
import json
import subprocess
import numpy as np
import torch
import torch.nn as nn

from fastapi import FastAPI, BackgroundTasks, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

import uvicorn
# from uvicorn.workers import UvicornWorker

from PE.similarity import weight_distance, cosine_distance

AWS_ACCESS_KEY_ID = ''
AWS_SECRET_KEY = ''
AWS_S3_BUCKET_NAME = ''

s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name="ap-northeast-2"
)

app = FastAPI()

origins = []

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

cos = nn.CosineSimilarity(dim=1, eps=1e-6)

@app.post('/inference')
def get_inference(file_path: str='user_LoveDive_0.mp4'):
    '''
    A inference function of AlphaPose(backbone: ResNet50, detector: YOLOv3)
    <input>
    --------------------------------------------
    - file path in S3 Bucket
    
    <return>
    --------------------------------------------
    - user_result_path: user's pose result(coordinate, confidence score etc.) save
    '''
    file_name = file_path[file_path.find('/')+1:file_path.find('.')]
    video_path = f'./input/video/{file_path}'
    user_result_path = f"./input/result/{file_name}-alphapose-results.json"

    # download user's video from s3 
    s3_client.download_file(AWS_S3_BUCKET_NAME, file_path, video_path)

    # run pose estimation using AlphaPose
    pose_inference = f"python scripts/demo_inference.py --cfg configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml \
                        --checkpoint pretrained_models/fast_res50_256x192.pth --video {video_path} --save_video --vis_fast \
                        --outdir ./input/result"
    subprocess.call(pose_inference)

    os.rename(f"./input/result/alphapose-results.json", user_result_path)

    # upload user's inference video to s3 bucket 
    upload_name = f'AlphaPose_{file_path}'
    s3_client.upload_file(f'./input/result/AlphaPose_{file_path}',
                          AWS_S3_BUCKET_NAME,
                          upload_name)
    
    return {"status": "success", "upload_name": upload_name}

# L2 Normalize
def L2_normalize(pose_vector):
    absVectorPoseXY = [0,0]
    for position in pose_vector:
        absVectorPoseXY += np.power(position, 2)
    absVectorPoseXY = np.sqrt(absVectorPoseXY)
    pose_vector /= absVectorPoseXY
    return pose_vector

@app.post('/similarity')
def get_similiarity_score(label_file_name: str='LoveDive_0', user_file_name: str='user_LoveDive_0'):
    '''
    A function of Weighted Distance Similarity (labeled pose vs user's pose)
    '''
    user_result_path: str=f'./input/result/{user_file_name}-alphapose-results.json'
    label_path: str=f'./label/result/{label_file_name}.json'

    # user result
    with open(user_result_path) as json_file:
        user_results = json.load(json_file)

    # label
    with open(label_path) as json_file:
        labels = json.load(json_file)   
   
    weight_similarity_score = []
    cos_similarity_score = []
    label_XY = []
    label_Conf = []
    user_XY = []
    user_Conf = []

    for label in labels:
        temp_Conf = []
        temp_XY = []
        for idx, lb in enumerate(label['keypoints']):
            if idx % 3 == 2:
                temp_Conf.append(lb)
            else:
                temp_XY.append(lb)
        label_Conf.append(temp_Conf)
        label_XY.append(temp_XY)
        
    for user_result in user_results:
        temp_Conf = []
        temp_XY = []
        for idx, lb in enumerate(user_result['keypoints']):
            if idx % 3 == 2:
                temp_Conf.append(lb)
            else:
                temp_XY.append(lb)
        
        user_Conf.append(temp_Conf)
        user_XY.append(temp_XY)

    # L2 Normalization 
    #user_XY = torch.nn.functional.normalize(torch.Tensor(user_XY), p=2.0, dim=1, eps=1e-12, out=None)
    #label_XY = torch.nn.functional.normalize(torch.Tensor(label_XY), p=2.0, dim=1, eps=1e-12, out=None)    
 
    for pe_result in zip(user_XY, user_Conf, label_XY):
        user_xy, user_conf, label_xy = pe_result
         # L2 Normalization 
        user_xy = torch.nn.functional.normalize(torch.Tensor(user_xy).unsqueeze(0), p=2.0, dim=1, eps=1e-12, out=None)
        label_xy = torch.nn.functional.normalize(torch.Tensor(label_xy).unsqueeze(0), p=2.0, dim=1, eps=1e-12, out=None)    
        user_xy = user_xy.squeeze(0).tolist()
        label_xy = label_xy.squeeze(0).tolist()

        # get similarity score
        weight_sim = weight_distance(user_xy, label_xy, user_conf)
        cosine_sim = cos(torch.Tensor(label_xy).unsqueeze(0), torch.Tensor(user_xy).unsqueeze(0))
        weight_similarity_score.append(weight_sim)
        cos_similarity_score.append(cosine_sim)
    
    weight_similarity_score = [100-(sim*1000) for sim in weight_similarity_score] 
    
    return {"weight_similarity_score": weight_similarity_score, "cos_similarity_score": cos_similarity_score}

if __name__ == '__main__':
    uvicorn.run("main:app", port=80, host='0.0.0.0', reload=True)

#get_inference(file_path = 'user2_LoveDive_0.mp4')
# a = get_similiarity_score(label_file_name=f'LoveDive_0', user_file_name=f'user2_LoveDive_0')
# print(a['weight_similarity_score'], len(a['weight_similarity_score']))
# print(sum(a['weight_similarity_score'])/len(a['weight_similarity_score']))