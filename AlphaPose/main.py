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

from similarity import weight_distance, cosine_distance

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
def get_inference(file_path: str='5sec.mp4'):
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
    video_path = f'./output/{file_path}'
    user_result_path = f"./result/{file_name}-alphapose-results.json"
    s3_client.download_file(AWS_S3_BUCKET_NAME, file_path, video_path)

    pose_inference = f"python scripts/demo_inference.py --cfg configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml \
                        --checkpoint pretrained_models/fast_res50_256x192.pth --video {video_path} --save_video --vis_fast \
                        --outdir ./result"
    subprocess.call(pose_inference)

    os.rename(f"./result/alphapose-results.json", user_result_path)
    
    return user_result_path

@app.post('/similarity')
def get_similiarity_score(user_result_path: str='./result/5sec-alphapose-results.json',
                          label_path: str='./label/5sec.json'):
    '''
    A function of Weighted Distance Similarity (labeled pose vs user's pose)
    '''
    # user result
    with open(user_result_path) as json_file:
        user_results = json.load(json_file)

    # label
    with open(label_path) as json_file:
        labels = json.load(json_file)   
   
    similarity_score = []
    for label in labels:
        label_XY = []
        label_Conf = []

        for idx, lb in enumerate(label['keypoints']):
            if idx % 3 == 2:
                label_Conf.append(lb)
            else:
                label_XY.append(lb)

        # get simliartiy score
        weight_sim = ((1-weight_distance(label_XY, label_XY, label_Conf))*100)
        cosine_sim = cos(torch.Tensor(label_XY).unsqueeze(0), torch.Tensor(label_XY).unsqueeze(0))
        similarity_score.append(weight_sim)

    return similarity_score

if __name__ == '__main__':
    uvicorn.run("main:app", port=8000, host='0.0.0.0', reload=True)