import os
import boto3 
import subprocess

from fastapi import FastAPI, BackgroundTasks, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

import uvicorn
# from uvicorn.workers import UvicornWorker

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

@app.post('/inference')
def get_inference(file_path: str='5sec.mp4'):
    '''
    A inference function of AlphaPose(backbone: ResNet50, detector: YOLOv3)
    '''
    file_name = file_path[file_path.find('/')+1:file_path.find('.')]
    video_path = f'./output/{file_path}'
    result_path = './result/'
    s3_client.download_file(AWS_S3_BUCKET_NAME, file_path, video_path)

    pose_inference = f"python scripts/demo_inference.py --cfg configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml \
                        --checkpoint pretrained_models/fast_res50_256x192.pth --video {video_path} --save_video --vis_fast \
                        --outdir ./result"
    subprocess.call(pose_inference)

    os.rename(f"{result_path}alphapose-results.json", f"{result_path}{file_name}-alphapose-results.json")
    
    return True

@app.post('/similarity')
def get_similiarity(file_path: str='5sec.mp4'):
    '''
    A function of Weighted Distance Similarity (labeled pose vs user's pose)
    '''
    return True

if __name__ == '__main__':
    uvicorn.run("main:app", port=8000, host='0.0.0.0', reload=True)
