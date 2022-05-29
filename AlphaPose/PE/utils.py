from moviepy.editor import *
from typing import List, Tuple

def cut_off_video(video_path: str, cut_list: List[List]):
    for idx, cut in enumerate(cut_list):
        print(f'start: {cut[0]}, end: {cut[1]}')
        clip = VideoFileClip(video_path).subclip(cut[0], cut[1])
        clip.to_videofile(f"../input/video/new_user2_LoveDive_5.mp4",  audio_codec='aac',fps=23.98)
        #clip.write_videofile(f"../output/LoveDive_{idx}.mp4")

if __name__ == '__main__':
    video_path = '../input/video/user2_LoveDive_5.mp4'
    cut_list = [[0,10]]
    cut_off_video(video_path, cut_list)