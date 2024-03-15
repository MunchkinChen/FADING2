#%%
import numpy as np
import imageio
from skimage.transform import resize
from skimage import img_as_ubyte
from PIL import Image
import os

#%% read video frames
# for video_name in ["juNQcMYkfBM_1", "oH3u2QHoR2A_3", "iJROOxmADv8_7", "SN8buDY-7LM_0", "mrm31EMpIi8_13"]+["6Bet906AgNE_0", "92s-RpM7Tks_6", "DIdGRrayLCM_8"]:
for video_name in ["0s1UUn9aSSw_1"]:
    video_path = f"/home/ids/xchen-21/FADING/data/CelebV-HQ/downloaded_celebvhq/processed/{video_name}.mp4"
    # video_path = f"/home/ids/xchen-21/FADING2/output_blender/{video_name}-photo of a 5 year old girl.mp4"
    reader = imageio.get_reader(video_path)
    fps = reader.get_meta_data()['fps']

    video_frames = []
    try:
        for im in reader:
            video_frames.append(Image.fromarray(im).resize((512, 512)))
    except RuntimeError:
        reader.close()

    #%% save at given # frames and fps
    imageio.mimsave(f'./output_blender/{video_name}-input.mp4',
                    [img_as_ubyte(frame) for frame in video_frames[:40]], fps=10)
    print(f'{video_name} saved')
