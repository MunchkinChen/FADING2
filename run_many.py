#%%
import subprocess
import random

video_names = ["juNQcMYkfBM_1", "oH3u2QHoR2A_3", "iJROOxmADv8_7", "SN8buDY-7LM_0", "mrm31EMpIi8_13"]
video_names2 = ["6Bet906AgNE_0", "92s-RpM7Tks_6", "DIdGRrayLCM_8"]
prompt_pairs = [
    [
        "photo of a 25 year old woman",
        "photo of a 75 year old woman",
        ""
    ],
    [
        "photo of a 25 year old woman",
        "photo of a 5 year old girl",
        ""
    ],
    [
        "photo of a woman with blond hair",
        "photo of a woman with red hair",
        "hair"
    ],
    [
        "photo of a woman",
        "photo of a woman with curly hair",
        "hair"
    ],
    [
        "photo of a woman",
        "photo of a woman with a bang",
        "bang"
    ],
    [
        "photo of a woman",
        "photo of a woman with a tanned skin",
        "skin"
    ],
    [
        "photo of a woman",
        "photo of a woman with freckles",
        "freckles"
    ],
    [
        "photo of a woman",
        "photo of a woman with sunglasses",
        "sunglasses"
    ],
    [
        "photo of a woman",
        "photo of a woman with a mask",
        "mask"
    ],
    [
        "photo of a woman with open eyes",
        "photo of a woman with closed eyes",
        "eyes"
    ],
    [
        "photo of a woman with closed eyes",
        "photo of a woman with open eyes",
        "eyes"
    ],
    [
        "photo of a woman",
        "photo of a woman wearing big earrings",
        "earrings"
    ],
    [
        "photo of a woman",
        "photo of a woman wearing a necklace",
        "necklace"
    ],
]
prompt_pairs2 = [
    [
        "photo of a 40 year old man",
        "photo of a 75 year old man",
        ""
    ],
    [
        "photo of a 40 year old man",
        "photo of a 5 year old boy",
        ""
    ],
    [
        "photo of a man",
        "photo of a woman",
        ""
    ],
    [
        "photo of a man",
        "photo of a man wearing glasses",
        "glasses"
    ],
    [
        "photo of a man",
        "photo of a man with a moustache",
        "moustache"
    ],

]
params = []

for video_name in video_names:
    for prompt_pair in prompt_pairs:
        params.append([video_name]+prompt_pair)
for video_name in video_names2:
    for prompt_pair in prompt_pairs2:
        params.append([video_name]+prompt_pair)

random.shuffle(params)

for param in params:
    param_ = [f"'{p}'" for p in param]
    cmd = "bash run.sh " + ' '.join(param_)
    print(cmd)
    process = subprocess.run(cmd, shell=True, capture_output=True)
    print(process.stdout)