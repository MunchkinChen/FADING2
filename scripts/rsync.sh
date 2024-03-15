VIDEO_NAME="0s1UUn9aSSw_1"
PROMPT_TGT="photo of a man"
rsync -avz --progress "xchen-21@gpu-gw:'/home/ids/xchen-21/FADING2/output_blender/$VIDEO_NAME-$PROMPT_TGT.mp4'" \
"../output_blender/$VIDEO_NAME-$PROMPT_TGT.mp4"

