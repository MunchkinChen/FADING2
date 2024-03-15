#!/bin/bash
DATASET_DIR="/home/ids/xchen-21/FADING/data/CelebV-HQ/downloaded_celebvhq/processed"
OUTPUT_BLENDER_DIR="/home/ids/xchen-21/FADING2/output_blender"
OUTPUT_KEYFRAMES_DIR="/home/ids/xchen-21/FADING2/output"

VIDEO_NAME="0s1UUn9aSSw_1"
PROMPT_TGT="photo of a 70 year old woman"

python video_blend.py "$OUTPUT_BLENDER_DIR/$VIDEO_NAME" \
--input_video_dir "$DATASET_DIR/$VIDEO_NAME.mp4" \
--key_frames_dir "$OUTPUT_KEYFRAMES_DIR/$VIDEO_NAME-$PROMPT_TGT" \
--beg 0 --end 49 --itv 10 \
--output_frames_subdir "$PROMPT_TGT" \
--output "$OUTPUT_BLENDER_DIR/$VIDEO_NAME.mp4" \
--fps 10.0 -ps

