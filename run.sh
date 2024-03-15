DATASET_DIR="/home/ids/xchen-21/FADING/data/CelebV-HQ/downloaded_celebvhq/processed"
OUTPUT_BLENDER_DIR="/home/ids/xchen-21/FADING2/output_blender"
OUTPUT_KEYFRAMES_DIR="/home/ids/xchen-21/FADING2/output"

VIDEO_NAME="0s1UUn9aSSw_1"
PROMPT_SRC="photo of a woman"
PROMPT_TGT="photo of a man"
MASK_WORD=""

#VIDEO_NAME=$1
#PROMPT_SRC=$2
#PROMPT_TGT=$3
#MASK_WORD=$4

echo "VIDEO_NAME: $VIDEO_NAME, PROMPT_SRC: $PROMPT_SRC, PROMPT_TGT: $PROMPT_TGT, MASK_WORD: $MASK_WORD"
echo "Editing key frames"
python main.py \
--video_path "$DATASET_DIR/$VIDEO_NAME.mp4" \
--prompt_src "$PROMPT_SRC" \
--prompt_tgt "$PROMPT_TGT" \
--i_mask_word "$MASK_WORD"

echo "Blending..."
cd ./Rerender_A_Video
python video_blend.py "$OUTPUT_BLENDER_DIR/$VIDEO_NAME" \
--input_video_dir "$DATASET_DIR/$VIDEO_NAME.mp4" \
--key_frames_dir "$OUTPUT_KEYFRAMES_DIR/$VIDEO_NAME-$PROMPT_TGT" \
--beg 0 --end 49 --itv 10 \
--output_frames_subdir "$PROMPT_TGT" \
--output "$OUTPUT_BLENDER_DIR/$VIDEO_NAME-$PROMPT_TGT.mp4" \
--fps 10.0 -ps
cd ..
