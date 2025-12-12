#!/bin/bash
set -e # exit on error

# activate conda environment
source /home/cizinsky/miniconda3/etc/profile.d/conda.sh
module load gcc ffmpeg
conda activate difix

output_dir="/home/cizinsky/difix3d/playground/outputs/"
src_cam_id=28
tgt_cam_id=4

python playground/demo.py \
  --model-id nvidia/difix_ref \
  --root-dir /scratch/izar/cizinsky/thesis/preprocessing/hi4d_pair17_dance/lhm/evaluation/difix/epoch_0000 \
  --src-cam-id $src_cam_id \
  --tgt-cam-id $tgt_cam_id \
  --prompt "remove degradation" \
  --output-dir $output_dir \
  --num-images 125 \
  --num-inference-steps 1 \
  --timesteps 199 \
  --guidance-scale 0.0 \
  --torch-home /scratch/izar/cizinsky/.cache \
  --hf-home /scratch/izar/cizinsky/.cache \
  --device cuda

ffmpeg -y -framerate 20 -pattern_type glob -i "$output_dir/refined_cam$tgt_cam_id/frames/*.png" "$output_dir/refined_cam$tgt_cam_id/refined_cam$tgt_cam_id.gif"