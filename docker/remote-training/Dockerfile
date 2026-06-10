# syntax=docker/dockerfile:1.7
# Remote training image — AMD64 + CUDA variant with all training deps pre-baked.
# Used by RunPod and Vast.ai backends. Eliminates the 4-6 min pip install that
# happens on every pod start when using the vanilla pytorch image.
#
# Uses the CUDA *runtime* base (not -devel): every baked dependency ships
# prebuilt CUDA kernels (bitsandbytes) or is pure Python (transformers, peft,
# accelerate, datasets), so the ~3-5 GB CUDA toolkit/compilers in -devel are
# dead weight. Dropping them shrinks the image pulled from ECR to the remote
# GPU provider on every run — the dominant data-transfer-out cost.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

RUN pip install --no-cache-dir \
    transformers>=5.7.0 \
    datasets>=4.8.5 \
    accelerate>=1.1.0 \
    bitsandbytes>=0.43.0 \
    peft>=0.9.0 \
    sentencepiece>=0.2.0 \
    boto3 \
    runpod

# Tells bootstrap.py that heavy deps are already present so it skips
# reinstalling [training] extras and only installs the project wheel itself.
ENV TRAINING_DEPS_PREINSTALLED=1
