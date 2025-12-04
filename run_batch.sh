docker run --rm \
  --gpus all \
  -v $(pwd)/input_audio:/app/input_audio \
  -v $(pwd)/batch_output:/app/batch_output \
  -v $(pwd)/.env:/app/.env \
  whisper-batch-optimized
