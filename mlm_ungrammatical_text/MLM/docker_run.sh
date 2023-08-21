docker run -v ~/MLM/scripts:/app/scripts -v ~/datasets:/app/datasets -v /dev/shm/:/dev/shm --gpus '"device=3,4"'  -it mlm_bernardo bash
# --gpus all
