CUDA_VISIBLE_DEVICES=0,1 \
python main.py --base configs/autoencoder/seg_diff_autoencoder.yaml \
                -t \
                --gpus 1,    