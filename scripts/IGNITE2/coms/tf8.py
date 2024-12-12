import tensorflow as tf
import os 

for lambda_ in [0.01]:
    for rho in [0.2]:
        for r in [0.2]:
            for iter in range(16):
                os.system(f'coms-IGNITE2 \
                        --logging-dir results/ours/lambda_{lambda_}/rho_{rho}/r_{r}/coms-IGNITE2/tf-bind-8/ \
                        --task TFBind8-Exact-v0 \
                        --no-task-relabel \
                        --normalize-ys \
                        --no-normalize-xs \
                        --not-in-latent-space \
                        --vae-hidden-size 64 \
                        --vae-latent-size 256 \
                        --vae-activation relu \
                        --vae-kernel-size 3 \
                        --vae-num-blocks 4 \
                        --vae-lr 0.0003 \
                        --vae-beta 1.0 \
                        --vae-batch-size 128 \
                        --vae-val-size 500 \
                        --vae-epochs 2 \
                        --particle-lr 2.0 \
                        --particle-train-gradient-steps 50 \
                        --particle-evaluate-gradient-steps 50 \
                        --particle-entropy-coefficient 0.0 \
                        --forward-model-activations relu \
                        --forward-model-activations relu \
                        --forward-model-hidden-size 2048 \
                        --no-forward-model-final-tanh \
                        --forward-model-lr 0.0003 \
                        --forward-model-alpha 0.1 \
                        --forward-model-alpha-lr 0.01 \
                        --forward-model-overestimation-limit 2.0 \
                        --forward-model-noise-std 0.0 \
                        --forward-model-batch-size 128 \
                        --forward-model-val-size 500 \
                        --forward-model-epochs 2 \
                        --evaluation-samples 128 \
                        --fast \
                        --lambda_ {lambda_} \
                        --rho {rho} \
                        --r {r} \
                        ')
