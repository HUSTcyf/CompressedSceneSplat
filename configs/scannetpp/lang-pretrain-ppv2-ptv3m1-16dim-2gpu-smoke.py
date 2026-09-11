_base_ = ["lang-pretrain-ppv2-ptv3m1-16dim.py"]

# smoke: 2-GPU training logic test (no BeginningEvaluator, short run)
epoch = 1
eval_epoch = 1
save_path = "exp/smoke_ppv2_16dim_2gpu"
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="CheckpointSaver", save_freq=None),
]
