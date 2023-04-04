# currently model is base
# crop is 512
# iter is 160k

_base_ = [
    '../_base_/models/upernet_convnext.py', '../_base_/datasets/custom_voc_512_temp.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k_adamw.py'
]
crop_size = (512, 512)

model = dict(
    backbone=dict(
        type='ConvNeXt',
        in_chans=3,
        depths=[3, 3, 27, 3],
        dims=[256, 512, 1024, 2048],
        drop_path_rate=0.4,
        layer_scale_init_value=1.0,
        out_indices=[0, 1, 2, 3],
    ),
    decode_head=dict(
        in_channels=[256, 512, 1024, 2048],
        num_classes=150,
    ),
    auxiliary_head=dict(
        in_channels=1024,
        num_classes=150
    ),
)

custom_imports = dict(
    imports=['mmcls.core.optimizer.layer_decay_optimizer_constructor'], allow_failed_imports=False)

# optimizer
optimizer = dict(constructor='LearningRateDecayOptimizerConstructor', _delete_=True, type='AdamW',
                 lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg={'decay_rate': 0.9,
                                'decay_type': 'stage_wise',
                                'num_layers': 12})

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

# By default, models are trained on 8 GPUs with 2 images per GPU
# data=dict(samples_per_gpu=2)

runner = dict(type='IterBasedRunner')

# do not use mmdet version fp16
fp16 = None
runner = dict(type='IterBasedRunner', max_iters=160000)
checkpoint_config = dict(by_epoch=False, interval=16000)
evaluation = dict(interval=16000, metric='mIoU')
