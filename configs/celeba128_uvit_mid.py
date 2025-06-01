import ml_collections


def d(**kwargs):
    """Helper of creating a config dict."""
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ml_collections.ConfigDict()

    config.seed = 1234
    config.pred = 'noise_pred'

    config.train = d(
        n_steps=500000,
        batch_size=8,
        mode='uncond',
        log_interval=50000,
        eval_interval=510000,
        save_interval=50000,
    )

    config.optimizer = d(
        name='adamw',
        lr=0.0002,
        weight_decay=0.03,
        betas=(0.99, 0.99),
    )

    config.lr_scheduler = d(
        name='customized',
        warmup_steps=5000
    )

    config.nnet = d(
        name='uvit',
        img_size=128,
        patch_size=4,
        embed_dim=768,
        depth=16,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=False,
        mlp_time_embed=False,
        num_classes=-1,
    )

    config.dataset = d(
        name='celebahq',
        path='../data/celeba_hq_256',
        resolution=128,
    )

    config.sample = d(
        sample_steps=100,
        n_samples=50000,
        mini_batch_size=8,
        # algorithm='euler_maruyama_sde',
        algorithm='dpm_solver',
        path=''
    )

    return config
