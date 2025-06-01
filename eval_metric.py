from tools.fid_score import calculate_fid_given_paths, get_activations
import ml_collections
import torch
from torch import multiprocessing as mp
import accelerate
import utils
import sde
from datasets import get_dataset
import tempfile
from dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver
from absl import logging
import builtins
import os
import numpy as np
from tools.inception import InceptionV3


def calculate_precision_recall(real_features, fake_features, k=3):
    real_features = real_features.cpu()
    fake_features = fake_features.cpu()

    real_dists = torch.cdist(real_features, real_features, p=2)
    real_dists.fill_diagonal_(float('inf'))

    kth_dists, _ = real_dists.kthvalue(k, dim=1)

    tau = kth_dists.median().item()

    dists_fake2real = torch.cdist(fake_features, real_features, p=2)
    min_dists_fake, _ = dists_fake2real.min(dim=1)
    precision = (min_dists_fake < tau).float().mean().item()

    dists_real2fake = torch.cdist(real_features, fake_features, p=2)
    min_dists_real, _ = dists_real2fake.min(dim=1)
    recall = (min_dists_real < tau).float().mean().item()
    
    return precision, recall


def evaluate(config):
    if config.get('benchmark', False):
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    mp.set_start_method('spawn')
    accelerator = accelerate.Accelerator()
    device = accelerator.device
    accelerate.utils.set_seed(config.seed, device_specific=True)
    logging.info(f'Process {accelerator.process_index} using device: {device}')

    config.mixed_precision = accelerator.mixed_precision
    config = ml_collections.FrozenConfigDict(config)
    if accelerator.is_main_process:
        utils.set_logger(log_level='info', fname=config.output_path)
    else:
        utils.set_logger(log_level='error')
        builtins.print = lambda *args: None

    dataset = get_dataset(**config.dataset)

    nnet = utils.get_nnet(**config.nnet)
    nnet = accelerator.prepare(nnet)
    logging.info(f'load nnet from {config.nnet_path}')
    accelerator.unwrap_model(nnet).load_state_dict(torch.load(config.nnet_path, map_location='cpu'))
    nnet.eval()
    if 'cfg' in config.sample and config.sample.cfg and config.sample.scale > 0:  # classifier free guidance
        logging.info(f'Use classifier free guidance with scale={config.sample.scale}')
        def cfg_nnet(x, timesteps, y):
            _cond = nnet(x, timesteps, y=y)
            _uncond = nnet(x, timesteps, y=torch.tensor([dataset.K] * x.size(0), device=device))
            return _cond + config.sample.scale * (_cond - _uncond)
        score_model = sde.ScoreModel(cfg_nnet, pred=config.pred, sde=sde.VPSDE())
    else:
        score_model = sde.ScoreModel(nnet, pred=config.pred, sde=sde.VPSDE())

    logging.info(config.sample)
    logging.info(f'sample: n_samples={config.sample.n_samples}, mode={config.train.mode}, mixed_precision={config.mixed_precision}')

    def sample_fn(_n_samples):
        x_init = torch.randn(_n_samples, *dataset.data_shape, device=device)
        if config.train.mode == 'uncond':
            kwargs = dict()
        elif config.train.mode == 'cond':
            kwargs = dict(y=dataset.sample_label(_n_samples, device=device))
        else:
            raise NotImplementedError

        if config.sample.algorithm == 'euler_maruyama_sde':
            rsde = sde.ReverseSDE(score_model)
            return sde.euler_maruyama(rsde, x_init, config.sample.sample_steps, verbose=accelerator.is_main_process, **kwargs)
        elif config.sample.algorithm == 'euler_maruyama_ode':
            rsde = sde.ODE(score_model)
            return sde.euler_maruyama(rsde, x_init, config.sample.sample_steps, verbose=accelerator.is_main_process, **kwargs)
        elif config.sample.algorithm == 'dpm_solver':
            noise_schedule = NoiseScheduleVP(schedule='linear')
            model_fn = model_wrapper(
                score_model.noise_pred,
                noise_schedule,
                time_input_type='0',
                model_kwargs=kwargs
            )
            dpm_solver = DPM_Solver(model_fn, noise_schedule)
            return dpm_solver.sample(
                x_init,
                steps=config.sample.sample_steps,
                eps=1e-4,
                adaptive_step_size=False,
                fast_version=True,
            )
        else:
            raise NotImplementedError

    with tempfile.TemporaryDirectory() as temp_path:

        gen_path = config.sample.path or temp_path
        if accelerator.is_main_process:
            os.makedirs(gen_path, exist_ok=True)

        utils.sample2dir(accelerator, gen_path, config.sample.n_samples, config.sample.mini_batch_size, sample_fn, dataset.unpreprocess)
        
        if accelerator.is_main_process:
            real_path = config.get('real_images_path', dataset.fid_stat)

            if os.path.isdir(real_path):
                fid = calculate_fid_given_paths((real_path, gen_path))
                logging.info(f'nnet_path={config.nnet_path}, fid={fid}')
            else:
                if os.path.exists(dataset.fid_stat):
                    fid = calculate_fid_given_paths((dataset.fid_stat, gen_path))
                    logging.info(f'nnet_path={config.nnet_path}, fid={fid} (using pre-computed statistics)')
                else:
                    logging.error(f'Both real_images_path and dataset.fid_stat are invalid. Cannot calculate FID.')
                    fid = None
            
            if os.path.isdir(real_path):
                dims = 2048
                block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
                model = InceptionV3([block_idx]).to(device)
                
                batch_size = config.sample.get('feature_batch_size', 50)

                from pathlib import Path
                real_files = sorted([str(f) for ext in ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp'] 
                                    for f in Path(real_path).glob(f'*.{ext}')])
                gen_files = sorted([str(f) for ext in ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp'] 
                                   for f in Path(gen_path).glob(f'*.{ext}')])

                if len(real_files) > 0 and len(gen_files) > 0:
                    # Extract features
                    logging.info(f'Extracting features for {len(real_files)} real images and {len(gen_files)} generated images...')
                    real_features = torch.from_numpy(get_activations(real_files, model, batch_size, dims, device))
                    fake_features = torch.from_numpy(get_activations(gen_files, model, batch_size, dims, device))

                    k = config.sample.get('precision_recall_k', 3)
                    precision, recall = calculate_precision_recall(real_features, fake_features, k=k)
                    
                    logging.info(f'Precision: {precision:.4f}, Recall: {recall:.4f}')

                    metrics = {
                        'fid': fid,
                        'precision': precision,
                        'recall': recall
                    }

                    metrics_path = os.path.join(os.path.dirname(config.output_path), 'metrics.txt')
                    with open(metrics_path, 'w') as f:
                        for name, value in metrics.items():
                            f.write(f'{name}: {value}\n')
                    
                    logging.info(f'Metrics saved to {metrics_path}')
                else:
                    logging.error(f'Not enough images found in real or generated directories to calculate precision/recall.')
            

from absl import flags
from absl import app
from ml_collections import config_flags
import os


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=False)
flags.mark_flags_as_required(["config"])
flags.DEFINE_string("nnet_path", None, "The nnet to evaluate.")
flags.DEFINE_string("output_path", None, "The path to output log.")
flags.DEFINE_string("real_images_path", None, "Path to real images for FID and precision/recall calculation.")


def main(argv):
    config = FLAGS.config
    config.nnet_path = FLAGS.nnet_path
    config.output_path = FLAGS.output_path
    if FLAGS.real_images_path:
        config.real_images_path = FLAGS.real_images_path
    evaluate(config)


if __name__ == "__main__":
    app.run(main)