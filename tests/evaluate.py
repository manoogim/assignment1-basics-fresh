from argparse import ArgumentParser
import os
import time

import numpy as np

from tests.bpe_tokenizer import read_tokens_binary
from tests.nn_loader import load_checkpoint
from tests.nn_status_tracker import StatusTracker
from tests.nn_transformer import MyTransformer
from tests.nn_utils import add_stats, calc_validation_loss, plant_seed
from tests.nn_yaml import Config, load_yaml_config

def save_losses(ckpt_path, losses, num_eval_batches):
    parent_dir = os.path.dirname(ckpt_path)
    file_name = f'batch_losses_{num_eval_batches}.npy'
    loss_file = os.path.join(parent_dir, file_name)
    np.save(loss_file, losses)
    return loss_file

def evaluate(model, validation_tokens, config: Config, ckpt_path):
    load_checkpoint(ckpt_path, model, None, None, config.run.device)
    eval_batch_size, seq_size, num_eval_batches = config.eval.batch_size, config.model.seq_len, config.eval.num_batches
    start = time.perf_counter()
    eval_result, batch_losses = calc_validation_loss(model, validation_tokens, eval_batch_size, seq_size, num_eval_batches, config.eval.eval_seed, config.run.device)
    eval_result['duration'] = time.perf_counter() - start
    eval_result['loss_file'] = save_losses(ckpt_path, batch_losses, num_eval_batches)
    eval_result['loss_stats'] = add_stats(batch_losses)
    return eval_result

def main(args):
    cfg_path = args.config
    _, config = load_yaml_config(cfg_path, args)
    plant_seed(config.eval.eval_seed) #dont care about return value b/c each eval loop creates own generator

    model = MyTransformer.from_config(config.model, config.run.device)

    folder_name = config.data.tokens_folder
    tokens_file = os.path.join(folder_name, 'tokens_valid.bin')
    validation_tokens = read_tokens_binary(tokens_file, config.data.dtype)

    eval_result = evaluate(model, validation_tokens, config, config.eval.best_ckpt)
    StatusTracker.log(f'robust_eval: {eval_result}')
"""
1) download artifact from wandb (use repl and copy-paste under artifacts usage tab in wand, ex: artifacts/last_1349:v0/last_ckpt.pt )
2) run this script - 
2a) must override -c, -bp (and for non-kaggle eval -tf)
        -c: points to config yaml file, will be used to construct model
        -bp: path to the weights for the model state, this is the downloaded artifact
        -tf: tokens folder, normally should come from the config yaml, but if training was on kaggle, and eval is local then path fill be different
2b) should override -nb: start with 1 eval batches, to measure time.. aim for 5x or 10x number of tokens used for eval during training .. goal is to make pair-wise comparisson conclusive, so may need to increase
2c) if comparing seeds , then also override -es
3) results will be saved in the same folder where the weights came from, ex: last_1699-v0/val_losses_50.ny , where 50 is num eval batches
"""
if __name__ == '__main__':
    parser = ArgumentParser(description="Robust validation with sufficient data points on non-training data.")
    parser.add_argument('-c', '--config', type=str, default='tests/config/abla_batch/cs336_owt.yaml', help='Path to the YAML configuration file: needs only for loading validation dataset of BPE tokens')
    parser.add_argument('-es', '--eval_seed', type=int, default=12345, help="Prime number to control randomness")
    parser.add_argument('-nb', '--num_batches', type=int, default=50, help='For robust eval of validation loss')
    parser.add_argument('-bp', '--best_path', type=str, default='artifacts/last_1349-v0/last_ckpt.pt', help='Path to the best checkpoint that was created during training.')
    parser.add_argument('-tf', '--tokens_folder', type=str, default='out/owt', help='Path to bpe tokens, useful when training config points to kaggle/inputs and eval is local')
    args = parser.parse_args()

    main(args)

    