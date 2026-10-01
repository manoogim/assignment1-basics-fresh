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

def save_losses(ckpt_path, losses):
    parent_dir = os.path.dirname(ckpt_path)
    loss_file = os.path.join(parent_dir, 'batch_losses.npy')
    np.save(loss_file, losses)
    return loss_file

def evaluate(model, validation_tokens, config: Config, ckpt_path):
    load_checkpoint(model, None, ckpt_path, config.run.device)
    eval_batch_size, seq_size, num_eval_batches = config.eval.batch_size, config.model.seq_len, config.eval.num_batches
    start = time.perf_counter()
    eval_result, batch_losses = calc_validation_loss(model, validation_tokens, eval_batch_size, seq_size, num_eval_batches, config.eval.eval_seed, config.run.device)
    eval_result['duration'] = time.perf_counter() - start
    eval_result['loss_file'] = save_losses(ckpt_path, batch_losses)
    eval_result['loss_stats'] = add_stats(batch_losses)
    return eval_result

def main(args):
    cfg_path = args.config
    _, config = load_yaml_config(cfg_path, args)
    plant_seed(config.eval.eval_seed)

    model = MyTransformer.from_config(config.model, config.run.device)

    folder_name = config.data.tokens_folder
    tokens_file = os.path.join(folder_name, 'tokens_valid.bin')
    validation_tokens = read_tokens_binary(tokens_file, config.data.dtype)

    eval_result = evaluate(model, validation_tokens, config, config.eval.best_ckpt)
    StatusTracker.log(f'robust_eval: {eval_result}')

if __name__ == '__main__':
    parser = ArgumentParser(description="Robust validation with sufficient data points on never seen data.")
    parser.add_argument('-c', '--config', type=str, default='tests/config/gpt2_tiny.yaml', help='Path to the YAML configuration file: needs only for loading validation dataset of BPE tokens')
    parser.add_argument('-tb', '--token_budget', type=str, default='small', help="For compatibility only.")
    parser.add_argument('-es', '--eval_seed', type=int, default=12345, help="Prime number to control randomness")
    parser.add_argument('-nb', '--num_batches', type=int, default=100, help='For robust eval of validation loss')
    parser.add_argument('-bp', '--best_path', type=str, default='artifacts/best_ckpt-v29/best_ckpt.pt', help='Path to the best checkpoint that was used during training.')
    args = parser.parse_args()

    main(args)

    