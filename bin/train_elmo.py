
import argparse

import numpy as np

from bilm.training import train, load_options_latest_checkpoint, load_vocab
from bilm.data import BidirectionalLMDataset


def main(args):
    # load the vocab
    vocab = load_vocab(args.vocab_file, 50)

    # define the options
    batch_size = args.batch_size  # batch size for each GPU
    n_gpus = args.n_gpus

    # number of tokens in training data (this for 1B Word Benchmark)
    n_train_tokens = args.ntokens

    n_negative_samples_batch=8192
    if n_negative_samples_batch > vocab.size:
        n_negative_samples_batch=int(vocab.size/2)


    options = {
     'bidirectional': True,

     'char_cnn': {'activation': 'relu',
      'embedding': {'dim': 16},
      'filters': [[1, 32],
       [2, 32],
       [3, 64],
       [4, 128],
       [5, 256],
       [6, 512],
       [7, 1024]],
      'max_characters_per_token': 50,
      'n_characters': 261,
      'n_highway': 2},
    
     'dropout': 0.1,
    
     'lstm': {
      'cell_clip': 3,
      'dim': 4096,
      'n_layers': 2,
      'proj_clip': 3,
      'projection_dim': 512,
      'use_skip_connections': True},
    
     'all_clip_norm_val': 10.0,
    
     'n_epochs': args.n_epochs,
     'n_train_tokens': n_train_tokens,
     'batch_size': batch_size,
     'n_tokens_vocab': vocab.size,
     'unroll_steps': 20,
     'n_negative_samples_batch': n_negative_samples_batch,
    }

    prefix = args.train_prefix
    data = BidirectionalLMDataset(prefix, vocab, test=False,
                                      shuffle_on_load=True)

    tf_save_dir = args.save_dir
    tf_log_dir = args.save_dir
    print("NGPUS in train_elmo: %i" %(n_gpus,))
    train(options, data, n_gpus, tf_save_dir, tf_log_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', help='Location of checkpoint files')
    parser.add_argument('--vocab_file', help='Vocabulary file')
    parser.add_argument('--train_prefix', help='Prefix for train files')
    parser.add_argument('--n_gpus', help="Number of GPUs", type=int, default = 1)
    parser.add_argument('--ntokens',type = int, help="Number of tokens in Trainingset")
    parser.add_argument('--batch_size', help="Number of tokens in Trainingset",type=int, default = 128)
    parser.add_argument("--n_epochs", type = int, help="Number of epochs",default = 1)

    args = parser.parse_args()
    main(args)


#python bin/train_elmo.py --save_dir save/ --vocab_file data/vocabulary --train_prefix data/train/* --n_gpus 2 --ntokens $(wc -l data/vocabulary | awk -F" " '{print$1}') --batch_size 128 --n_epochs=1
