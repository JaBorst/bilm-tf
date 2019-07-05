from bilm.training import train, load_options_latest_checkpoint, load_vocab
from bilm.data import BidirectionalLMDataset


# if __name__ == '__main__':
# parser = argparse.ArgumentParser()
# parser.add_argument('--save_dir', help='Location of checkpoint files')
# parser.add_argument('--vocab_file', help='Vocabulary file')
# parser.add_argument('--train_prefix', help='Prefix for train files')
#
# args = parser.parse_args()


vocab_file = "data/vocabulary"
vocab = load_vocab(vocab_file, 50)
vocabsize = len(list(vocab._word_to_id.values()))

# define the options
batch_size = 128  # batch size for each GPU
n_gpus = 2

# number of tokens in training data (this for 1B Word Benchmark)
n_train_tokens = 48952

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

    'n_epochs': 10,
    'n_train_tokens': n_train_tokens,
    'batch_size': batch_size,
    'n_tokens_vocab': vocab.size,
    'unroll_steps': 20,
    'n_negative_samples_batch': 4096,
}

prefix = "data/train/train.txt"
data = BidirectionalLMDataset(prefix, vocab, test=False,
                              shuffle_on_load=True)

tf_save_dir = "test/save/"
tf_log_dir =  "test/logs/"
train(options, data, n_gpus, tf_save_dir, tf_log_dir)