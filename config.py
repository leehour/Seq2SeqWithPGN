import os

# stop_words
stop_words_path = os.path.join(os.path.abspath('./'), 'datasets', 'stopwords.txt')

# dataset
train_path = os.path.join(os.path.abspath('./'), 'datasets', 'AutoMaster_TrainSet.csv')
test_path = os.path.join(os.path.abspath('./'), 'datasets', 'AutoMaster_TestSet.csv')

# file after split words
train_seg_path = os.path.join(os.path.abspath('./'), 'datasets', 'train_seg.csv')
test_seg_path = os.path.join(os.path.abspath('./'), 'datasets', 'test_seg.csv')

# merge lines
train_seg_merge_path = os.path.join(os.path.abspath('./'), 'datasets', 'train_seg_merge.csv')
test_seg_merge_path = os.path.join(os.path.abspath('./'), 'datasets', 'test_seg_merge.csv')

# split data to three files
train_seg_x_path = os.path.join(os.path.abspath('./'), 'datasets', 'train_seg_x.csv')
train_seg_target_path = os.path.join(os.path.abspath('./'), 'datasets', 'train_seg_target.csv')
test_seg_x_path = os.path.join(os.path.abspath('./'), 'datasets', 'test_seg_x.csv')

# parent directory os.path.join(os.path.abspath('..'), 'datasets', 'train_seg_x.csv')
train_seg_x_parent_path = os.path.join(os.path.abspath('..'), 'datasets', 'train_seg_x.csv')
train_seg_target_parent_path = os.path.join(os.path.abspath('..'), 'datasets', 'train_seg_target.csv')
test_seg_x_parent_path = os.path.join(os.path.abspath('..'), 'datasets', 'test_seg_x.csv')

# all sentences
sentences_path = os.path.join(os.path.abspath('./'), 'datasets', 'sentences.txt')

# vocab path
vocab_path = os.path.join(os.path.abspath('./'), 'datasets', 'vocab.txt')

# word2vec txt
w2v_output_path = os.path.join(os.path.abspath('./'), 'datasets', 'word2vec.txt')

# Word2Vec模型存放路径
w2v_bin_path = os.path.join(os.path.abspath('./'), 'model', 'model.bin')

# checkpoints 存储路径
checkpoint_dir = os.path.join(os.path.abspath('./'), 'training_checkpoints', 'checkpoint')

# result path
result_path = os.path.join(os.path.abspath('./'), 'datasets', 'result.csv')
test_save_dir = os.path.join(os.path.abspath('./'), 'datasets', 'result_pgn.txt')
log_dir = "./log/"

embedding_size = 256
max_words_size = 30000
max_input_size = 500
max_target_size = 100
dataset_num = 100

# open gru switch
open_bigru = False

EPOCHS = 5
BATCH_SIZE = 32
units = 512
test_data_size = 10

params = {
    'learning_rate': 0.0001,
    'adagrad_init_acc': 0.1,
    'max_grad_norm': 2,
    'vocab_size': 50000,
    'embedding_dim': 256,
    'enc_units': 256,
    'dec_units': 256,
    'attn_units': 512,
    'batch_size': 32,
    "checkpoint_dir": checkpoint_dir,
    'model_dir': "../model",
    "max_enc_len": 500,
    "max_dec_len": 100,
    "mode": "train",
    "checkpoints_save_steps": 500,
    "max_steps": 10000,
    "num_to_test": 5,
    "test_save_dir": test_save_dir,
    "vocab_path": vocab_path,
    "beam_size": 4,
    "test_data_size": test_data_size,
    "log_dir": log_dir,
    "epochs": 3

}
