import argparse

# Arguments for data
def add_data_options(parser):
    parser.add_argument('-save_path', default='')
    parser.add_argument('-train_from_state_dict', default='', type=str)
    parser.add_argument('-train_from', default='', type=str)

    parser.add_argument('-online_process_data', action="store_true")
    parser.add_argument('-process_shuffle', action="store_true")
    parser.add_argument('-train_src')
    parser.add_argument('-src_vocab')
    parser.add_argument('-train_tgt')
    parser.add_argument('-tgt_vocab')

    parser.add_argument('-dev_input_src')
    parser.add_argument('-dev_ref')
    parser.add_argument('-beam_size', type=int, default=12)
    parser.add_argument('-max_sent_length', type=int, default=100)

# Arguments for models
def add_model_options(parser):
    parser.add_argument('-layers', type=int, default=6)
    parser.add_argument('-enc_size', type=int, default=1024)
    parser.add_argument('-dec_size', type=int, default=1024)

    parser.add_argument('-word_vec_size', type=int, default=1024)
    parser.add_argument('-maxout_pool_size', type=int, default=2)
    parser.add_argument('-input_feed', type=int, default=1)
    # parser.add_argument('-residual',   action="store_true")

# Arguments for train step
def add_train_options(parser):
    parser.add_argument('-batch_size', type=int, default=512,)
    parser.add_argument('-max_generator_batches', type=int, default=32)
    parser.add_argument('-epochs', type=int, default=20)
    parser.add_argument('-start_epoch', type=int, default=1)
    parser.add_argument('-param_init', type=float, default=0.1)
    parser.add_argument('-optim', default='sgd',
                        help="""Optimization method. [sgd(1)|adagrad(0.1)
                        |adadelta(1)|adam(0.001)]""")
    parser.add_argument('-max_grad_norm', type=float, default=5)
    parser.add_argument('-max_weight_value', type=float, default=15)
    parser.add_argument('-dropout', type=float, default=0.3)
    parser.add_argument('-curriculum', type=int, default=1)
    parser.add_argument('-extra_shuffle', action="store_true")

    # learning rate
    parser.add_argument('-learning_rate', type=float, default=0.001)
    parser.add_argument('-learning_rate_decay', type=float, default=0.5)
    parser.add_argument('-start_decay_at', type=int, default=8)
    parser.add_argument('-start_eval_batch', type=int, default=15000)
    parser.add_argument('-eval_per_batch', type=int, default=1000)
    parser.add_argument('-halve_lr_bad_count', type=int, default=6)

    # pretrained word vectors
    parser.add_argument('-pre_word_vecs_enc')
    parser.add_argument('-pre_word_vecs_dec')

    # GPU
    parser.add_argument('-gpus', default=[], nargs='+', type=int,
                        help="Use CUDA on the listed devices.")
    parser.add_argument('-log_interval', type=int, default=100,
                        help="logger.info stats at this interval.")
    parser.add_argument('-seed', type=int, default=-1)
    parser.add_argument('-cuda_seed', type=int, default=-1)
    parser.add_argument('-log_home', default='')

