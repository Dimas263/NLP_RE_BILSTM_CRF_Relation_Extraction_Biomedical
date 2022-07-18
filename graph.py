# coding:utf-8
import os
import tempfile
import random
import json
import logging
from termcolor import colored
import modeling
import args
from args import PoolingStrategy
import contextlib


def import_tf(device_id=-1, verbose=False):
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1' if device_id < 0 else str(device_id)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0' if verbose else '3'
    import tensorflow as tf
    tf.logging.set_verbosity(tf.logging.DEBUG if verbose else tf.logging.ERROR)
    return tf


def set_logger(context, verbose=False):
    logger = logging.getLogger(context)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    formatter = logging.Formatter(
        '%(levelname)-.1s:' + context + ':[%(filename).5s:%(funcName).3s:%(lineno)3d]:%(message)s', datefmt=
        '%m-%d %H:%M:%S')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_handler.setFormatter(formatter)
    logger.handlers = []
    logger.addHandler(console_handler)
    return logger


def optimize_graph(logger=None, verbose=False, pooling_strategy=PoolingStrategy.REDUCE_MEAN, max_seq_len=40):
    if not logger:
        logger = set_logger(colored('BERT_VEC', 'yellow'), verbose)
    try:
        # we don't need GPU for optimizing the graph
        tf = import_tf(device_id=0, verbose=verbose)
        from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference

        # allow_soft_placement:Automatically select running devices
        config = tf.ConfigProto(allow_soft_placement=True)
        config_fp = args.config_name
        init_checkpoint = args.ckpt_name
        logger.info('model config: %s' % config_fp)

        # Load the bert configuration file
        with tf.gfile.GFile(config_fp, 'r') as f:
            bert_config = modeling.BertConfig.from_dict(json.load(f))

        logger.info('build graph...')
        # input placeholders, not sure if they are friendly to XLA
        input_ids = tf.placeholder(tf.int32, (None, max_seq_len), 'input_ids')
        input_mask = tf.placeholder(tf.int32, (None, max_seq_len), 'input_mask')
        input_type_ids = tf.placeholder(tf.int32, (None, max_seq_len), 'input_type_ids')

        # xla acceleration
        jit_scope = tf.contrib.compiler.jit.experimental_jit_scope if args.xla else contextlib.suppress

        with jit_scope():
            input_tensors = [input_ids, input_mask, input_type_ids]

            model = modeling.BertModel(
                config=bert_config,
                is_training=False,
                input_ids=input_ids,
                input_mask=input_mask,
                token_type_ids=input_type_ids,
                use_one_hot_embeddings=False)

            # Get all variables to train
            tvars = tf.trainable_variables()

            (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,
                                                                                                       init_checkpoint)

            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

            minus_mask = lambda x, m: x - tf.expand_dims(1.0 - m, axis=-1) * 1e30
            mul_mask = lambda x, m: x * tf.expand_dims(m, axis=-1)
            masked_reduce_max = lambda x, m: tf.reduce_max(minus_mask(x, m), axis=1)
            masked_reduce_mean = lambda x, m: tf.reduce_sum(mul_mask(x, m), axis=1) / (
                    tf.reduce_sum(m, axis=1, keepdims=True) + 1e-10)

            # Shared convolution kernel
            with tf.variable_scope("pooling"):
                # If there is only one layer, only the weight corresponding to that layer is taken.
                if len(args.layer_indexes) == 1:
                    encoder_layer = model.all_encoder_layers[args.layer_indexes[0]]
                else:
                    # Otherwise, traverse the layers to be taken, take out the weights of all layers and
                    # splicing them together shape: 768*number of layers
                    all_layers = [model.all_encoder_layers[l] for l in args.layer_indexes]
                    encoder_layer = tf.concat(all_layers, -1)

                input_mask = tf.cast(input_mask, tf.float32)

                # The following code is the generation method of the sentence vector.
                # It can be understood as a convolution operation, but the results are not added.
                # The convolution kernel is input_mask
                if pooling_strategy == PoolingStrategy.REDUCE_MEAN:
                    pooled = masked_reduce_mean(encoder_layer, input_mask)
                elif pooling_strategy == PoolingStrategy.REDUCE_MAX:
                    pooled = masked_reduce_max(encoder_layer, input_mask)
                elif pooling_strategy == PoolingStrategy.REDUCE_MEAN_MAX:
                    pooled = tf.concat([masked_reduce_mean(encoder_layer, input_mask),
                                        masked_reduce_max(encoder_layer, input_mask)], axis=1)
                elif pooling_strategy == PoolingStrategy.FIRST_TOKEN or \
                        pooling_strategy == PoolingStrategy.CLS_TOKEN:
                    pooled = tf.squeeze(encoder_layer[:, 0:1, :], axis=1)
                elif pooling_strategy == PoolingStrategy.LAST_TOKEN or \
                        pooling_strategy == PoolingStrategy.SEP_TOKEN:
                    seq_len = tf.cast(tf.reduce_sum(input_mask, axis=1), tf.int32)
                    rng = tf.range(0, tf.shape(seq_len)[0])
                    indexes = tf.stack([rng, seq_len - 1], 1)
                    pooled = tf.gather_nd(encoder_layer, indexes)
                elif pooling_strategy == PoolingStrategy.NONE:
                    pooled = mul_mask(encoder_layer, input_mask)
                else:
                    raise NotImplementedError()

            pooled = tf.identity(pooled, 'final_encodes')

            output_tensors = [pooled]
            tmp_g = tf.get_default_graph().as_graph_def()

        with tf.Session(config=config) as sess:
            logger.info('load parameters from checkpoint...')
            sess.run(tf.global_variables_initializer())
            logger.info('freeze...')
            tmp_g = tf.graph_util.convert_variables_to_constants(sess, tmp_g, [n.name[:-2] for n in output_tensors])
            dtypes = [n.dtype for n in input_tensors]
            logger.info('optimize...')
            tmp_g = optimize_for_inference(
                tmp_g,
                [n.name[:-2] for n in input_tensors],
                [n.name[:-2] for n in output_tensors],
                [dtype.as_datatype_enum for dtype in dtypes],
                False)
        # tmp_file = tempfile.NamedTemporaryFile('w', delete=True).name
        # r = random.randint(1, 1000)
        # tmp_file = "./tmp_graph"+str(r)
        tmp_file = "/content/drive/MyDrive/Rearch_Dimas/BILSTM_CRF_RE/tmp_graph11"
        logger.info('write graph to a tmp file: %s' % tmp_file)
        with tf.gfile.GFile(tmp_file, 'wb') as f:
            f.write(tmp_g.SerializeToString())
        return tmp_file
    except Exception as e:
        logger.error('fail to optimize the graph!')
        logger.error(e)
