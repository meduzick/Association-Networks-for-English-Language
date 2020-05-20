# -*- coding: utf-8 -*-
"""
Created on Sun May 17 15:57:06 2020

@author: nikit
"""


import tensorflow as tf
from input import get_input

VOCAB_DIM = 28

EMBEDDING_DIM = 100

NUM_CLASSES = 2


def _locate_variable(name, shape, initializer, trainable, wd):
    
    var = tf.get_variable(name = name,
                          shape = tf.TensorShape(shape),
                          initializer = initializer,
                          trainable = trainable)
    
    if wd is not None:
        
        tf.add_to_collection('losses', 
                             tf.multiply(wd, tf.nn.l2_loss(var)))
        
    return var



def _return_loss(y: tf.Tensor, logits: tf.Tensor, class_weights: tf.Tensor) -> tf.Tensor:
    
    weights = tf.gather(class_weights, y)
    
    batch_loss = tf.losses.sparse_softmax_cross_entropy(labels = y,
                                    logits = logits,
                                    weights = weights
                                    )
    
    loss = tf.reduce_sum(tf.get_collection('losses')) + batch_loss
    
    return loss




def build(regime: str, epochs: int, batch_size: int, 
          learning_rate: float, lstm_units_cues: int,
          lstm_units_resps: int, buffer_size: int, wd: float) -> list:
    
    summaries = []
    
    with tf.name_scope('inputs'):
        
        batch, dataset_len = get_input(regime,
                                        epochs, 
                                        batch_size,
                                        buffer_size)
        
        cues, cues_lens, resps, resps_lens, target = batch
        
        
        global_step = tf.get_variable('global_step',
                                      shape = tf.TensorShape([]),
                                      initializer = tf.constant_initializer(0),
                                      trainable = False)
        
        class_weights = tf.constant(
                                   [0.78125421, 1.38887557]
        )
        
        
    with tf.variable_scope('char_embeddings', reuse = False):
        
        cue_embeddings = _locate_variable('cue_embeddings',
                                      shape = [VOCAB_DIM, EMBEDDING_DIM],
                                      initializer = tf.random_normal_initializer(),
                                      trainable = True,
                                      wd = wd)
        
        resps_embeddings = _locate_variable('resps_embeddings',
                                      shape = [VOCAB_DIM, EMBEDDING_DIM],
                                      initializer = tf.random_normal_initializer(),
                                      trainable = True,
                                      wd = wd)
        
    
    with tf.name_scope('rnn_for_cues'):
        
        with tf.name_scope('cues_embeddings_lookup'):
            
             cues_embedded = tf.nn.embedding_lookup(cue_embeddings,
                                                    cues)
             
             summaries.append(tf.summary.histogram('cues_embedded',
                                                   cues_embedded))
             
             
        with tf.variable_scope('rnn_setup_cue', reuse = False):
            
            fw_cell_cue = tf.nn.rnn_cell.LSTMCell(lstm_units_cues)
            
            bw_cell_cue = tf.nn.rnn_cell.LSTMCell(lstm_units_cues)
        
        
            outputs_cue, hiddens_cue = tf.nn.bidirectional_dynamic_rnn(cell_fw = fw_cell_cue,
                                                               cell_bw = bw_cell_cue,
                                                               inputs = cues_embedded,
                                                               sequence_length = cues_lens,
                                                               dtype = tf.float32)
        
        with tf.variable_scope('cue_attention', reuse = False):
            
            cue_attention = tf.get_variable('attention',
                                        shape = [2 * lstm_units_cues, 1],
                                        initializer = tf.orthogonal_initializer(),
                                        trainable = True)
            
        
        outputs = tf.concat(outputs_cue, axis = -1)
        
        
        cue_alphas = tf.nn.tanh(
            tf.reshape(
                tf.matmul(tf.reshape(outputs, shape = (-1, 2 * lstm_units_cues)),
                           cue_attention), 
                shape = (-1, 16))
            )
        
        cue_attention_vector = tf.matmul(tf.transpose(outputs, perm = [0, 2, 1]),
                                     tf.reshape(cue_alphas, shape = (-1, 16, 1)))
        
        c_attention = tf.reshape(cue_attention_vector,
                                 shape = (-1, 2 * lstm_units_cues))
        
        c_hiddens = tf.concat((hiddens_cue[0].h, hiddens_cue[1].h, c_attention),
                              axis = 1)
        
        
        summaries.append(tf.summary.histogram('cue_rnn_outputs',
                         outputs))
        
        summaries.append(tf.summary.histogram('cue_attention',
                                              c_attention))
        
        
    with tf.name_scope('rnn_for_resps'):
        
        with tf.name_scope('resps_embeddings_lookup'):
            
             resps_embedded = tf.nn.embedding_lookup(resps_embeddings,
                                                    resps)
             
             summaries.append(tf.summary.histogram('resps_embedded', 
                                                   resps_embedded))
             
        with tf.variable_scope('rnn_setup', reuse = False):
            
            fw_cell_r = tf.nn.rnn_cell.LSTMCell(lstm_units_resps)
            
            bw_cell_r = tf.nn.rnn_cell.LSTMCell(lstm_units_resps)
        
        
            outputs_r, hiddens_r = tf.nn.bidirectional_dynamic_rnn(cell_fw = fw_cell_r,
                                                               cell_bw = bw_cell_r,
                                                               inputs = resps_embedded,
                                                               sequence_length = resps_lens,
                                                               dtype = tf.float32)
        
        
        
        with tf.variable_scope('resp_attention', reuse = False):
            
            resp_attention = tf.get_variable('attention',
                                        shape = [2 * lstm_units_resps, 1],
                                        initializer = tf.orthogonal_initializer(),
                                        trainable = True)
            
        
        outputs_r = tf.concat(outputs_r, axis = -1)
        
        
        r_alphas = tf.nn.tanh(
            tf.reshape(
                tf.matmul(tf.reshape(outputs_r, shape = (-1, 2 * lstm_units_resps)),
                           resp_attention), 
                shape = (-1, 44))
            )
        
        resp_attention_vector = tf.matmul(tf.transpose(outputs_r, perm = [0, 2, 1]),
                                     tf.reshape(r_alphas, shape = (-1, 44, 1)))
        
        r_attention = tf.reshape(resp_attention_vector, 
                                 shape = (-1, 2 * lstm_units_resps))
        
        resp_hiddens = tf.concat((hiddens_r[0].h, hiddens_r[1].h, r_attention),
                                 axis = 1)
        
        
        summaries.append(tf.summary.histogram('cue_rnn_outputs',
                         outputs_r))
        
        summaries.append(tf.summary.histogram('resp_attention',
                         r_attention))
        
        
        
        
    with tf.name_scope('dense_body'):
        
        dense_input = tf.concat((c_hiddens, resp_hiddens),
                                axis = 1,
                                name = 'dense_representations')
        
        summaries.append(tf.summary.histogram('dense_input',
                                              dense_input))
        
        
        with tf.variable_scope('dense', reuse = False):
            
            w = tf.get_variable('w',
                                shape = tf.TensorShape([4 * lstm_units_cues + \
                                                        4 * lstm_units_resps, 
                                                        NUM_CLASSES]),
                                initializer = tf.orthogonal_initializer(),
                                trainable = True)
            
            b = tf.get_variable('b',
                                shape = tf.TensorShape([NUM_CLASSES, ]),
                                initializer = tf.random_normal_initializer(),
                                trainable = True)
            
        logits = tf.matmul(dense_input, w) + b
        
        
        predictions = tf.argmax(tf.nn.softmax(logits, axis = 1),
                                axis = 1)
        
        summaries.append(tf.summary.histogram('logits',
                                              logits))
        
        
    with tf.name_scope('accuracy'):
    
        accuracy, metric_update_op = tf.metrics.accuracy(labels = target,
                                                      predictions = predictions,
                                                      name = 'accuracy')
        
        running_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES,
                                              scope = 'accuracy/accuracy')
        
        running_variables_init = tf.variables_initializer(var_list = running_variables)
        
        accuracy_summary = tf.summary.scalar('accuracy',
                                        accuracy)
    
        
    
    with tf.name_scope('loss'):
        
        loss = _return_loss(target,
                            logits,
                            class_weights)
        
        summaries.append(tf.summary.scalar('loss',
                                           loss))
        
    
    with tf.name_scope('optimization'):
        
        with tf.variable_scope('adam'):
            
            optimizer = tf.train.AdamOptimizer(learning_rate)
            
        
        grads_vars = optimizer.compute_gradients(loss)
        
        op_step = optimizer.apply_gradients(grads_vars, 
                                            global_step = global_step)
        
        
        for grad, var in grads_vars:
            
            if grad is not None:
                
                summaries.append(tf.summary.histogram('grad_at_{}'.format(var.op.name),
                                                      grad))
                
        for var in tf.trainable_variables():
            
            summaries.append(tf.summary.histogram('var_at_{}'.format(var.op.name),
                                                  var))
            
            
    merge_op = tf.summary.merge(summaries)
    
    activation_summaries  = tf.get_collection(tf.GraphKeys.SUMMARIES,
                                              scope = 'rnn_for_cues') + \
                            tf.get_collection(tf.GraphKeys.SUMMARIES,
                                              scope = 'rnn_for_resps') + \
                            tf.get_collection(tf.GraphKeys.SUMMARIES,
                                              scope = 'dense_body')
                            
    activations_merge_op = tf.summary.merge(activation_summaries)
        
    
    acc_merge_op = tf.summary.merge([accuracy_summary])
    
    train_step = tf.group((op_step, metric_update_op))
        
    
    with tf.name_scope('saving'):
        
        saver = tf.train.Saver()
        
        
    return global_step, train_step, merge_op, activations_merge_op, \
        acc_merge_op, saver, running_variables_init, metric_update_op, \
            dataset_len
    
    
            
            
    
    