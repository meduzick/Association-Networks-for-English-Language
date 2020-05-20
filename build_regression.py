# -*- coding: utf-8 -*-
"""
Created on Fri May 15 21:39:24 2020

@author: nikit
"""


import tensorflow as tf
from input import get_input, MAX_RESP_LEN

CUES_MAP_SIZE = 12217

RESPONSES_MAP_SIZE = 56679



def _locate_variable(name, shape, initializer, trainable, wd):
    
    var = tf.get_variable(name = name,
                          shape = tf.TensorShape(shape),
                          initializer = initializer,
                          trainable = trainable)
    
    if wd is not None:
        
        tf.add_to_collection('losses', 
                             tf.multiply(wd, tf.nn.l2_loss(var)))
        
    return var



def _return_loss(y: tf.Tensor, logits: tf.Tensor) -> tf.Tensor:
    
    batch_loss = tf.losses.huber_loss(labels = y,
                                      predictions = logits,
                                      delta = 0.5)
    
    loss = tf.reduce_sum(tf.get_collection('losses')) + batch_loss
    
    return loss


def _return_mask(lens: tf.Tensor) -> tf.Tensor:
    
    lens = tf.expand_dims(lens, axis = 1)
    
    ones = tf.fill(dims = (1, MAX_RESP_LEN), 
                   value = 1)
    
    left_side = tf.matmul(lens, ones)
    
    right_side = tf.expand_dims(tf.range(MAX_RESP_LEN), axis = 0)
    
    binary_mask = tf.cast(tf.math.less(right_side, left_side), dtype = tf.float32)
    
    mask_ones = tf.fill(dims = (1, MAX_RESP_LEN, 100), 
                        value = 1.)
    
    
    mask = tf.multiply(mask_ones, tf.expand_dims(binary_mask, axis = -1))
    
    return mask 
    
    



def build(regime: str, epochs: int, batch_size: int, wd: float,
          learning_rate: float, buffer_size: int) -> list:
    
    summaries = []
    
    with tf.name_scope('inputs'):
        
        batch, dataset_len = get_input(regime, epochs, batch_size, buffer_size)
        
        cues, responses, lens, target = batch
        
        global_step = _locate_variable('global_step',
                                       [],
                                       tf.constant_initializer(0),
                                       False,
                                       None)
    
    with tf.variable_scope('embeddings_matrices', reuse = False):
        
        cues_embeddings = _locate_variable('cues_embeddings',
                                           shape = [CUES_MAP_SIZE, 100], 
                                           initializer = tf.random_normal_initializer(),
                                           trainable = True,
                                           wd = wd)
        
        responses_embeddings = _locate_variable('responses_embeddings',
                                                shape = [RESPONSES_MAP_SIZE, 100],
                                                initializer = tf.random_normal_initializer(),
                                                trainable = True,
                                                wd = wd)
        
    with tf.name_scope('lookup'):
        
        cues_embedded = tf.nn.embedding_lookup(cues_embeddings,
                                               cues)
        
        summaries.append(tf.summary.histogram('cues_embeddings',
                                              cues_embedded))
        
        responses_embedded_raw = tf.nn.embedding_lookup(responses_embeddings,
                                                    responses)
        
        
        mask = _return_mask(lens)
        
        responses_embedded_masked = tf.multiply(mask, responses_embedded_raw)
        
        responses_embedded = tf.reduce_sum(responses_embedded_masked,
                                           axis = 1) / tf.cast(tf.expand_dims(lens,
                                                                              axis = 1),
                                                               dtype = tf.float32)
                                                                              
        summaries.append(tf.summary.histogram('response_embeddings',
                                              responses_embedded))                                                                     
    
    
    with tf.name_scope('dense_body'):
        
        dense_input = tf.concat((cues_embedded, responses_embedded),
                            axis = 1)
        
        summaries.append(tf.summary.histogram('dense_input',
                                              dense_input))
        
        
        with tf.variable_scope('first_dense', reuse = False):
            
            w1 = _locate_variable('w', 
                                  [200, 1024],
                                  tf.contrib.layers.variance_scaling_initializer(),
                                  True,
                                  wd)
            
            b1 = _locate_variable('b',
                                  [1024, ],
                                  tf.random_normal_initializer(),
                                  True,
                                  wd)
            
        z1 = tf.nn.elu(tf.matmul(dense_input, w1) + b1)
            
        with tf.variable_scope('second_dense', reuse = False):
            
            w2 = _locate_variable('w', 
                                  [1024, 512],
                                  tf.contrib.layers.variance_scaling_initializer(),
                                  True,
                                  wd)
            
            b2 = _locate_variable('b',
                                  [512, ],
                                  tf.random_normal_initializer(),
                                  True,
                                  wd)
            
        z2 = tf.nn.elu(tf.matmul(z1, w2) + b2)
        
        with tf.variable_scope('third_dense', reuse = False):
            
            w3 = _locate_variable('w', 
                                  [512, 256],
                                  tf.contrib.layers.variance_scaling_initializer(),
                                  True,
                                  wd)
            
            b3 = _locate_variable('b',
                                  [256, ],
                                  tf.random_normal_initializer(),
                                  True,
                                  wd)
            
        z3 = tf.nn.elu(tf.matmul(z2, w3) + b3)
        
        
        with tf.variable_scope('forth_dense', reuse = False):
            
            w4 = _locate_variable('w', 
                                  [256, 128],
                                  tf.contrib.layers.variance_scaling_initializer(),
                                  True,
                                  wd)
            
            b4 = _locate_variable('b',
                                  [128, ],
                                  tf.random_normal_initializer(),
                                  True,
                                  wd)
            
        z4 = tf.nn.elu(tf.matmul(z3, w4) + b4)
        
        
        with tf.variable_scope('fivth_dense', reuse = False):
            
            w5 = _locate_variable('w', 
                                  [128, 64],
                                  tf.contrib.layers.variance_scaling_initializer(),
                                  True,
                                  wd)
            
            b5 = _locate_variable('b',
                                  [64, ],
                                  tf.random_normal_initializer(),
                                  True,
                                  wd)
            
        z5 = tf.nn.elu(tf.matmul(z4, w5) + b5)
        
        
        with tf.variable_scope('sixth_dense', reuse = False):
            
            w6 = _locate_variable('w', 
                                  [64, 32],
                                  tf.contrib.layers.variance_scaling_initializer(),
                                  True,
                                  wd)
            
            b6 = _locate_variable('b',
                                  [32, ],
                                  tf.random_normal_initializer(),
                                  True,
                                  wd)
            
        z6 = tf.nn.elu(tf.matmul(z5, w6) + b6)
        
        
        with tf.variable_scope('final_dense', reuse = False):
            
            w_final = _locate_variable('w', 
                                  [32, 1],
                                  tf.contrib.layers.variance_scaling_initializer(),
                                  True,
                                  wd)
            
            b_final = _locate_variable('b',
                                  [1, ],
                                  tf.random_normal_initializer(),
                                  True,
                                  wd)
            
        
        with tf.variable_scope('skip_connection', reuse = False):
            
            w_sc = _locate_variable('w', 
                                  [200, 1],
                                  tf.contrib.layers.variance_scaling_initializer(),
                                  True,
                                  wd)
            
            
            
        
        z_sc = tf.matmul(dense_input, w_sc)
        
        logits = tf.nn.sigmoid(tf.squeeze(tf.matmul(z6, w_final)  + b_final + z_sc), 
                            name = 'sig_logits')
                               
        
        summaries.append(tf.summary.histogram('logits',
                                              logits))
        
        
    with tf.name_scope('mae'):
        
        mae, metric_update_op = tf.metrics.mean_absolute_error(labels = target,
                                                      predictions = logits
                                                      )
        
        running_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES,
                                              scope = 'mae/mean_absolute_error')
        
        running_variables_init = tf.variables_initializer(var_list = running_variables)
        
        mae_summary = tf.summary.scalar('mae',
                                        mae)
        
        
    with tf.name_scope('loss'):
    
        loss = _return_loss(target, logits)
        
        summaries.append(tf.summary.scalar('loss',
                                            loss))
        
        
    
    with tf.name_scope('optimization'):
        
        with tf.variable_scope('optimizer', reuse = False):
            
            optimizer = tf.train.AdamOptimizer(learning_rate)
            
        
        grads_vars = optimizer.compute_gradients(loss)
        
        train_step = optimizer.apply_gradients(grads_vars,
                                             global_step = global_step)
        
    
    with tf.name_scope('gradients'):
        
        for grad, var in grads_vars:
            
            if grad is not None:
                
                summaries.append(tf.summary.histogram('gradient_at_{}'.format(var.op.name),
                                                      grad))
                
    with tf.name_scope('variables'):
        
        for var in tf.trainable_variables():
            
            summaries.append(tf.summary.histogram('var_at_{}'.format(var.op.name),
                                                  var))
            
    
    merge_op = tf.summary.merge(summaries)
    
    
    dense_activations = tf.get_collection(tf.GraphKeys.SUMMARIES,
                                         scope = 'dense_body')
    
    input_summaries = tf.get_collection(tf.GraphKeys.SUMMARIES,
                                        scope = 'lookup')
    
    test_summaries_op = tf.summary.merge(dense_activations + input_summaries)
    
    mae_merge_op = tf.summary.merge([mae_summary])
    
    
    with tf.name_scope('saving'):
        
        saver = tf.train.Saver()
        
    
    return global_step, train_step, merge_op, test_summaries_op, \
        saver, dataset_len, mae_merge_op, running_variables_init, \
            metric_update_op
    
    
            
        
        
                                                                              
                                                                              
                                                                              
        
                                                                             
                                                                              
                                                                              
                                                                              
                                                                              
        
        
        
        
        
        
  