# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 12:15:37 2020

@author: nikit
"""


import tensorflow as tf
from build import build
from os.path import join

LOG_DIR = join('tb_logs', 'val')



def eval(batch_size: int, learning_rate: float, 
         save_dir: str, wd: float, buffer_size: int) -> None:
    
    graph = tf.Graph()
    
    
    print(80 * "=")
    print("BUILDING COMPUTATIONAL GRAPH...")
    print(80 * "=")
    
    
    with graph.as_default():
        
        global_step, train_step, merge_op, test_summaries_op, \
        saver, dataset_len, mae_merge_op, running_variables_init, \
        metric_update_op =\
            build(regime = 'val',
                  epochs = 1,
                  batch_size = batch_size,
                  learning_rate = learning_rate,
                  wd = wd,
                  buffer_size = buffer_size)
            
    
    sess_config = tf.ConfigProto(allow_soft_placement = True,
                                 log_device_placement = False)
    
    chkpt = tf.train.get_checkpoint_state(save_dir)
    
    num_steps = dataset_len // batch_size + 1 if \
        dataset_len % batch_size else \
            dataset_len // batch_size
    
    
    with tf.Session(graph = graph, config = sess_config) as sess:
        
        writer = tf.summary.FileWriter(LOG_DIR,
                                       graph = graph)
        
        if chkpt:
            
            current_step = chkpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            
            saver.restore(sess, chkpt.model_checkpoint_path)
            
            sess.run(running_variables_init)
                        
            for step in range(num_steps):
                 
                _, act_summaries = sess.run([metric_update_op, test_summaries_op])
                
                writer.add_summary(act_summaries, int(current_step) + step)
                
            mae_inf = sess.run(mae_merge_op)
            
            writer.add_summary(mae_inf, current_step)
            
        else:
            
            print('No checkpoints found')
            