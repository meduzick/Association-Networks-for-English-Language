# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 12:15:37 2020

@author: nikit
"""


import tensorflow as tf
from build import build
import numpy as np
from sklearn.metrics import r2_score
from tqdm import tqdm

from train import SAVE_DIR


def test() -> None:
    
    graph = tf.Graph()
    
    
    print(80 * "=")
    print("BUILDING COMPUTATIONAL GRAPH...")
    print(80 * "=")
    
    
    with graph.as_default():
        
        global_step, train_step, merge_op, test_summaries_op, \
        saver, dataset_len, mae_merge_op, running_variables_init, \
        metric_update_op =\
            build(regime = 'test',
                  epochs = 1,
                  batch_size = 1000,
                  learning_rate = 3e-04,
                  wd = None,
                  buffer_size = 1)
            
    
    sess_config = tf.ConfigProto(allow_soft_placement = True,
                                 log_device_placement = False)
    
    chkpt = tf.train.get_checkpoint_state(SAVE_DIR)
    
    num_steps = dataset_len // 1000 + 1 if \
        dataset_len % 1000 else \
            dataset_len // 1000
            
    logit_op = graph.get_tensor_by_name('dense_body/logits:0')
    
    target_op = graph.get_tensor_by_name('inputs/IteratorGetNext:3')
    
    logits = []
    
    targets = []
    
    
    with tf.Session(graph = graph, config = sess_config) as sess:
        
        if chkpt:
            
            current_step = chkpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            
            print(80 * "=")
            print("RESTORING FROM {} STEP".format(current_step))
            print(80 * "=")
            
            saver.restore(sess, chkpt.model_checkpoint_path)
            
            sess.run(running_variables_init)
                        
            with tqdm(total = num_steps) as prog:
                
                for step in range(num_steps):
                     
                    _, current_logits, current_target = sess.run([metric_update_op, logit_op, target_op])
                    
                    logits.append(current_logits)
                    
                    targets.append(current_target)
                    
                    prog.update(1)
                
                
            mae_inf = sess.run(mae_merge_op)
            
            
        else:
            
            print('No checkpoints found')
            
    all_logits = np.concatenate(logits)
    
    all_target = np.concatenate(targets)
    
    r2 = r2_score(all_target, all_logits)
    
    return tf.Summary.FromString(mae_inf).value[0].simple_value, r2, all_logits, all_target


if __name__ == '__main__':
    
    test()
            