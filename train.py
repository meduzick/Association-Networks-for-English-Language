# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 12:08:55 2020

@author: nikit


"""

#C:\Users\nikit\anaconda3\Library\bin - add this to avoid dll errors


import tensorflow as tf
from build import build
from tqdm import tqdm
from os.path import join

from eval import eval

flags = tf.app.flags

FLAGS = flags.FLAGS

flags.DEFINE_integer('epochs', 1, 'number of epochs to train')

flags.DEFINE_integer('batch_size', 100, 'size of the batch')

flags.DEFINE_float('learning_rate', 3e-04, 'learning rate')

flags.DEFINE_integer('check_performance_steps', 100, 'check perf every n steps')

flags.DEFINE_integer('shuffle_buffer', 100_000, 'shuffle bufer size')


WD = None


LOG_DIR = join('tb_logs', 'train')

SAVE_DIR = 'model_chkpts'


def train():
    
    graph = tf.Graph()
    
    
    print(80 * "=")
    print("BUILDING COMPUTATIONAL GRAPH...")
    print(80 * "=")
    
    
    with graph.as_default():
        
        global_step, train_step, merge_op, test_summaries_op, \
        saver, dataset_len, mae_merge_op, running_variables_init, \
            metric_update_op =\
            build(regime = 'train',
                  epochs = FLAGS.epochs,
                  batch_size = FLAGS.batch_size,
                  learning_rate = FLAGS.learning_rate,
                  wd = WD,
                  buffer_size = FLAGS.shuffle_buffer)
                
    
    
    num_steps = (dataset_len * FLAGS.epochs) // FLAGS.batch_size + 1 if \
        (dataset_len * FLAGS.epochs) % FLAGS.batch_size else \
            (dataset_len * FLAGS.epochs) // FLAGS.batch_size
            
                
            
    sess_config = tf.ConfigProto(allow_soft_placement = True,
                                 log_device_placement = False)
    
    
    with tf.Session(graph = graph, config = sess_config) as sess:
        
        writer = tf.summary.FileWriter(LOG_DIR,
                                       graph = graph)
        
        sess.run([running_variables_init, tf.global_variables_initializer()])
        
        print(80 * "=")
        print("TRAINING")
        print(80 * "=") 
            
        with tqdm(total = num_steps) as prog:
            
            for step in range(num_steps):
                
                if step % 100 == 0:
                    
                    glb_stp, _, mrg_inf = sess.run([global_step,
                                                    train_step,
                                                    merge_op])
                    
                    writer.add_summary(mrg_inf, glb_stp)
                    
                else:
                    
                    glb_stp, _ = sess.run([global_step,
                                           train_step])
                    
                if step % FLAGS.check_performance_steps == 0:
                    
                    saver.save(sess = sess,
                               save_path = join(SAVE_DIR, 'model'),
                               global_step = global_step)
                    
                    eval(FLAGS.batch_size,
                         FLAGS.learning_rate,
                         SAVE_DIR,
                         WD,
                         FLAGS.shuffle_buffer)
                    
                prog.update(1)
                    

if __name__ == '__main__':
    
    train()