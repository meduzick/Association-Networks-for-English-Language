# -*- coding: utf-8 -*-
"""
Created on Sun May 10 17:28:12 2020

@author: nikit
"""

import tensorflow as tf
import pickle as p
from os.path import isfile
from typing import Generator


MAX_RESP_LEN = 15


PATH_TO_CUE_TRAIN = './data/train/cue.p'

PATH_TO_RESP_TRAIN = './data/train/response.p'

PATH_TO_TARGET_TRAIN = './data/train/target.p'


PATH_TO_CUE_VAL = './data/val/cue.p'

PATH_TO_RESP_VAL = './data/val/response.p'

PATH_TO_TARGET_VAL = './data/val/target.p'


PATH_TO_CUE_TEST = './data/test/cue.p'

PATH_TO_RESP_TEST = './data/test/response.p'

PATH_TO_TARGET_TEST = './data/test/target.p'


assert isfile(PATH_TO_CUE_TRAIN), 'missed train cues'

assert isfile(PATH_TO_RESP_TRAIN), 'missed train resps'

assert isfile(PATH_TO_TARGET_TRAIN), 'missed train target'


assert isfile(PATH_TO_CUE_VAL), 'missed val cues'

assert isfile(PATH_TO_RESP_VAL), 'missed val resps'

assert isfile(PATH_TO_TARGET_VAL), 'missed val target'


assert isfile(PATH_TO_CUE_TEST), 'missed test cues'

assert isfile(PATH_TO_RESP_TEST), 'missed test resps'

assert isfile(PATH_TO_TARGET_TEST), 'missed test target'





def _get_data(regime: str):
    
    assert regime in ['train', 'val', 'test'], 'wrong regime...'
    
    
    if regime == 'train':
        
        C = p.load(open(PATH_TO_CUE_TRAIN, 'rb'))
        
        R = p.load(open(PATH_TO_RESP_TRAIN, 'rb'))
        
        T = p.load(open(PATH_TO_TARGET_TRAIN, 'rb'))
        
    if regime == 'val':
        
        C = p.load(open(PATH_TO_CUE_VAL, 'rb'))
        
        R = p.load(open(PATH_TO_RESP_VAL, 'rb'))
        
        T = p.load(open(PATH_TO_TARGET_VAL, 'rb'))
        
    if regime == 'test':
        
        C = p.load(open(PATH_TO_CUE_TEST, 'rb'))
        
        R = p.load(open(PATH_TO_RESP_TEST, 'rb'))
        
        T = p.load(open(PATH_TO_TARGET_TEST, 'rb'))
        
    
    return C, R, T


def _generator(C, R, T):
    
    for c, r, t in zip(C, R, T):
        
        yield c, r, len(r), t


def _return_batch(_generator: Generator[list, None, None],
                  buffer_size: int, epochs: int, batch_size: int) -> tf.Tensor:
    
    dataset = tf.data.Dataset.from_generator(_generator,
                                             output_types = (tf.int32, 
                                                             tf.int32, 
                                                             tf.int32,
                                                             tf.float32),
                                             output_shapes = (tf.TensorShape([]),
                                                              tf.TensorShape([None,]),
                                                              tf.TensorShape([]),
                                                              tf.TensorShape([]))
                                             )
    
    dataset = (dataset
               .shuffle(buffer_size, reshuffle_each_iteration = True)
               .repeat(epochs)
               .padded_batch(batch_size,
                             padded_shapes = (
                                 (),
                                 (MAX_RESP_LEN, ),
                                 (),
                                 ()
                                              ),
                             drop_remainder = False)
               )
    
    iterator = dataset.make_one_shot_iterator()
    
    batch = iterator.get_next()
    
    return batch


def get_input(regime: str, epochs: int, batch_size: int, buffer_size: int) -> list:
    
    C, R, T = _get_data(regime)

    batch = _return_batch(lambda: _generator(C, R, T),
                          buffer_size,
                          epochs,
                          batch_size)
    
    return batch, len(T)