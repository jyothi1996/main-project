import sys
sys.path.insert(0, "../mxnet/python")

import mxnet as mx
import numpy as np
from PIL import Image
import os
import argparse
import logging
import glob
import data_processing
import dcnn
import cv2
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser(description='sketch inversion')


parser.add_argument('--long-edge', type=int, default=96,
                    help='height of the image')

args = parser.parse_args()

def do_processing(path_to_file):
    # Choose which CPU or GPU card to use
    dev = mx.cpu()
    ctx = dev

    # Params
    long_edge = args.long_edge

    # Load data
    dir_sketch = "data/sk/"
    file_sketch = os.listdir(dir_sketch)
    file_sketch.sort()
    training = int(0.7*len(file_sketch)) # 70%
    num_sketch = int(len(file_sketch) - training) # 30%
    logging.info("Num of sketches: %d" % num_sketch)
    print(dir_sketch+file_sketch[training])
    # Init sketch
    sketch_np = data_processing.PreprocessSketchImage(path_to_file, long_edge)
    logging.info("load the sketch image, size = %s", sketch_np.shape[2:])
    dshape = sketch_np.shape
    clip_norm = 0.05 * np.prod(dshape)

    # Load pretrained params
    gens = dcnn.get_module("g0", dshape, ctx)
    gens.load_params("model/0265-0011975-sketch-vgg16.params")

    # Testing 
    logging.info('Start testing arguments %s', args)
    
    # Load sketch
    sketch_data = []
    path_sketch = dir_sketch + file_sketch[training]
    sketch_np = data_processing.PreprocessSketchImage(path_to_file, long_edge)
    mySketch = mx.nd.array(sketch_np)
    sketch_data.append(mySketch)

    gens.forward(mx.io.DataBatch([sketch_data[-1]], [0]), is_train=False)
    new_img = gens.get_outputs()[0]
    #cv2.namedWindow('output', cv2.WINDOW_NORMAL)
    #new_image=new_img.asnumpy()
    #print(new_image)
    
    print(file_sketch[training])
    #cv2.destroyAllWindows()
    data_processing.SaveImage(new_img.asnumpy(),
                              "output/test/out_%s" % file_sketch[training])
    new_image=cv2.imread("output/test/out_%s" % file_sketch[training])
    cv2.imshow('img',new_image)
    cv2.waitKey(10)                    
                             
   

   
    
