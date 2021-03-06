import sys
sys.path.insert(0, "../mxnet/python")

import mxnet as mx
import numpy as np
import cv2
import os
import argparse
import logging

import data_processing
import dcnn

parser = argparse.ArgumentParser(description='sketch inversion')

parser.add_argument('--gpu', type=int, default=1,
                    help='which gpu card to use, -1 means using cpu')
parser.add_argument('--long-edge', type=int, default=96,
                    help='height of the image')

args = parser.parse_args()
def do_processing(path_to_file):

    # Choose which CPU or GPU card to use
    dev = mx.gpu(args.gpu) if args.gpu >= 0 else mx.cpu()
    ctx = dev

    # Params
    long_edge = args.long_edge

 
    # load data
    img = cv2.imread(path_to_file)
    img_path="/home/jo/mainproject/sketch/data/sk/images1.jpg"
    
    # Init sketch
    sketch_np = data_processing.PreprocessSketchImage(img_path, long_edge)
    print("Colored")
    # sketch_np = data_processing.PreprocessSketchImage(dir_sketch + file_sketch[training], long_edge)
    logging.info("load the sketch image, size = %s", sketch_np.shape[2:])
    dshape = sketch_np.shape
    clip_norm = 0.05 * np.prod(dshape)


    # Load pretrained params
    gens = dcnn.get_module("g0", dshape, ctx)
    gens.load_params("model/0020-0000088-sketch.params")

    #Testing
    
    sketch_data = []
    sketch_np = data_processing.PreprocessSketchImage(img, long_edge)
    # mySketch = mx.nd.array(sketch_np)
    # sketch_data.append(mySketch)
 
    gens.forward(mx.io.DataBatch([sketch_data[-1]], [0]), is_train=False)
    new_img = gens.get_outputs()[0]
    # output = data_processing.SaveImage(new_img.asnumpy(),"output/test/out_%s" % file_sketch[training+idx])    
    output = data_processing.SaveImage(new_img.asnumpy(),"output/test/out_%s")
    cv2.imwrite("output.jpg",output)

