import numpy as np
from scipy.misc import imresize
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import transform, filters
import sys, pylab, os, urllib

caffe_root = '~/sample_code/Caffe_ExcitationBP/'
sys.path.insert(0, os.path.join(caffe_root, 'python'))
import caffe

sys.path.insert(0, '~/packages/caffe/python')
from helpers import *

from heatmaps import compute_heatmap, overlay_map


def repro_fig_3(gpu = None, interp = 'nearest'):
    net = caffe.Net('/home/ruthfong/packages/caffe/models/vgg16/VGG_ILSVRC_16_layers_deploy_force_backward.prototxt', 
                   '/home/ruthfong/packages/caffe/models/vgg16/VGG_ILSVRC_16_layers.caffemodel',
                    caffe.TEST)
    transformer = get_ILSVRC_net_transformer(net)
    
    topName = 'fc8'
    bottomNames = ['pool5', 'pool4', 'pool3', 'pool2', 'pool1']
    tabby_i = 281
    
    #img_path = '/home/ruthfong/packages/caffe/examples/images/cat.jpg'
    img_path = '/home/ruthfong/neural_coding/images/tabby_cat_cropped.jpg'
    img = caffe.io.load_image(img_path)
    
    pylab.rcParams['figure.figsize'] = (12.0, 12.0)
    
    f, ax = plt.subplots(1, len(bottomNames)+1)
    ax[0].imshow(img)
    
    for i in range(len(bottomNames)):
        heatmap = compute_heatmap(net = net, transformer = transformer, paths = img_path, 
                                 labels = tabby_i, heatmap_type = 'excitation_backprop', 
                                 topBlobName = topName, topLayerName = topName, 
                                 outputBlobName = bottomNames[i], outputLayerName = bottomNames[i],
                                 gpu = gpu)
        ax[i+1].imshow(overlay_map(img, heatmap, overlay = False, interp = interp), 
                       interpolation = interp)
    

def repro_fig_4(gpu = None, interp = 'bicubic'):
    net = caffe.Net('/home/ruthfong/packages/caffe/models/bvlc_googlenet/deploy_force_backward.prototxt',
                   '/home/ruthfong/packages/caffe/models/bvlc_googlenet/bvlc_googlenet.caffemodel',
                    caffe.TEST)
    topName = 'loss3/classifier'
    bottomName = 'pool2/3x3_s2'
    zebra_i = 340
    elephant_i = 386 # African elephant; Indian elephant = 385
    transformer = get_ILSVRC_net_transformer(net)
    img_path = '/home/ruthfong/neural_coding/fnn_images/zeb-ele1.jpg'
    zebra_map = compute_heatmap(net = net, transformer = transformer, paths = img_path, 
                                labels = zebra_i, heatmap_type = 'excitation_backprop', 
                                topBlobName = topName, topLayerName = topName,
                                outputBlobName = bottomName, outputLayerName = bottomName, 
                                gpu = gpu)
    elephant_map = compute_heatmap(net = net, transformer = transformer, paths = img_path, 
                            labels = elephant_i, heatmap_type = 'excitation_backprop', 
                            topBlobName = topName, topLayerName = topName,
                            outputBlobName = bottomName, outputLayerName = bottomName, 
                            gpu = gpu)
    img = caffe.io.load_image(img_path)
    
    pylab.rcParams['figure.figsize'] = (12.0, 12.0)
    f, ax = plt.subplots(1, 3)
    ax[0].imshow(img)
    ax[1].imshow(overlay_map(img, zebra_map, overlay = False, interp = interp), 
                 interpolation = interp)
    #ax[1].set_title('zebra')
    ax[2].imshow(overlay_map(img, elephant_map, overlay = False, interp = interp), 
                 interpolation = interp)
    #ax[2].set_title('elephant')
