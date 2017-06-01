import caffe

import numpy as np
from scipy.misc import imresize
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import transform, filters
import sys, pylab, os, urllib, getopt, argparse, time
from collections import OrderedDict

from helpers import *
from defaults import voc_labels_desc 
from heatmaps import compute_heatmap

def get_maximum_from_heatmap(heatmap, resize = None):
    if resize is not None:
        heatmap = imresize(heatmap, resize)
    
    max_coords = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    return (max_coords[1], max_coords[0])


def play_pointing_game(net, transformer, paths, labels, ann_paths, heatmap_type, labels_desc, 
                       top_name = 'loss3/classifier-ft', bottom_name = 'data', norm_deg = np.inf, batch_size = 64, 
                       gpu = None):
    num_imgs = len(paths)
    assert(num_imgs == len(labels))
    assert(num_imgs == len(ann_paths))
    num_classes = len(labels_desc)
    num_hits = np.zeros(num_classes)
    num_total = np.array([np.sum(labels == i) for i in range(num_classes)])
    num_diff_hits = np.zeros(num_classes)
    num_diff_total = np.zeros(num_classes)
    num_batches = int(np.ceil(num_imgs/float(batch_size)))
    print heatmap_type
    for i in range(num_batches):
        start = time.time()
        if (i+1)*batch_size < num_imgs:
            idx = range(i*batch_size, (i+1)*batch_size)
        else:
            idx = range(i*batch_size, num_imgs)

        if heatmap_type != 'center':
            heatmaps = compute_heatmap(net = net, transformer = transformer, paths = paths[idx], labels = labels[idx],
                                       heatmap_type = heatmap_type, topBlobName = top_name, topLayerName = top_name,
                                       outputBlobName = bottom_name, outputLayerName = bottom_name, norm_deg = norm_deg,
                                       gpu = gpu)
        for j in range(len(idx)):
            c = labels[idx[j]]
            resize = caffe.io.load_image(paths[idx[j]]).shape[:2]
            if heatmap_type == 'center':
                max_coords = (resize[1]/float(2), resize[0]/float(2))
                #max_coords (resize[1]/2, resize[0],2)
            else:
                max_coords = get_maximum_from_heatmap(heatmaps[j], resize = resize)
            #print max_coords
            objs = load_objs(ann_paths[idx[j]])
            target_objs = objs[labels_desc[labels[idx[j]]]]
            is_hit = False
            exists_distractor = len(np.unique(objs.keys())) > 1
            bb_area = 0
            for k in range(len(target_objs)):
                bb_coords = target_objs[k]
                is_hit = is_hit or (bb_coords[0] <= max_coords[0] and bb_coords[1] <= max_coords[1] 
                          and bb_coords[2] >= max_coords[0] and bb_coords[3] >= max_coords[1])
                #print bb_coords, is_hit
                bb_area += (bb_coords[2]-bb_coords[0])*(bb_coords[3]-bb_coords[1])
                if is_hit and not exists_distractor:
                    break
            is_diff = exists_distractor and bb_area < 0.25*np.prod(resize)
            if is_hit:
                num_hits[c] += 1
                num_diff_hits[c] += 1 if is_diff else 0
            num_diff_total[c] += 1 if is_diff else 0
        print '%d/%d: %.4f' % (i, num_batches, time.time() - start)

    accs = np.true_divide(num_hits, num_total)
    diff_accs = np.true_divide(num_diff_hits, num_diff_total)
    return (accs, num_hits, num_total, diff_accs, num_diff_hits, num_diff_total)

def main(argv):
    parser = argparse.ArgumentParser(description='Learn perturbation masks for ImageNet examples.')

    parser.add_argument('dataset', default='voc2007', help="choose from ['voc2007', 'COCO']")

    parser.add_argument('-g', '--gpu', default=None, type=int, help="zero-indexed gpu to use [i.e. 0-3]")
    parser.add_argument('-s', '--saliency', default='saliency', type=str, help="saliency heatmap method")

    args = parser.parse_args(argv)
    dataset = args.dataset
    gpu = args.gpu
    heatmap = args.saliency

    caffe.set_device(gpu)
    caffe.set_mode_gpu()

    net = get_net('googlenet_voc')

    voc_dir = '/data/ruthfong/VOCdevkit/VOC2007/'
    labels_desc = voc_labels_desc
    transformer = get_VOC_net_transformer(net)
    (paths, labels) = read_imdb(os.path.join(voc_dir, 'caffe/test.txt'))
    ann_dir = os.path.join(voc_dir, 'Annotations')
    ann_paths = np.array([os.path.join(ann_dir, f.strip('.jpg') + '.xml') for f in paths])
    paths = np.array([os.path.join(voc_dir, 'JPEGImages', f) for f in paths])

    if heatmap == 'excitation_backprop':
        norm_deg = -1
        bottom_name = 'pool2/3x3_s2'
    elif heatmap == 'contrast_excitation_backprop':
        norm_deg = -2
        bottom_name = 'pool2/3x3_s2'
    else:
        norm_deg = np.inf
        bottom_name = 'data'

    (accs, num_hits, num_total, diff_accs, num_diff_hits, num_diff_total) = play_pointing_game(net, transformer, paths, labels, ann_paths, 
                                                         heatmap, labels_desc, 
                               top_name = 'loss3/classifier-ft', bottom_name = bottom_name, norm_deg = norm_deg, batch_size = 64, 
                               gpu = gpu)

    print 'all'
    for i in range(len(labels_desc)):
        print labels_desc[i], accs[i]
    print ''
    print 'diff'
    for i in range(len(labels_desc)):
        print labels_desc[i], diff_accs[i] 
    print ''
    print '%s - mean acc: all=%.4f, diff=%.4f' % (heatmap, np.mean(accs), np.mean(diff_accs))
    print np.sum(num_diff_total)

if __name__ == '__main__':
    main(sys.argv[1:])
