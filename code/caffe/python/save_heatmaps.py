import caffe

import numpy as np
from scipy.misc import imresize
import sys, os, urllib, argparse, time

from helpers import *
from heatmaps import compute_heatmap

def main(argv):
    parser = argparse.ArgumentParser(description='Save numpy files of heatmaps (use default settings).') # TODO make default settings a boolean flag

    parser.add_argument('dataset', default='imagenet', type=str, help="choose from ['imagenet', 'voc2007', 'COCO']")
    parser.add_argument('split', default='val', type=str, help="choose from ['train', 'train_heldout', 'val', 'test']")
    parser.add_argument('heatmap', default='saliency', type=str, 
    	help="choose from ['saliency', 'guided_backprop', 'excitation_backprop', 'contrast_excitation_backprop', 'grad_cam'")
    parser.add_argument('-r', '--results_dir', default=None, type=str, help="directory to save heatmaps")
    parser.add_argument('-g', '--gpu', default=None, type=int, help="zero-indexed gpu to use [i.e. 0-3]")
    parser.add_argument('-b', '--batch_size', default=64, type=int, help="batch size")
    #parser.add_argument('-t', '--top_name', default='loss3/classifier', type=str, help="name of the top layer")
    #parser.add_argument('-b', '--bottom_name', default='data', type=str, help="name of the bottom layer")
    #parser.add_argument('-n', '--norm_deg', default=np.inf, type=int)
    parser.add_argument('-a', '--start', default=0, type=int, help="start index")
    parser.add_argument('-z', '--end', default=None, type=int, help="end index")

    args = parser.parse_args(argv)
    dataset = args.dataset
    split = args.split
    heatmap_type = args.heatmap
    results_dir = args.results_dir
    gpu = args.gpu
    batch_size = args.batch_size
    #top_name = args.top_name
    start = args.start
    end = args.end

    if gpu is None:
        caffe.set_mode_cpu()
    else:
        caffe.set_device(gpu)
        caffe.set_mode_gpu()

    if dataset == 'imagenet':
        net = get_net('googlenet')
        top_name = 'loss3/classifier'
        labels_desc = np.loadtxt('/home/ruthfong/packages/caffe/data/ilsvrc12/synset_words.txt', str, delimiter='\t')
        #synsets = np.loadtxt('/home/ruthfong/packages/caffe/data/ilsvrc12/synsets.txt', str, delimiter='\t')
        transformer = get_ILSVRC_net_transformer(net)
        if split == 'train_heldout':
            (paths, labels) = read_imdb('/home/ruthfong/packages/caffe/data/ilsvrc12/annotated_train_heldout_imdb.txt')
        elif split == 'val':
            (paths, labels) = read_imdb('/home/ruthfong/packages/caffe/data/ilsvrc12/val_imdb.txt')
        elif split == 'animal_parts':
            (paths, labels) = read_imdb('/home/ruthfong/packages/caffe/data/ilsvrc12/animal_parts_require_both_min_10_imdb.txt')
        else:
            print '%s is not supported' % split
        paths = np.array(paths)
        labels = np.array(labels)
        #ann_dir = '/data/ruthfong/ILSVRC2012/annotated_train_heldout_ground_truth_annotations'
        #ann_paths = [os.path.join(ann_dir, f) for f in os.listdir(ann_dir)]
    elif dataset == 'voc2007':
        net = get_net('googlenet_voc')
        top_name = 'loss3/classifier-ft'

        voc_dir = '/data/ruthfong/VOCdevkit/VOC2007/'
        labels_desc = voc_labels_desc
        transformer = get_VOC_net_transformer(net)
        (paths, labels) = read_imdb(os.path.join(voc_dir, 'caffe/%s.txt' % split))
        #ann_dir = os.path.join(voc_dir, 'Annotations')
        #ann_paths = np.array([os.path.join(ann_dir, f.strip('.jpg') + '.xml') for f in paths])
        paths = np.array([os.path.join(voc_dir, 'JPEGImages', f) for f in paths])
    else:
    	assert(False)

    if results_dir is not None and not os.path.exists(results_dir):
        os.makedirs(results_dir)

    if end is None:
        end = len(paths)

    if heatmap == 'excitation_backprop':
        norm_deg = -1
        bottom_name = 'pool2/3x3_s2'
    elif heatmap == 'contrast_excitation_backprop':
        norm_deg = -2
        bottom_name = 'pool2/3x3_s2'
    elif heatmap == 'grad_cam':
    	norm_deg = None
    	bottom_name = 'inception_4e/output' 
    else:
        norm_deg = np.inf
        bottom_name = 'data'
    
    img_idx = range(start, end)
    num_imgs = len(img_idx)
    num_batches = int(np.ceil(num_imgs/float(batch_size)))
    for i in range(num_batches):
    	start_time = time.time()
        if (i+1)*batch_size < num_imgs:
            idx = img_idx[range(i*batch_size, (i+1)*batch_size)]
        else:
            idx = img_idx[range(i*batch_size, num_imgs)]
        out_file = os.path.join(results_dir, '%d.npy' % idx[-1])
        if os.path.exists(out_file):
            print '%s already exists; skipping batch from %d to %d' % (out_file, idx[0], idx[-1])
            continue
        heatmaps = compute_heatmap(net, transformer, paths[idx], labels[idx], heatmap_type, top_name, top_name,
                    outputBlobName = bottom_name, outputLayerName = bottom_name, norm_deg = norm_deg, gpu = gpu)
        for j in range(len(idx)):
        	out_file = os.path.join(results_dir, '%d.npy' % idx[j])
        	np.save(out_file, heatmaps[j])
        print 'gpu %d - batch %d/%d complete [%d-%d] (time: %.4f s)' % (gpu if gpu is not None else -1, i, num_batches, 
        	idx[0], idx[-1], time.time() - start_time)

if __name__ == '__main__':
	main(sys.argv[1:])