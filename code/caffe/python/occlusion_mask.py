import numpy as np
#from scipy.misc import imresize
#import matplotlib.pyplot as plt
#import matplotlib.patches as patches
#from skimage import transform, filters
#import pylab
import sys, os, argparse, time
#from collections import OrderedDict

from helpers import *
from defaults import caffe_dir, voc_labels_desc

def generate_occlusion_masks(mask_size = (224,224), occ_size = 35, num_occ_per_side = 28):
    assert(mask_size[0] == mask_size[1])
    masks = np.zeros((num_occ_per_side**2, 3, mask_size[0], mask_size[1]))
    locs = np.linspace(0, mask_size[0], num_occ_per_side, False).astype(int)

    c = 0
    for i in range(num_occ_per_side):
        for j in range(num_occ_per_side):
            masks[c,:,locs[i]:locs[i]+occ_size,locs[j]:locs[j]+occ_size] = 1
            c += 1
    return masks

def compute_occlusion_heatmap(net, transformer, path, label, masks = None, occ_size = 35, num_occ_per_side = 28, 
                              batch_size = 64, top_name = 'loss3/classifier', out_file = None, gpu = None):
    if gpu == None:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
    
    img = transformer.preprocess('data', caffe.io.load_image(path))
    net.blobs['data'].data[...] = img
    net.forward()
    orig_score = net.blobs[top_name].data[0,label]
   
    if masks is None:
        masks = generate_occlusion_masks(img.shape[2:], occ_size, num_occ_per_side)
    num_masks = masks.shape[0]
    
    net.blobs['data'].reshape(batch_size, 3, net.blobs['data'].data.shape[2], 
                              net.blobs['data'].data.shape[3])
    scores = np.zeros(num_masks)
    for i in range(int(np.ceil(num_masks/float(batch_size)))):
        if (i+1)*batch_size < num_masks:
            idx = range(i*batch_size, (i+1)*batch_size)
        else:
            idx = range(i*batch_size, num_masks)
            net.blobs['data'].reshape(len(idx), 3, net.blobs['data'].data.shape[2], 
                              net.blobs['data'].data.shape[3])
        net.blobs['data'].data[...] = (1-masks[idx]) * np.broadcast_to(img, masks[idx].shape)
        net.forward()
        scores[idx] = net.blobs[top_name].data[:,label]
    
    m = np.squeeze(masks[:,0,:,:])
    
    heatmap = np.true_divide(np.sum(m * np.broadcast_to(np.expand_dims(np.expand_dims(scores, axis=1), axis=2), 
                                                          m.shape),0), np.sum(m,0)) - orig_score
    
    if out_file is not None:
        np.save(out_file, heatmap)

    return heatmap

def main(argv):
    parser = argparse.ArgumentParser(description='Learn occlusion heatmaps.')

    parser.add_argument('dataset', default='imagenet', type=str, help="choose from ['imagenet', 'voc2007', 'COCO']")
    parser.add_argument('split', default='val', type=str, help="choose from ['train', 'train_heldout', 'val', 'test']")
    parser.add_argument('-r', '--results_dir', default=None, type=str, help="directory to save occlusion masks")
    parser.add_argument('-g', '--gpu', default=None, type=int, help="zero-indexed gpu to use [i.e. 0-3]")
    parser.add_argument('-b', '--batch_size', default=64, type=int, help="batch size")
    parser.add_argument('-s', '--occ_size', default=35, type=int, help="length of occlusion square mask")
    parser.add_argument('-n', '--num_occ_per_side', default=28, type=int, help="number of occlusions per length of reshaped image")
    parser.add_argument('-t', '--top_name', default='loss3/classifier', type=str, help="name of the top layer")
    parser.add_argument('-a', '--start', default=0, type=int, help="start index")
    parser.add_argument('-z', '--end', default=None, type=int, help="end index")

    args = parser.parse_args(argv)
    dataset = args.dataset
    split = args.split
    results_dir = args.results_dir
    gpu = args.gpu
    batch_size = args.batch_size
    occ_size = args.occ_size
    num_occ_per_side = args.num_occ_per_side
    top_name = args.top_name
    start = args.start
    end = args.end

    if gpu is None:
        caffe.set_mode_cpu()
    else:
        caffe.set_device(gpu)
        caffe.set_mode_gpu()

    if dataset == 'imagenet':
        net = get_net('googlenet')
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

        voc_dir = '/data/ruthfong/VOCdevkit/VOC2007/'
        labels_desc = voc_labels_desc
        transformer = get_VOC_net_transformer(net)
        (paths, labels) = read_imdb(os.path.join(voc_dir, 'caffe/%s.txt' % split))
        #ann_dir = os.path.join(voc_dir, 'Annotations')
        #ann_paths = np.array([os.path.join(ann_dir, f.strip('.jpg') + '.xml') for f in paths])
        paths = np.array([os.path.join(voc_dir, 'JPEGImages', f) for f in paths])

    if results_dir is not None and not os.path.exists(results_dir):
        os.makedirs(results_dir)

    if end is None:
        end = len(paths)
    
    masks = generate_occlusion_masks() 
    for i in range(start, end):
        start_time = time.time()
        out_file = os.path.join(results_dir, '%d.npy' % i) if results_dir is not None else None
        if os.path.exists(out_file):
            print '%s already exists; skipping' % out_file
            continue
        _ = compute_occlusion_heatmap(net, transformer, paths[i], labels[i], masks = masks, 
                occ_size = occ_size, num_occ_per_side = num_occ_per_side,
                batch_size = batch_size, top_name = top_name, out_file = out_file, gpu = gpu)
        print 'gpu %d - %d: saved to %s (time %.4f s)' % (gpu if gpu is not None else -1, i, out_file, time.time() - start_time)

if __name__ == '__main__':
    main(sys.argv[1:])
