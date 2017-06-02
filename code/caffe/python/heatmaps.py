import caffe

import numpy as np
from scipy.misc import imresize
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import transform, filters
import sys, pylab, os, urllib, getopt
from collections import OrderedDict

from helpers import *
from defaults import caffe_dir

def visualize_heatmaps(net, img_path, mask_path, label, ann_path = None, show_titles = True, show_bbs = True, mask_alpha = 1.5, bb_method = 'mean', mask_flip = True, thres_first = True, fig_path = None, gpu = None,
                       synsets = np.loadtxt('/home/ruthfong/packages/caffe/data/ilsvrc12/synsets.txt', str, delimiter='\t'),
                       indexing = np.loadtxt('/home/ruthfong/packages/caffe/data/ilsvrc12/ascii_order_to_synset_order.txt')):
    pylab.rcParams['figure.figsize'] = (10.0, 4.0)
    f, ax = plt.subplots(1,5)
    
    results = {}
    
    ax[0].imshow(caffe.io.load_image(img_path))
    if show_titles:
        ax[0].set_title('orig img + gt bb')
    ax[0].set_ylabel(get_short_class_name(label))
    ax[0].get_xaxis().set_visible(False)
    ax[0].get_yaxis().set_ticks([])
    if ann_path != None:
        objs = load_objs(ann_path)
        results['ground_truth'] = objs
        for k in objs.keys():
            for i in range(len(objs[k])):
                bb_coords = objs[k][i]
                rect = patches.Rectangle((bb_coords[0],bb_coords[1]),bb_coords[2]-bb_coords[0],bb_coords[3]-bb_coords[1],
                             linewidth=1,edgecolor='r',facecolor='none')
                ax[0].add_patch(rect)

    transformer = get_ILSVRC_net_transformer(net)
    resize = caffe.io.load_image(img_path).shape[:2]
    heatmap = np.load(mask_path)
    if mask_flip:
        heatmap = 1 - heatmap
    
    if show_bbs:
        bb_coords = getbb_from_heatmap(heatmap = heatmap, alpha = mask_alpha, method = bb_method, 
                                        resize = resize, thres_first = thres_first)
    ax[1].imshow(imresize(heatmap, resize))
    if show_bbs and ann_path != None:
        rect = patches.Rectangle((bb_coords[0],bb_coords[1]),bb_coords[2]-bb_coords[0],bb_coords[3]-bb_coords[1],
                             linewidth=1,edgecolor='r',facecolor='none')
        ax[1].add_patch(rect)
        overlap = max(compute_overlap(bb_coords, objs, synsets[label]))
        if show_titles:
            ax[1].set_title('mask (%.2f)' % overlap)
        results['mask'] = {'bb': bb_coords, 'overlap': overlap}
    else:
        if show_titles:
            ax[1].set_title('mask')
    ax[1].get_xaxis().set_visible(False)
    ax[1].get_yaxis().set_visible(False)

    options = OrderedDict()
    options['saliency'] = {'alpha': 5.0, 'top_name': 'loss3/classifier', 'bottom_name': 'data', 'norm_deg': np.inf}
    options['guided_backprop'] = {'alpha': 4.5, 'top_name': 'loss3/classifier', 'bottom_name': 'data', 'norm_deg': np.inf}
    #options['excitation_backprop'] = {'alpha': 1.5, 'top_name': 'loss3/classifier', 'bottom_name': 'pool2/3x3_s2', 'norm_deg': -1}
    options['contrast_excitation_backprop'] = {'alpha': 1.5, 'top_name': 'loss3/classifier', 'bottom_name': 'pool2/3x3_s2', 'norm_deg': -2}
    heatmap_dispnames = ['gradient','guided', 'contrast excitation']

    for i in range(len(options.keys())):
        heatmap_type = options.keys()[i]
        top_name = options[heatmap_type]['top_name']
        bottom_name = options[heatmap_type]['bottom_name']
        norm_deg = options[heatmap_type]['norm_deg']
        alpha = options[heatmap_type]['alpha']
        
        heatmap = compute_heatmap(net = net, transformer = transformer, paths = img_path, labels = label,
                                   heatmap_type = heatmap_type, topBlobName = top_name, topLayerName = top_name,
                                   outputBlobName = bottom_name, outputLayerName = bottom_name, norm_deg = norm_deg,
                                   gpu = gpu)
        resize = caffe.io.load_image(img_path).shape[:2]
        if show_bbs:
            bb_coords = getbb_from_heatmap(heatmap = heatmap, alpha = alpha, method = bb_method, resize = resize, thres_first = thres_first)
        
        ax[i+2].imshow(imresize(heatmap, resize))

        if show_bbs:
            rect = patches.Rectangle((bb_coords[0],bb_coords[1]),bb_coords[2]-bb_coords[0],bb_coords[3]-bb_coords[1],
                                     linewidth=1,edgecolor='r',facecolor='none')
            ax[i+2].add_patch(rect)
            overlap = max(compute_overlap(bb_coords, objs, synsets[label]))
            if show_titles:
                ax[i+2].set_title('%s (%.2f)' % (heatmap_type, overlap))
            results[heatmap_type] = {'bb': bb_coords, 'overlap': overlap}
        else:
            if show_titles:
                ax[i+2].set_title('%s' % heatmap_dispnames[i])
        ax[i+2].get_xaxis().set_visible(False)
        ax[i+2].get_yaxis().set_visible(False)

    f.subplots_adjust(wspace=0.05,hspace=0)
    #plt.tight_layout()

    if fig_path != None:
        plt.savefig(fig_path, bbox_inches='tight', pad_inches=0.0)

    if show_bbs:
        return results
 

def overlay_map(img, heatmap, overlay = True, interp = 'bicubic'):
    # normalize heatmap to be between [0,1]
    heatmap -= heatmap.min()
    if heatmap.max() > 0:
        heatmap /= heatmap.max()
    
    # resize heatmap
    #heatmap = transform.resize(heatmap, (img.shape[:2]), order = 3, mode = 'nearest')
    heatmap = imresize(heatmap, (img.shape[:2]), interp=interp)
    cmap = plt.get_cmap('jet')
    heatmapV = cmap(heatmap)
    heatmapV = np.delete(heatmapV, 3, 2)
    if overlay:
        heatmap = (1*(1-heatmap**0.8).reshape(heatmap.shape + (1,))*img 
            + (heatmap**0.8).reshape(heatmap.shape + (1,))*heatmapV)
    else:
        heatmap = heatmapV
    return heatmap


def localization_with_masks(net, img_paths, labels, mask_paths, alpha, out_path, norm_deg = np.inf, bb_method = 'mean', mask_flip = True, thres_first = True, batch_size = 50, indexing = np.loadtxt('/home/ruthfong/packages/caffe/data/ilsvrc12/ascii_order_to_synset_order.txt')):
    if not isinstance(img_paths, np.ndarray):
        img_paths = np.array(img_paths)
    if not isinstance(labels, np.ndarray):
        labels = np.array(labels)
    if not isinstance(mask_paths, np.ndarray):
        mask_paths = np.array(mask_paths)
 
    transformer = get_ILSVRC_net_transformer(net)
    num_imgs = len(img_paths)
    assert(num_imgs == len(labels))
    assert(num_imgs == len(mask_paths))
    for i in range(int(np.ceil(num_imgs/float(batch_size)))):
        if (i+1)*batch_size < num_imgs:
            idx = range(i*batch_size, (i+1)*batch_size)
        else:
            idx = range(i*batch_size, num_imgs)
        
        bb_coords = np.empty((len(idx), 4), dtype=int)
        for j in range(len(idx)):
            resize = caffe.io.load_image(img_paths[idx[j]]).shape[:2]
            heatmap = np.load(mask_paths[idx[j]])
            if mask_flip:
                heatmap = 1 - heatmap
            bb_coords[j][...] = getbb_from_heatmap(heatmap = heatmap, alpha = alpha, method = bb_method, 
                                                   resize = resize, thres_first = thres_first)

        print 'writing to %s (%d-%d out of %d)' % (out_path, idx[0], idx[-1], num_imgs)
        directory = os.path.dirname(out_path)
        if not os.path.isdir(directory):
            os.makedirs(directory)
        f = open(out_path, 'a')
        output = np.hstack((np.reshape(indexing[labels[idx]], (-1, 1)), bb_coords))
        np.savetxt(f, output, fmt = '%d %d %d %d %d')
        f.close()

    
def compute_heatmap(net, transformer, paths, labels, heatmap_type, topBlobName, topLayerName,
                    outputBlobName = 'data', outputLayerName = 'data', secondTopBlobName = 'pool5/7x7_s1', 
                    secondTopLayerName = 'pool5/7x7_s1', norm_deg = np.inf, gpu = None):
    if gpu == None:
        if heatmap_type == 'saliency' or heatmap_type == 'grad_cam':
            caffe.set_mode_cpu()
        elif heatmap_type == 'guided_backprop':
            caffe.set_mode_dc_cpu()
        elif heatmap_type == 'excitation_backprop' or heatmap_type == 'contrast_excitation_backprop':
            caffe.set_mode_eb_cpu()
        else:
            print 'heatmap_type %s is not supported' % heatmap_type
            return
    else:
        caffe.set_device(gpu)
        if heatmap_type == 'saliency' or heatmap_type == 'grad_cam':
            caffe.set_mode_gpu()
        elif heatmap_type == 'guided_backprop':
            caffe.set_mode_dc_gpu()
        elif heatmap_type == 'excitation_backprop' or heatmap_type == 'contrast_excitation_backprop':
            caffe.set_mode_eb_gpu()
        else:
            print 'heatmap_type %s is not supported' % heatmap_type
            return
    
    if isinstance(paths, basestring):
        num_imgs = 1
        assert(isinstance(labels, int))
    else:
        num_imgs = len(paths)
        assert(num_imgs == len(labels))
        
    net.blobs['data'].reshape(num_imgs,
                              3,
                              net.blobs['data'].data.shape[2], 
                              net.blobs['data'].data.shape[3]) 

    if num_imgs == 1:
        net.blobs['data'].data[...] = transformer.preprocess('data', 
                                                        caffe.io.load_image(paths))
        net.forward()
        net.blobs[topBlobName].diff[0][...] = 0
        net.blobs[topBlobName].diff[0][labels] = 1
    else:
        for i in range(num_imgs):
            net.blobs['data'].data[i, ...] = transformer.preprocess('data', 
                                                                    caffe.io.load_image(paths[i]))
        net.forward()
        for i in range(num_imgs):
            net.blobs[topBlobName].diff[i][...] = 0
            net.blobs[topBlobName].diff[i][labels[i]] = 1
    if heatmap_type == 'contrast_excitation_backprop':
        # invert top layer weights
        net.params[topLayerName][0].data[...] *= -1
        out = net.backward(start = topLayerName, end = secondTopLayerName)
        buff = net.blobs[secondTopBlobName].diff.copy()

        # invert back
        net.params[topLayerName][0].data[...] *= -1
        out = net.backward(start = topLayerName, end = secondTopLayerName)

        # compute the contrastive signal
        net.blobs[secondTopBlobName].diff[...] -= buff
        net.backward(start = secondTopLayerName, end = outputLayerName)
    elif heatmap_type == 'grad_cam':
        net.backward(start = topLayerName, end = outputLayerName)
        activations = net.blobs[outputBlobName].data.copy() # TODO: check if copy is needed
        gradient = net.blobs[outputBlobName].diff.copy()
        alphas = np.mean(np.mean(gradient,3),2)
        attMaps = np.squeeze(np.maximum(np.sum(activations * np.broadcast_to(
            np.expand_dims(np.expand_dims(alphas, 2),3), activations.shape), 1), 0))
        return attMaps
    else:
        net.backward(start = topLayerName, end = outputLayerName)
    
    if np.isinf(norm_deg):
        if norm_deg == np.inf:
            attMaps = np.squeeze(np.abs(net.blobs[outputBlobName].diff).max(1))
        else:
            attMaps = np.squeeze(np.abs(net.blobs[outputBlobName].diff).min(1))
    else:
        if norm_deg == 0:
            attMaps = np.squeeze(net.blobs[outputBlobName.diff])
        elif norm_deg == -1:
            attMaps = np.squeeze(net.blobs[outputBlobName].diff.sum(1))
        elif norm_deg == -2:
            attMaps = np.squeeze(np.maximum(net.blobs[outputBlobName].diff.sum(1), 0))
        else:
            attMaps = np.squeeze(((np.abs(net.blobs[outputBlobName].diff)**norm_deg).sum(1))**(1/float(norm_deg)))
            # TODO: test this case        
    
    return attMaps


def getbb_from_heatmap(heatmap, alpha, method = 'mean', resize = None, thres_first = True):
    if not thres_first and resize is not None:
        heatmap = imresize(heatmap, resize)
    
    assert(method == 'mean' or method == 'min_max_diff' or method == 'energy')

    if method == 'mean':
        threshold = alpha*heatmap.mean()
        heatmap[heatmap < threshold] = 0
    elif method == 'min_max_diff':
        threshold = alpha*(heatmap.max()-heatmap.min())
        heatmap_m = heatmap - heatmap.min()
        heatmap[heatmap_m < threshold] = 0
    elif method == 'energy':
        heatmap_f = heatmap.flatten()
        sorted_idx = np.argsort(heatmap_f)[::-1]
        tot_energy = heatmap.sum()
        heatmap_cum = np.cumsum(heatmap_f[sorted_idx])
        ind = np.where(heatmap_cum >= alpha*tot_energy)[0][0]
        heatmap_f[sorted_idx[ind:]] = 0
        heatmap = np.reshape(heatmap_f, heatmap.shape)

    if thres_first and resize is not None:
        heatmap = imresize(heatmap, resize)
    
    if (heatmap == 0).all():
        if resize is not None:
            return [1,1,resize[0],resize[1]]
        else:
            return [1,1,heatmap.shape[0],heatmap.shape[1]]
        
    x = np.where(heatmap.sum(0) > 0)[0] + 1
    y = np.where(heatmap.sum(1) > 0)[0] + 1
    return [x[0],y[0],x[-1],y[-1]]


def localization_with_heatmaps(net, paths, labels, alpha, heatmap_type, out_path, top_name, bottom_name = 'data',
                               batch_size = 200, gpu = None, norm_deg = np.inf, bb_method = 'mean', thres_first = True,
                               indexing = np.loadtxt('/home/ruthfong/packages/caffe/data/ilsvrc12/ascii_order_to_synset_order.txt')):
    if not isinstance(paths, np.ndarray):
        paths = np.array(paths)
    if not isinstance(labels, np.ndarray):
        labels = np.array(labels)
    transformer = get_ILSVRC_net_transformer(net)
    num_imgs = len(paths)
    assert(num_imgs == len(labels))
    for i in range(int(np.ceil(num_imgs/float(batch_size)))):
        if (i+1)*batch_size < num_imgs:
            idx = range(i*batch_size, (i+1)*batch_size)
        else:
            idx = range(i*batch_size, num_imgs)
        heatmaps = compute_heatmap(net = net, transformer = transformer, paths = paths[idx], labels = labels[idx],
                                   heatmap_type = heatmap_type, topBlobName = top_name, topLayerName = top_name, 
                                   outputBlobName = bottom_name, outputLayerName = bottom_name, norm_deg = norm_deg,
                                   gpu = gpu)
        bb_coords = np.empty((len(idx), 4), dtype=int)
        for j in range(len(idx)):
            resize = caffe.io.load_image(paths[idx[j]]).shape[:2]
            bb_coords[j][...] = getbb_from_heatmap(heatmap = heatmaps[j], alpha = alpha, method = bb_method, 
                    resize = resize, thres_first = thres_first)
        
        print 'writing to %s (%d-%d out of %d)' % (out_path, idx[0], idx[-1], num_imgs)
        directory = os.path.dirname(out_path)
        if not os.path.isdir(directory):
            os.makedirs(directory)
        f = open(out_path, 'a')
        output = np.hstack((np.reshape(indexing[labels[idx]], (-1, 1)), bb_coords))
        np.savetxt(f, output, fmt = '%d %d %d %d %d')
        f.close()


def main(argv):
    if len(argv) < 4:
        print 'Not enough arguments'
        print 'Usage: python heatmaps.py OUT_DIR/OUT_FILE DATA_DESC HEATMAP_TYPE THRESHOLD_TYPE [MASK_DIR] [GPU]'
        return
    out_path = argv[0]
    data_desc = argv[1]
    heatmap_type = argv[2]
    bb_method = argv[3]
    if heatmap_type == 'mask':
        assert(len(argv) > 4)
        mask_dir = argv[4]
        gpu = int(argv[5]) if len(argv) >= 6 else None
    else:
        gpu = int(argv[4]) if len(argv) >= 5 else None
    
    batch_size = 50 
    net_type = 'googlenet'
    #data_desc = 'val'
    #heatmap_type = 'excitation_backprop'
    #bb_method = 'mean'

    assert(data_desc == 'annotated_train_heldout' or data_desc == 'val' or data_desc == 'animal_parts')
    assert(heatmap_type == 'mask' or heatmap_type == 'saliency' or heatmap_type == 'guided_backprop' or heatmap_type == 'excitation_backprop')
    assert(bb_method == 'mean' or bb_method == 'min_max_diff' or bb_method == 'energy')
    
    if gpu is not None:
        caffe.set_device(gpu)
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()

    net = get_net(net_type)

    if data_desc == 'annotated_train_heldout':
        (paths, labels) = read_imdb('../../../data/ilsvrc12/annotated_train_heldout_imdb.txt')
        num_imgs = len(labels)
        if bb_method == 'mean':
            alphas = np.arange(0,10.5,0.5)
        elif bb_method == 'min_max_diff':
            alphas = np.arange(0,1,0.05)
        elif bb_method == 'energy':
            alphas = np.arange(0,1,0.05)
        out_dir = out_path
    elif data_desc == 'val':
        (paths, labels) = read_imdb('../../../data/ilsvrc12/val_imdb.txt')
        num_imgs = len(labels)
        data_desc = 'val'
    elif data_desc == 'animal_parts':
        (paths, labels) = read_imdb('../../../data/ilsvrc12/animal_parts_require_both_min_10_imdb.txt')
        num_imgs = len(labels)

    if data_desc != 'annotated_train_heldout':
        if heatmap_type == 'mask':
            if bb_method == 'mean':
                #alphas = [1.0] # target class setting
                alphas = [0.5] # top5 setting
            elif bb_method == 'min_max_diff':
                alphas = [0.10]
            elif bb_method == 'energy':
                alphas = [0.95]
        elif heatmap_type == 'saliency':
            if bb_method == 'mean':
                alphas = [5.0]
            elif bb_method == 'min_max_diff':
                alphas = [0.25]
            elif bb_method == 'energy':
                alphas = [0.10]
        elif heatmap_type == 'guided_backprop':
            if bb_method == 'mean':
                alphas = [4.5]
            elif bb_method == 'min_max_diff':
                alphas = [0.05]
            elif bb_method == 'energy':
                alphas = [0.30]
        elif heatmap_type == 'excitation_backprop':
            if bb_method == 'mean':
                alphas = [1.5]
            elif bb_method == 'min_max_diff':
                alphas = [0.15]
            elif bb_method == 'energy':
                alphas = [0.60]

    labels_desc = np.loadtxt(os.path.join(caffe_dir, 'data/ilsvrc12/synset_words.txt'), str, delimiter='\t')

    if type(paths) is not np.ndarray:
        paths = np.array(paths)
    if type(labels) is not np.ndarray:
        labels = np.array(labels)

    if heatmap_type == 'mask':
        mask_paths = [os.path.join(mask_dir, '%d.npy' % x) for x in range(num_imgs)]

        for i in range(len(alphas)):
            alpha = alphas[i]
            if data_desc == 'annotated_train_heldout':
                out_path = os.path.join(out_dir, '%s_%s_alpha_%.2f_norm_Inf.txt' % (heatmap_type, bb_method, alpha))
            if os.path.exists(out_path):
                print 'skipping %s because it exists' % out_path
                continue
            localization_with_masks(net, paths[:num_imgs], labels[:num_imgs], mask_paths[:num_imgs], alpha, out_path, 
                                    norm_deg = np.inf, bb_method = bb_method, mask_flip = True, thres_first = True, batch_size = batch_size, 
                                    indexing = np.loadtxt('../../../data/ilsvrc12/ascii_order_to_synset_order.txt'))
    else:
        if net_type == 'googlenet':
            top_name = 'loss3/classifier'
        else:
            assert(False)

        if heatmap_type == 'excitation_backprop':
            if net_type == 'googlenet':
                bottom_name = 'pool2/3x3_s2'
            else:
                assert(False)
            norm_deg = -1
            norm_desc = '%d' % norm_deg
        else:
            bottom_name = 'data'
            norm_deg = np.inf
            norm_desc = 'Inf'

        for i in range(len(alphas)):
            alpha = alphas[i]
            if data_desc == 'annotated_train_heldout':
                out_path = os.path.join(out_dir, '%s_%s_alpha_%.2f_norm_%s.txt' % (heatmap_type, bb_method, alpha, norm_desc))
            localization_with_heatmaps(net, paths, labels, alpha, heatmap_type, out_path, top_name, bottom_name = bottom_name,
                               batch_size = batch_size, gpu = gpu, norm_deg = norm_deg, bb_method = bb_method, thres_first = True,
                               indexing = np.loadtxt('/home/ruthfong/packages/caffe/data/ilsvrc12/ascii_order_to_synset_order.txt'))


if __name__ == '__main__':
    main(sys.argv[1:])
    #main()
