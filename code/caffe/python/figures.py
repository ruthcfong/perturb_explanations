import numpy as np
from scipy.misc import imresize
import pylab
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import caffe

from helpers import *
from optimize_mask import *
from heatmaps import *

labels_desc = np.loadtxt('/home/ruthfong/packages/caffe/data/ilsvrc12/synset_words.txt', str, delimiter='\t')
default_fig_size = [6.4, 4.8]

def generate_splash_img(net, path, label, mask_path, perturbation = 'blur', fig_path = None):
    pylab.rcParams['figure.figsize'] = default_fig_size
    net_transformer = get_ILSVRC_net_transformer(net)

    target = np.zeros(1000)
    target[label] = 1

    orig_img = caffe.io.load_image(path)
    img_ = net_transformer.preprocess('data', orig_img)
    if perturbation == 'blur':
        null_img_ = net_transformer.preprocess('data', get_blurred_img(path, radius = 10))
    elif perturbation == 'mean_img':
        null_img_ = np.zeros(img_.shape)
    elif perturbation == 'random_noise':
        null_img_ = np.random.random(img_.shape)*255

    mask = np.load(mask_path)
    comp_img_ = img_ * mask + null_img_ * (1 - mask)
    comp_img_r = imresize(net_transformer.deprocess('data', comp_img_), orig_img.shape[:2])
    mask_r = imresize(mask, orig_img.shape[:2])/float(255)

    orig_score = forward_pass(net, img_, target)
    blur_score = forward_pass(net, comp_img_, target)

    f, ax = plt.subplots(1,3)
    ax[0].imshow(orig_img)
    ax[0].set_title('%s: %.4f' % (get_short_class_name(label), orig_score))
    ax[1].imshow(comp_img_r)
    ax[1].set_title('%s: %.4f' % (get_short_class_name(label), blur_score))
    ax[2].imshow(mask_r)
    ax[2].set_title('Learned Mask')
    for a in ax:
        a.get_xaxis().set_visible(False)
        a.get_yaxis().set_visible(False)
    plt.tight_layout()
    if fig_path is not None:
        plt.savefig(fig_path, bbox_inches='tight', pad_inches=0.0)

def show_saliency_map(net, img_path, label, ann_path = None, gpu = None, fig_path = None):
    pylab.rcParams['figure.figsize'] = default_fig_size
    transformer = get_ILSVRC_net_transformer(net)
    heatmap = compute_heatmap(net, transformer, img_path, label, 'saliency', 'loss3/classifier', 'loss3/classifier',
                        outputBlobName = 'data', outputLayerName = 'data', norm_deg = np.inf, gpu = gpu)
    img = caffe.io.load_image(img_path)
    f, ax = plt.subplots(1,2)
    ax[0].imshow(img)
    if ann_path is not None:
        objs = load_objs(ann_path)
        for k in objs.keys():
            for i in range(len(objs[k])):
                bb_coords = objs[k][i]
                rect = patches.Rectangle((bb_coords[0],bb_coords[1]),bb_coords[2]-bb_coords[0],bb_coords[3]-bb_coords[1],
                             linewidth=1,edgecolor='r',facecolor='none')
                ax[0].add_patch(rect)
    ax[0].set_title(get_short_class_name(label))
    ax[1].imshow(imresize(heatmap, img.shape[:2])/float(255))
    ax[1].set_title('gradient')
    for a in ax:
        a.get_xaxis().set_visible(False)
        a.get_yaxis().set_visible(False)
    #plt.tight_layout()
    f.subplots_adjust(wspace=0.05)
    if fig_path is not None:
        plt.savefig(fig_path, bbox_inches='tight', pad_inches=0.0)

def generate_perturbations_figure(transformer, img_path, mask_paths, show_masks = False, 
                                  fig_path = None):
    assert(len(mask_paths) == 3)
    orig_img = caffe.io.load_image(img_path)
    orig_img_ = transformer.preprocess('data', orig_img)
    blur_img_ = transformer.preprocess('data', blur(orig_img))
    constant_img_ = np.zeros(orig_img_.shape)
    rand_img_ = np.random.random(orig_img_.shape)*255
    null_imgs_ = [blur_img_, constant_img_, rand_img_]
    if show_masks:
        f, ax = plt.subplots(1,len(mask_paths))
    else:
        f, ax = plt.subplots(1,len(mask_paths))
    disp_name=['blur','constant','noise']
    for i in range(len(mask_paths)):
        mask = np.load(mask_paths[i])
        comp_img_r = imresize(transformer.deprocess('data', 
                                    orig_img_ * mask + null_imgs_[i] * (1 - mask)),
                                    orig_img.shape[:2])
        mask_r = imresize(mask, orig_img.shape)
        if show_masks:
            ax[i].imshow(mask_r)
            ax[i].get_xaxis().set_visible(False)
            ax[i].get_yaxis().set_visible(False)
        else:
            ax[i].imshow(comp_img_r)
            ax[i].set_title(disp_name[i])
            ax[i].get_xaxis().set_visible(False)
            ax[i].get_yaxis().set_visible(False)
            #ax[i].set_aspect('equal')
    f.subplots_adjust(wspace=0.05)
    #plt.tight_layout()
    if fig_path is not None:
        plt.savefig(fig_path, bbox_inches='tight', pad_inches=0.0)

def disp_img(img_path, resize = None, mask_path = None, hide_ticks = True, title = None, fig_path = None):
    pylab.rcParams['figure.figsize'] = default_fig_size
    f, ax = plt.subplots(1,1)
    img = caffe.io.load_image(img_path)
    if resize is not None:
        img = scipy.misc.imresize(img, resize)
    if mask_path is not None:
        mask = np.load(mask_path)
        if resize is not None and mask.shape != resize:
            mask = scipy.misc.imresize(mask, resize)
        else:
            mask = imresize(mask, img.shape[:2])
        ax.imshow(img)
        ax.imshow(mask, alpha = 0.5, cmap = 'jet')
    else:
        ax.imshow(img)
    if hide_ticks:
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    if title is not None:
        ax.set_title(title)
    #plt.tight_layout()
    f.subplots_adjust(wspace=0,hspace=0)
    if fig_path != None:
        plt.savefig(fig_path, bb_inches='tight', pad_inches=0.0)

def resized_bbs(bbs, curr_size, new_size):
    [x0, y0, x1, y1] = bbs
    mask = np.zeros(curr_size)
    mask[y0:y1,x0:x1] = 1
    new_mask = imresize(mask, new_size)/float(255)
    x = np.where(new_mask.sum(0) > 0.5)[0]+1
    y = np.where(new_mask.sum(1) > 0.5)[0]+1
    return [x[0],y[0],x[-1],y[-1]]

def mask_img(net, img_path, label, bb_coords, fig_path = None, labels_desc = np.loadtxt('/home/ruthfong/packages/caffe/data/ilsvrc12/synset_words.txt', str, delimiter='\t')):
    net_transformer = get_ILSVRC_net_transformer(net)
    orig_img = caffe.io.load_image(img_path)
    img = net_transformer.preprocess('data', orig_img)
    null_img = net_transformer.preprocess('data', get_blurred_img(img_path))
    bb_mask = np.ones(img.shape[1:])
    [x0, y0, x1, y1] = bb_coords
    bb_mask[y0:y1, x0:x1] = 0
    
    bb_masked_img = img * bb_mask + null_img * (1 - bb_mask)

    net.blobs['data'].data[...] = bb_masked_img
    net.forward()
    scores = np.squeeze(net.blobs['prob'].data)
    sorted_idx = np.argsort(scores)

    for i in range(5):
        print (i+1), labels_desc[sorted_idx[-(i+1)]], scores[sorted_idx[-(i+1)]]

    f, ax = plt.subplots(1,1)
    ax.imshow(imresize(net_transformer.deprocess('data', bb_masked_img), orig_img.shape[:2]))
    [x0, y0, x1, y1] = resized_bbs(bb_coords, bb_mask.shape, orig_img.shape[:2])
    ax.add_patch(
        patches.Rectangle(
            (x0, y0),
            x1-x0,
            y1-y0,
            linewidth=1,
            edgecolor='r',
            facecolor='none'
        )
    )
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title('%s: %.3f' % (get_short_class_name(label), scores[label]))
    f.subplots_adjust(wspace=0,hspace=0)
    if fig_path != None:
        plt.savefig(fig_path, bb_inches='tight', pad_inches=0.0)

def manual_edit_img(net, img_path, label, mask_path, objs_bbs, fig_path = None, labels_desc = labels_desc):
    num_bbs = len(objs_bbs)
    f, ax = plt.subplots(1, num_bbs+1)
    img = caffe.io.load_image(img_path)
    mask = np.load(mask_path)
    mask = imresize(mask, img.shape[:2])

    net_transformer = get_ILSVRC_net_transformer(net)
    img_ = net_transformer.preprocess('data', img)
    null_img_ = net_transformer.preprocess('data', get_blurred_img(img_path))
    
    scores = np.squeeze(forward_pass(net, img_))
    orig_score = scores[label]
    #print labels_desc[label], scores[label]
    ax[0].imshow(img)
    ax[0].imshow(mask, alpha = 0.5, cmap = 'jet')
    ax[0].set_title('Mask Overlay')
    ax[0].set_ylabel(get_short_class_name(label))

    for i in range(num_bbs):
        bb_mask = np.ones(img_.shape[1:])
        [x0, y0, x1, y1] = objs_bbs[i]
        bb_mask[y0:y1, x0:x1] = 0
        bb_masked_img = img_ * bb_mask + null_img_ * (1 - bb_mask)
        scores = np.squeeze(forward_pass(net, bb_masked_img))
        sorted_idx = np.argsort(scores)

        #for j in range(5):
        #    print (j+1), labels_desc[sorted_idx[-(j+1)]], scores[sorted_idx[-(j+1)]]

        ax[i+1].imshow(imresize(net_transformer.deprocess('data', bb_masked_img), img.shape[:2]))
        [x0, y0, x1, y1] = resized_bbs(objs_bbs[i], bb_mask.shape, img.shape[:2])
        ax[i+1].add_patch(
            patches.Rectangle(
                (x0, y0),
                x1-x0,
                y1-y0,
                linewidth=1,
                edgecolor='r',
                facecolor='none'
            )
        )
        ax[i+1].set_title('%.3f => %.3f' % (orig_score, scores[label]))

    for a in ax:
        a.get_xaxis().set_ticks([])
        a.get_yaxis().set_ticks([])
    
    plt.tight_layout()
    #f.subplots_adjust(wspace=0.05,hspace=0)
    if fig_path != None:
        plt.savefig(fig_path, bb_inches='tight', pad_inches=0.0)

def deletion_region_exp_setup_fig(path, label, mask_path, transformer, alphas = np.arange(0.2,1.0,0.2), show_titles = True, fig_path = None):
    f, ax = plt.subplots(1,len(alphas)+2)
    target = np.zeros(1000)
    target[label] = 1
    heatmap = np.load(mask_path)
    blur_heatmap = blur(heatmap)
    blur_heatmap = (blur_heatmap-blur_heatmap.min())/(blur_heatmap.max()-blur_heatmap.min())
    img = caffe.io.load_image(path)
    img_ = transformer.preprocess('data', img)
    blur_img_ = transformer.preprocess('data', blur(img))
    constant_img_ = np.zeros(img_.shape)
    rand_img_ = np.random.random(img_.shape)*255
    null_imgs_ = [blur_img_, constant_img_, rand_img_]
    ax[0].imshow(transformer.deprocess('data', img_))
    ax[1].imshow(heatmap)
    if show_titles:
        ax[0].set_title('Img')
        ax[1].set_title('Mask')

    for i in range(len(alphas)):
        alpha = alphas[i]
        bin_heatmap = np.ones(heatmap.shape)
        bin_heatmap[blur_heatmap <= alphas[i]] = 0
        comp_img_ = img_ * bin_heatmap + blur_img_ * (1 - bin_heatmap)
        ax[i+2].imshow(bin_heatmap)
        if show_titles:
            ax[i+2].set_title(r'$\alpha$=%.1f' % alpha)
    for a in ax:
        a.get_xaxis().set_visible(False)
        a.get_yaxis().set_visible(False)

    plt.subplots_adjust(wspace=0, hspace=0)
    if fig_path is not None:
        plt.savefig(fig_path, bbox_inches='tight', pad_inches=0.0)
