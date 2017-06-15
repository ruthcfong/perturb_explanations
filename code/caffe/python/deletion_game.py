import caffe

import time, os

from heatmaps import *
from helpers import *
from optimize_mask import *

def main():
    gpu = 0 
    net_type = 'googlenet'
    data_desc = 'train_heldout'
    # TODO: Change this to the directory in which learned masks are stored
    #mask_dir = '/data/ruthfong/neural_coding/pycaffe_results/googlenet_train_heldout_given_grad_1_norm_0/min_top0_prob_blur/lr_-1.00_l1_lambda_-4.00_tv_lambda_-inf_l1_lambda_2_-2.00_beta_3.00_mask_scale_8_blur_mask_5_jitter_4_noise_-inf_num_iters_300_tv2_mask_init'
    mask_dir = '/data/ruthfong/neural_coding/results_reb/occ_masks_imagenet_googlenet_train_heldout_defaults'
    # TODO: Change this to desired output directorya (subdirs will be created)
    out_dir = '/users/ruthfong/neural_coding/deletion_game'
    mask_flip = True
    bb_method = 'min_max_diff'
    threshold = 0.01

    caffe.set_device(gpu)
    caffe.set_mode_gpu()

    if data_desc == 'train_heldout':
        (paths, labels) = read_imdb('../../../data/ilsvrc12/annotated_train_heldout_imdb.txt')
    elif data_desc == 'val':
        (paths, labels) = read_imdb('../../../data/ilsvrc12/val_imdb.txt')
    elif data_desc == 'animal_parts':
        (paths, labels) = read_imdb('../../../data/ilsvrc12/animal_parts_require_both_min_10_imdb.txt')
    
    mask_paths = [os.path.join(mask_dir, '%d.npy' % x) for x in range(len(labels))]

    net = get_net(net_type)
    net_transformer = get_ILSVRC_net_transformer(net)
    alphas = np.arange(0,1,0.01)
    #heatmap_types = ['mask', 'saliency', 'guided_backprop', 'excitation_backprop', 'contrast_excitation_backprop']
    heatmap_types = ['occlusion', 'grad_cam']
    num_imgs = len(labels)
    best_bb_sizes = np.zeros((len(heatmap_types), num_imgs))
    for i in range(num_imgs):
        start = time.time()
        img = net_transformer.preprocess('data', caffe.io.load_image(paths[i]))
        null_img = net_transformer.preprocess('data', get_blurred_img(paths[i]))
        target = np.zeros(1000)
        target[labels[i]] = 1
        orig_score = forward_pass(net, img, target)
        null_score = forward_pass(net, null_img, target)
        for hh in range(len(heatmap_types)):
            h = heatmap_types[hh]
            resize = None
            if h == 'mask' or h == 'occlusion':
                heatmap = np.load(mask_paths[i])
                if mask_flip:
                    heatmap = 1 - heatmap            
            else:
                top_name = 'loss3/classifier'
                if h == 'excitation_backprop':
                    bottom_name = 'pool2/3x3_s2'
                    norm_deg = -1
                    resize = (224,224)
                elif h == 'contrast_excitation_backprop':
                    bottom_name = 'pool2/3x3_s2'
                    norm_deg = -2
                    resize = (224,224)
                elif h == 'grad_cam':
                    bottom_name = 'inception_4e/output'
                    norm_deg = None
                    resize = (224,224)
                else:
                    bottom_name = 'data'
                    norm_deg = np.inf
                heatmap = compute_heatmap(net, net_transformer, paths[i], labels[i], h, top_name, top_name,
                        outputBlobName = bottom_name, outputLayerName = bottom_name, norm_deg = norm_deg, gpu = gpu)
            bb_sizes = np.zeros(len(alphas))
            scores = np.zeros(len(alphas))
            for j in range(len(alphas)):
                alpha = alphas[j]
                (x0,y0,x1,y1) = getbb_from_heatmap(heatmap, alpha, method = bb_method, resize = resize)
                mask = np.ones((224,224))
                mask[y0:y1,x0:x1] = 0
                bb_size = (y1-y0)*(x1-x0)
                bb_sizes[j] = bb_size
                comp_img = img * mask + null_img * (1 - mask)
                masked_score = forward_pass(net, comp_img, target)
                scores[j] =(masked_score-null_score)/float(orig_score-null_score)
           # alphas_scores_path = os.path.join(out_dir, '%s_%s/%s/alpha_scores_num_imgs_%d_%s.txt' % (net_type, data_desc,
           #                                         mask_rel_dir, num_imgs, h))
           # bb_sizes_path = os.path.join(out_dir, '%s_%s/%s/bb_sizes_num_imgs_%d_%s.txt' % (net_type, data_desc,
           #                         mask_rel_dir, num_imgs, h))
            alphas_scores_path = os.path.join(out_dir, '%s_%s/alpha_scores_num_imgs_%d_%s.txt' % (net_type, data_desc,
                                                    num_imgs, h))
            bb_sizes_path = os.path.join(out_dir, '%s_%s/bb_sizes_num_imgs_%d_%s.txt' % (net_type, data_desc,
                                    num_imgs, h))
            directory = os.path.dirname(alphas_scores_path)
            if not os.path.exists(directory):
                os.makedirs(directory)
            f = open(alphas_scores_path, 'a')
            np.savetxt(f,scores[None,:])
            f.close()
            f = open(bb_sizes_path, 'a')
            np.savetxt(f, bb_sizes[None,:])
            f.close()
            try:
                best_i = np.where(scores <= threshold)[0][-1]
            except:
                best_i = np.argmin(scores)
            best_bb_sizes[hh][i] = bb_sizes[best_i]
        print 'Image %d took %f' % (i, time.time() - start) 
        if i % 50 == 0:
            print 'heatmap_type', 'best_bb_size_mean', 'best_bb_size_std'
            for hh in range(len(heatmap_types)):
                print heatmap_types[hh], best_bb_sizes[hh][:(i+1)].mean(), best_bb_sizes[hh][:(i+1)].std()

if __name__ == '__main__':
    main()
