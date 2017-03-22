import caffe
import time
import os

from optimize_mask import *
from helpers import *

def run_generality_experiment(net, paths, labels, mask_paths, out_path, alphas = np.arange(0,1.0,0.05)):
    transformer = get_ILSVRC_net_transformer(net)
    assert(len(paths) == len(labels) and len(paths) == len(mask_paths))
    idx = range(len(paths)))
    net_shape = net.blobs['data'].data.shape
    all_scores = np.zeros((len(idx), len(alphas)))
    start = time.time()
    for j in range(len(idx)):
        if j % 50 == 1:
            start = time.time()
        ind = idx[j]
        target = np.zeros(1000)
        target[labels[ind]] = 1
        heatmap = np.load(mask_paths[ind])
        blur_heatmap = blur(heatmap)
        blur_heatmap = (blur_heatmap-blur_heatmap.min())/float(blur_heatmap.max()-blur_heatmap.min())
        img = caffe.io.load_image(paths[ind])
        img_ = transformer.preprocess('data', img)
        blur_img_ = transformer.preprocess('data', blur(img))
        if perturbation == 'blur':
            null_img_ = blur_img_
        elif perturbation == 'mean_img':
            null_img_ = np.zeros(img_.shape)
        elif perturbation == 'random_noise':
            null_img_ = np.random.random(img_.shape)*255
        orig_score = forward_pass(net, img_, target)
        mask_score = forward_pass(net, img_ * heatmap + blur_img_ * (1 - heatmap), target)
        net.blobs['data'].reshape(len(alphas),net_shape[1], net_shape[2], net_shape[3])
        comp_imgs_ = np.zeros((len(alphas), net_shape[1], net_shape[2], net_shape[3]))
        for i in range(len(alphas)):
            alpha = alphas[i]
            bin_heatmap = np.ones(heatmap.shape)
            bin_heatmap[blur_heatmap < alphas[i]] = 0
            comp_imgs_[i,...] = img_ * bin_heatmap + null_img_ * (1 - bin_heatmap)
        net.blobs['data'].data[...] = comp_imgs_
        net.forward()
        target_scores = net.blobs['prob'].data[:,labels[ind]]
        norm_scores = (target_scores - mask_score)/float(orig_score - mask_score)
        all_scores[j] = norm_scores
        net.blobs['data'].reshape(net_shape[0],net_shape[1],net_shape[2],net_shape[3])
        if j % 50 == 0:
            print j, (time.time() - start)
    f = open(out_path, 'w')
    np.savetxt(f, all_scores)
    f.close()
    print 'Wrote to', out_path

def main():
    gpu = 0 
    net_type = 'googlenet'
    out_dir = '/home/ruthfong/neural_coding/sanity_check_test'

    caffe.set_device(gpu)
    caffe.set_mode_gpu()

    (paths, labels) = read_imdb('../../../data/ilsvrc12/annotated_train_heldout_imdb.txt')
    data_desc = 'train_heldout'
    # TODO: Replace with directory in which learned masks are saved
    mask_dir = '/data/ruthfong/neural_coding/pycaffe_results/googlenet_train_heldout_given_grad_1_norm_0/min_top0_prob_blur/lr_-1.00_l1_lambda_-4.00_tv_lambda_-inf_l1_lambda_2_-2.00_beta_3.00_mask_scale_8_blur_mask_5_jitter_4_noise_-inf_num_iters_300_tv2_mask_init'
    mask_paths = [os.path.join(mask_dir, '%d.npy' % x) for x in range(len(labels))]

    net = get_net(net_type)

    perturbations = ['blur', 'mean_img', 'random_noise']
    for i in range(len(perturbations)):
        perturbation = perturbations[i]
        out_path = os.path.join(out_dir, '%s_%s_%s_scores.txt' % (net_type, data_desc, perturbation))

if __name__ == '__main__':
    main()
