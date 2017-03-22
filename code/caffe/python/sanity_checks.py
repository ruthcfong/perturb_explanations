import caffe

from heatmaps import *
from optimize_mask import *


def main():
    gpu = 0
    net_type = 'googlenet'
    caffe.set_device(gpu)
    caffe.set_mode_gpu()
    net = get_net(net_type)
    labels_desc = np.loadtxt('/home/ruthfong/packages/caffe/data/ilsvrc12/synset_words.txt', str, delimiter='\t')
    synsets = np.loadtxt('/home/ruthfong/packages/caffe/data/ilsvrc12/synsets.txt', str, delimiter='\t')
    (paths, labels) = read_imdb('/home/ruthfong/packages/caffe/data/ilsvrc12/annotated_train_heldout_imdb.txt')
    paths = np.array(paths)
    labels = np.array(labels)

    res_dir = '/data/ruthfong/neural_coding/pycaffe_results'
    mask_rel_dir = 'googlenet_train_heldout_given_grad_1_norm_0/min_top0_prob_blur/lr_-1.00_l1_lambda_-4.00_tv_lambda_-inf_l1_lambda_2_-2.00_beta_3.00_mask_scale_8_blur_mask_5_jitter_4_noise_-inf_num_iters_300_tv2_mask_init'
    mask_paths = [os.path.join(res_dir, mask_rel_dir, x) for x in os.listdir(os.path.join(res_dir, mask_rel_dir))]
    num_top = 0 
    transformer = get_ILSVRC_net_transformer(net)
    pylab.rcParams['figure.figsize'] = (12.0,12.0)
    for i in range(100):
        img = transformer.preprocess('data', caffe.io.load_image(paths[i]))
        scores = forward_pass(net, img)
        sorted_idx = np.argsort(scores)
        target = np.zeros(scores.shape)
        if num_top == 0:
            target[labels[i]]   = 1
        else:
            target[sorted_idx[:-(num_top+1):-1]] = 1
        fig_path = os.path.join('/data/ruthfong/neural_coding/sanity_checks', mask_rel_dir, '%d.png' % i)
        check_mask_generalizability(net, paths[i], target, mask_paths[i], last_layer = 'prob', fig_path = fig_path)
        plt.close() 

if __name__ == '__main__':
    main()
