alexnet = load('/home/ruthfong/packages/matconvnet/data/models/imagenet-caffe-alex.mat');
vgg16 = load('/home/ruthfong/packages/matconvnet/data/models/imagenet-vgg-verydeep-16.mat');
googlenet = load('/home/ruthfong/packages/matconvnet/data/models/imagenet-googlenet-dag.mat');

use_train = false;
use_heldout = true;

if use_train
    imdb_paths = load('/data/ruthfong/ILSVRC2012/annotated_train_imdb_paths.mat');
    img_idx = load('/data/ruthfong/ILSVRC2012/annotated_train_heldout_idx.mat');
    img_idx = img_idx.heldout_idx;
    dataset_description = 'annotated_train_heldout';
else
    imdb_paths = load('/data/ruthfong/ILSVRC2012/val_imdb_paths.mat');
    annotation_dir = '/data/ruthfong/ILSVRC2012/val';
    if use_heldout
        img_idx = [1,2,5,8,3,6,7,20,57,12,14,18,21,27,37,41,61,70,76,91];
        %dataset_description = 'val_heldout';
    else
        img_idx = 1:50000;
        %dataset_description = 'val';
    end
end

mask_dirs = {'/data/ruthfong/neural_coding/results10/imagenet/alexnet_val_heldout/L0/min_classlabel_5_direct_blur/lr_1.000000_reg_lambda_-7.301030_tv_norm_-5.301030_beta_1.500000_num_iters_500_noise_1_jitter_10_adam', ...
    '/data/ruthfong/neural_coding/results10/imagenet/vgg16_val_heldout/L0/min_classlabel_5_direct_blur/lr_1.000000_reg_lambda_-7.301030_tv_norm_-5.301030_beta_1.500000_num_iters_500_noise_1_jitter_10_adam', ...
    '/data/ruthfong/neural_coding/results10/imagenet/googlenet_val_heldout/L0/min_classlabel_5_direct_blur/lr_1.000000_reg_lambda_-7.301030_tv_norm_-5.301030_beta_1.500000_num_iters_500_noise_1_jitter_10_adam'};
nets = {alexnet, vgg16, googlenet};

alphas = [1.5 1.5 1.5]; % to change

num_masks = length(mask_dirs);
opts = struct();
opts.flip = true;
opts.annotation_files = '';
opts.meta_file = '/data/ruthfong/ILSVRC2012/ILSVRC2014_devkit/data/meta_clsloc.mat';

load(opts.meta_file);
wnid_to_im_id = cellfun(@(net_out) find(cellfun(@(s) ~isempty(strfind(s, net_out)), ...
    {synsets.WNID})), alexnet.meta.classes.name);

for i=1:length(img_idx)
    img_i = img_idx(i);
    
    opts.gt_labels = repmat(wnid_to_im_id(imdb_paths.images.labels(img_i)), [1 3]);
    opts.title_prefix = {'alexnet: ', 'vgg16: ', 'googlenet: '};
    opts.save_fig_path = strcat('/data/ruthfong/neural_coding/figures11/compare_across_networks_val_heldout/lr_1.000000_reg_lambda_-7.301030_tv_norm_-5.301030_beta_1.500000_num_iters_500_noise_1_jitter_10_adam', ...
        sprintf('%d.jpg', img_i));
    
    if use_train
        assert(false);
    else
        [~,filename,~] = fileparts(imdb_paths.images.paths{img_i});
        opts.annotation_files = fullfile(annotation_dir, [filename, '.xml']);
    end

    mask_paths = cell([1 num_masks]);
    for j=1:num_masks
        mask_paths{j} = fullfile(mask_dirs{j}, [num2str(img_i), '.mat']);
    end
    img_paths = imdb_paths.images.paths{img_i};
    
    compare_multiple_masks_localization(nets, mask_paths, img_paths, alphas, opts);
end
