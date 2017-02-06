net = load('/home/ruthfong/packages/matconvnet/data/models/imagenet-caffe-alex.mat');

use_training_heldout = false;

if use_training_heldout
    imdb_paths = load('/data/ruthfong/ILSVRC2012/annotated_train_imdb_paths.mat');
    all_img_idx = load('/data/ruthfong/ILSVRC2012/annotated_train_heldout_idx.mat');
    all_img_idx = all_img_idx.heldout_idx;
else
    imdb_paths = load('/data/ruthfong/ILSVRC2012/val_imdb_paths.mat');
    all_img_idx = 1:50000;
end

heatmap_type = 'mask';
alpha = 1.5;
special_insertion = '';
opts = struct();
opts.gpu = 1;
opts.batch_size = 500;

if strcmp(heatmap_type, 'mask')
    if use_training_heldout
        opts.mask_dir = '/data/ruthfong/neural_coding/results10/imagenet/alexnet_annotated_train_heldout/L0/min_classlabel_5_direct_blur/lr_1.000000_reg_lambda_-7.301030_tv_norm_-5.301030_beta_1.500000_num_iters_500_noise_1_adam';
    else
        opts.mask_dir = '/data/ruthfong/neural_coding/results10/imagenet/alexnet_val/L0/min_classlabel_5_direct_blur/lr_1.000000_reg_lambda_-7.301030_tv_norm_-5.301030_beta_1.500000_num_iters_500_noise_1_adam';
    end
    opts.mask_flip = true;
    
    opts.batch_size = 500;
    ss = strsplit(opts.mask_dir, '/');
    special_insertion = [ss{end-1} '/' ss{end} '/'];
end

if use_training_heldout
    alphas = 0:0.5:10;
    %alphas = 1:0.5:10;
    for i=1:length(alphas)
        alpha = alphas(i);
        out_file = sprintf('/data/ruthfong/neural_coding/loc_preds/annotated_train_heldout/%s/%salpha_%.1f.txt', ...
            heatmap_type, special_insertion, alpha);
        if exist(out_file, 'file')
            fprintf('skipping %s because it already exists\n', out_file);
            continue;
        end
        localization_with_heatmaps(net, imdb_paths, all_img_idx, alpha, heatmap_type, out_file, opts);
    end
else
    out_file = sprintf('/data/ruthfong/neural_coding/loc_preds/val/%s/%salpha_%.1f.txt', ...
        heatmap_type, special_insertion, alpha);
    localization_with_heatmaps(net, imdb_paths, all_img_idx, alpha, heatmap_type, out_file, opts);
end