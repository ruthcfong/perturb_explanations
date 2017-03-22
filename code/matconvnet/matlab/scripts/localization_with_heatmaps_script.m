net_type = 'googlenet';
switch net_type
    case 'alexnet'
        net = load('/home/ruthfong/packages/matconvnet/data/models/imagenet-caffe-alex.mat');
    case 'vgg16'
        net = load('/home/ruthfong/packages/matconvnet/data/models/imagenet-vgg-verydeep-16.mat');
    case 'googlenet'
        net = load('/home/ruthfong/packages/matconvnet/data/models/imagenet-googlenet-dag.mat');
    otherwise
        assert(false);
end

is_training = false;
use_heldout = false;

if is_training
    imdb_paths = load('/data/ruthfong/ILSVRC2012/annotated_train_imdb_paths.mat');
    all_img_idx = load('/data/ruthfong/ILSVRC2012/annotated_train_heldout_idx.mat');
    all_img_idx = all_img_idx.heldout_idx;
else
    imdb_paths = load('/data/ruthfong/ILSVRC2012/val_imdb_paths.mat');
    if use_heldout
        all_img_idx = [1,2,3,5,6,7,8,12,14,18,20,21,27,37,41,57,61,70,76,91];
    else
        all_img_idx = 1:50000;
    end
end

heatmap_type = 'saliency';
alpha = 2.0;
special_insertion = '';
opts = struct();
opts.gpu = 3;
opts.batch_size = 250;
opts.resize_one_side = -1;
opts.layer_name = '';
%opts.layer_name = 'pool2';
opts.norm_deg = Inf;

if strcmp(heatmap_type, 'mask')
    if use_heldout
        opts.mask_dir = '/data/ruthfong/neural_coding/results10/imagenet/googlenet_annotated_train_heldout/L0/min_classlabel_5_direct_blur/lr_1.000000_reg_lambda_-7.301030_tv_norm_-5.301030_beta_1.500000_num_iters_500_noise_1_jitter_10_adam';
        %opts.mask_dir = '/data/ruthfong/neural_coding/results12/grid_masks/googlenet_annotated_train_heldout/num_top_5_flip_1_sigma_500_num_centers_484/';
        %opts.mask_dir = '/data/ruthfong/neural_coding/results10/imagenet/googlenet_annotated_train_heldout/L0/min_classlabel_5_direct_blur/lr_1.000000_reg_lambda_-7.301030_tv_norm_-5.301030_beta_1.500000_num_iters_500_noise_1_jitter_10_adam';
        %opts.mask_dir = '/data/ruthfong/neural_coding/results10/imagenet/alexnet_annotated_train_heldout/L0/min_classlabel_5_direct_blur/lr_1.000000_reg_lambda_-7.301030_tv_norm_-5.301030_beta_1.500000_num_iters_500_noise_1_jitter_10_adam';
        %opts.mask_dir = '/data/ruthfong/neural_coding/results10/imagenet/googlenet_annotated_train_heldout/L0/min_classlabel_5_direct_blur/lr_1.000000_reg_lambda_-7.301030_tv_norm_-5.301030_beta_1.500000_num_iters_500_noise_1_jitter_0_adam';
        %opts.mask_dir = '/data/ruthfong/neural_coding/results10/imagenet/vgg16_annotated_train_heldout/L0/min_classlabel_5_direct_blur/lr_1.000000_reg_lambda_-7.301030_tv_norm_-5.301030_beta_1.500000_num_iters_500_noise_1_jitter_10_adam';
        %opts.mask_dir = '/data/ruthfong/neural_coding/results10/imagenet/alexnet_annotated_train_heldout/L0/min_classlabel_5_direct_blur/lr_1.000000_reg_lambda_-7.301030_tv_norm_-5.301030_beta_1.500000_num_iters_500_noise_1_jitter_10_adam';
        %opts.mask_dir = '/data/ruthfong/neural_coding/results10/imagenet/vgg16_annotated_train_heldout/L0/min_classlabel_5_direct_blur/lr_1.000000_reg_lambda_-7.301030_tv_norm_-5.301030_beta_1.500000_num_iters_500_noise_1_jitter_0_adam';
        %opts.mask_dir = '/data/ruthfong/neural_coding/results10/imagenet/alexnet_annotated_train_heldout/L0/min_classlabel_5_direct_blur/lr_1.000000_reg_lambda_-7.301030_tv_norm_-5.301030_beta_1.500000_num_iters_500_noise_1_adam';
        %opts.mask_dir = '/data/ruthfong/neural_coding/results10/imagenet/alexnet_annotated_train_heldout/L0/min_classlabel_5_direct_blur/lr_1.000000_reg_lambda_-7.301030_tv_norm_-5.301030_beta_1.500000_num_iters_500_noise_1_jitter_10_adam';
    else
        opts.mask_dir = '/data/ruthfong/neural_coding/results10/imagenet/alexnet_val/L0/min_classlabel_5_direct_blur/lr_1.000000_reg_lambda_-7.301030_tv_norm_-5.301030_beta_1.500000_num_iters_500_noise_1_adam';
    end
    opts.mask_flip = true;
    
    opts.batch_size = 250;
    ss = strsplit(opts.mask_dir, '/');
    %special_insertion = ['grid_masks/' ss{end} '/'];
    special_insertion = [ss{end-1} '/' ss{end} '/'];
end

heatmap_types = {'saliency','guided_backprop'};
alphas = [5.0, 4.5];
%heatmap_types = {heatmap_type};
%alphas = [alpha];

if use_heldout
    lambdas = 10.^(-8:-1);
    betas = 1:3;
    jitters = [0 2 4 8];

    for j=1:length(heatmap_types)
        heatmap_type = heatmap_types{j};
        alphas = 0:0.5:10;
        %alphas = 1:0.5:10;
        for lambda=lambdas
            for tv_lambda=lambdas
                for beta = betas
                    for jitter = jitters
                        for i=1:length(alphas)
                            alpha = alphas(i);
                            special_insertion = sprintf('min_classlabel_5_direct_provided/lr_1.000000_reg_lambda_%f_tv_norm_%f_beta_%f_num_iters_500_noise_1_jitter_%d_adam/', ...
                                log10(lambda), log10(tv_lambda), beta, jitter);
                            opts.mask_dir = sprintf(strcat('/data/ruthfong/neural_coding/results10/', ...
                                        'imagenet/alexnet_val_heldout/L0/%s'), special_insertion);

                            out_file = sprintf('/data/ruthfong/neural_coding/loc_preds/%s_annotated_train_heldout_gt/%s/%salpha_%.1f%s_norm_deg_%d_rescale_%d_thres_first_exp_res.txt', ...
                                net_type, heatmap_type, special_insertion, alpha, opts.layer_name, opts.norm_deg, opts.resize_one_side);
                            if exist(out_file, 'file')
                                fprintf('skipping %s because it already exists\n', out_file);
                                continue;
                            end
                            localization_with_heatmaps(net, imdb_paths, all_img_idx, alpha, heatmap_type, out_file, opts);
                        end
                    end
                end
            end
        end
    end
else
    for j=1:length(heatmap_types)
        heatmap_type = heatmap_types{j};
        alpha = alphas(j);
        out_file = sprintf('/data/ruthfong/neural_coding/loc_preds/%s_val_gt/%s/%salpha_%.1f%s_norm_deg_%d_rescale_%d_thres_first_exp_res.txt', ...
        net_type, heatmap_type, special_insertion, alpha, opts.layer_name, opts.norm_deg, opts.resize_one_side);
        if exist(out_file, 'file')
            fprintf('skipping %s because it already exists\n', out_file);
        else
            localization_with_heatmaps(net, imdb_paths, all_img_idx, alpha, heatmap_type, out_file, opts);
        end
    end
end