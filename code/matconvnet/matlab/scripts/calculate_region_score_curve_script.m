net = load('/home/ruthfong/packages/matconvnet/data/models/imagenet-caffe-alex.mat');
%net = load('/home/ruthfong/packages/matconvnet/data/models/imagenet-vgg-verydeep-16.mat');
imdb_paths = load('/data/ruthfong/ILSVRC2012/val_imdb_paths.mat');
%mask_res_dir = '/home/ruthfong/neural_coding/results7/imagenet/L0/min_classlabel_5/reg_lambda_0.000000_tv_norm_0.001000_beta_3.000000_num_iters_500_rand_stop_-100.000000_trans_linear_a_50.000000_adam/';
mask_res_dir = '/home/ruthfong/neural_coding/results8/imagenet/L0/min_classlabel_5/lr_100.000000_reg_lambda_0.000000_tv_norm_0.001000_beta_3.000000_num_iters_300_rand_trans_linear_adam/';
mask_dims = 2;
flip_mask = true;

%img_idx = [1,2,5,8,3,6,7,20,57,12,14,18,21,27,37,41,61,70,76,91];
img_idx = 1:500;
%img_idx = [41];

opts = {};
opts.window_size = 9;
%opts.quantile_range = 0.99:-0.01:0.5;
opts.quantile_range = 0.999:-0.001:0.9;
norm_deg = 2;

scores_mask = zeros([length(opts.quantile_range) length(img_idx)], 'single');
scores_sal = zeros([length(opts.quantile_range) length(img_idx)], 'single');
scores_deconv = zeros([length(opts.quantile_range) length(img_idx)], 'single');
scores_guided = zeros([length(opts.quantile_range) length(img_idx)], 'single');
scores_lrp = zeros([length(opts.quantile_range) length(img_idx)], 'single');

for i=1:length(img_idx)
    img_i = img_idx(i);
    img_path = imdb_paths.images.paths{img_i};
    target_class = imdb_paths.images.labels(img_i);

    img = imread(img_path);
    if ndims(img) == 2 % deal with b&w images
        img = repmat(img, [1 1 3]);
    end
    img_ = cnn_normalize(net.meta.normalization, img, 1);

    % Fong and Vedaldi, 2017 (Optimized Mask)
    mask_res = load(fullfile(mask_res_dir, sprintf('%d_mask_dim_%d.mat', img_i, mask_dims)));
    mask_res = mask_res.new_res;

    if flip_mask
        heatmap_mask = 1 - mask_res.mask;
    else
        heatmap_mask = mask_res.mask;
    end

    if size(heatmap_mask) ~= net.meta.normalization.imageSize(1:2)
        heatmap_mask = clip_map(imresize(heatmap_mask, net.meta.normalization.imageSize(1:2)));
    end

    scores_mask(:,i) = get_region_score_curve(net, img_, target_class, heatmap_mask, opts);

    % Simonyan et al, 2014 (Gradient-based Saliency)
    heatmap_sal = compute_heatmap(net, img_, target_class, 'saliency', norm_deg);
    scores_sal(:,i) = get_region_score_curve(net, img_, target_class, heatmap_sal, opts);

    % Zeiler & Fergus, 2014 (Deconvolutional network)
    heatmap_deconv = compute_heatmap(net, img_, target_class, 'deconvnet', norm_deg);
    scores_deconv(:,i) = get_region_score_curve(net, img_, target_class, heatmap_deconv, opts);

    % Springenberg et al., 2015, Mahendran and Vedaldi, 2015 (DeSalNet/Guided Backprop)
    heatmap_guided = compute_heatmap(net, img_, target_class, 'guided_backprop', norm_deg);
    scores_guided(:,i) = get_region_score_curve(net, img_, target_class, heatmap_guided, opts);

    % TBD (LRP-epsilon, eps = 100)
    heatmap_lrp = compute_heatmap(net, img_, target_class, 'lrp_epsilon', norm_deg);
    scores_lrp(:,i) = get_region_score_curve(net, img_, target_class, heatmap_lrp, opts);
end

scores = zeros([size(scores_mask) 5], 'single');
scores(:,:,1) = scores_mask;
scores(:,:,2) = scores_sal;
scores(:,:,3) = scores_deconv;
scores(:,:,4) = scores_guided;
scores(:,:,5) = scores_lrp;
mean_scores = squeeze(mean(scores, 2));
std_scores = squeeze(std(scores, 0, 2));

curve = bsxfun(@rdivide, mean_scores', 1+(1:length(opts.quantile_range)));
figure; plot(opts.quantile_range, curve);
legend({'mask','sal','deconv','guided','lrp'});

err = bsxfun(@rdivide, std_scores', (1+(1:length(opts.quantile_range)))*sqrt(length(img_idx)));
figure;
errorbar(curve', err');
legend({'mask','sal','deconv','guided','lrp'});

% scores = [scores_mask; scores_sal; scores_deconv; scores_guided; scores_lrp];
% area_curve = bsxfun(@rdivide,cumsum(scores,2), 1+(1:length(scores_mask)));
% 
% figure;
% plot(opts.quantile_range, area_curve');
% legend({'mask','sal','deconv','guided','lrp'});