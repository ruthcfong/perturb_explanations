net = load('/home/ruthfong/packages/matconvnet/data/models/imagenet-caffe-alex.mat');
%net = load('/home/ruthfong/packages/matconvnet/data/models/imagenet-vgg-verydeep-16.mat');
imdb_paths = load('/data/ruthfong/ILSVRC2012/val_imdb_paths.mat');
%mask_res_dir = '/home/ruthfong/neural_coding/results7/imagenet/L0/min_classlabel_5/reg_lambda_0.000000_tv_norm_0.001000_beta_3.000000_num_iters_500_rand_stop_-100.000000_trans_linear_a_50.000000_adam/';
mask_res_dir = '/home/ruthfong/neural_coding/results8/imagenet/L0/min_classlabel_5/lr_100.000000_reg_lambda_0.000000_tv_norm_0.001000_beta_3.000000_num_iters_300_rand_trans_linear_adam/';
mask_dims = 2;
flip_mask = true;

%img_idx = [1,2,5,8,3,6,7,20,57,12,14,18,21,27,37,41,61,70,76,91];
img_i = 3;
img_path = imdb_paths.images.paths{img_i};
target_class = imdb_paths.images.labels(img_i);

img = imread(img_path);
if ndims(img) == 2 % deal with b&w images
    img = repmat(img, [1 1 3]);
end
img_ = cnn_normalize(net.meta.normalization, img, 1);

opts = {};
opts.num_iters = 100;
opts.num_perturbs = 10;
opts.window_size = 9;
opts.show_fig = false;

norm_deg = 2;

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

[aopc_mask, diff_scores_mask, pert_img_mask] = calculate_aopc(net, ...
    img, target_class, heatmap_mask, opts, opts);

% Simonyan et al, 2014 (Gradient-based Saliency)
heatmap_sal = compute_heatmap(net, img_, target_class, 'saliency', norm_deg);
[aopc_sal, diff_scores_sal, pert_img_sal] = calculate_aopc(net, ...
    img, target_class, heatmap_sal, opts);

% Zeiler & Fergus, 2014 (Deconvolutional network)
heatmap_deconv = compute_heatmap(net, img_, target_class, 'deconvnet', norm_deg);
[aopc_deconv, diff_scores_deconv, pert_img_deconv] = calculate_aopc(net, ...
    img, target_class, heatmap_deconv, opts);

% Springenberg et al., 2015, Mahendran and Vedaldi, 2015 (DeSalNet/Guided Backprop)
heatmap_guided = compute_heatmap(net, img_, target_class, 'guided_backprop', norm_deg);
[aopc_guided, diff_scores_guided, pert_img_guided] = calculate_aopc(net, ...
    img, target_class, heatmap_guided, opts);

% TBD (LRP-epsilon, eps = 100)
heatmap_lrp = compute_heatmap(net, img_, target_class, 'lrp_epsilon', norm_deg);
[aopc_lrp, diff_scores_lrp, pert_img_lrp] = calculate_aopc(net, ...
    img, target_class, heatmap_lrp, opts);

% random baseline
heatmap_random = rand(net.meta.normalization.imageSize(1:2), 'single');
[aopc_rand, diff_scores_rand, pert_img_rand] = calculate_aopc(net, ...
    img_, target_class, opts);

figure;
plot(aopc_mask-aopc_rand);
hold on;
plot(aopc_sal-aopc_rand);
plot(aopc_deconv-aopc_rand);
plot(aopc_guided-aopc_rand);
plot(aopc_lrp-aopc_rand);
plot(zeros(size(aopc_rand)));
hold off;
legend({'mask','sal','deconv','guided','lrp-eps','rand'});
ylabel('AOPC relative to random');
xlabel('perturbation steps');

figure;
subplot(2,3,1);
imshow(normalize(img_));
title('orig');
subplot(2,3,2);
imshow(normalize(bsxfun(@times, img_, heatmap_mask)));
title('mask');
subplot(2,3,3);
imshow(normalize(bsxfun(@times, img_, heatmap_sal)));
title('saliency');
subplot(2,3,4);
imshow(normalize(bsxfun(@times, img_, heatmap_deconv)));
title('deconv');
subplot(2,3,5);
imshow(normalize(bsxfun(@times, img_, heatmap_guided)));
title('guided');
subplot(2,3,6);
imshow(normalize(bsxfun(@times, img_, heatmap_lrp)));
title('lrp_epsilon');


figure;
subplot(2,3,1);
imshow(normalize(img_));
title('orig');
subplot(2,3,2);
imshow(normalize(heatmap_mask));
title('mask');
subplot(2,3,3);
imshow(normalize(heatmap_sal));
title('saliency');
subplot(2,3,4);
imshow(normalize(heatmap_deconv));
title('deconv');
subplot(2,3,5);
imshow(normalize(heatmap_guided));
title('guided');
subplot(2,3,6);
imshow(normalize(heatmap_lrp));
title('lrp_epsilon');

figure;
subplot(2,4,1);
imshow(normalize(img_));
title('orig');
subplot(2,4,2);
imshow(pert_img_mask);
title('mask');
subplot(2,4,3);
imshow(pert_img_sal);
title('saliency');
subplot(2,4,4);
imshow(pert_img_deconv);
title('deconv');
subplot(2,4,5);
imshow(pert_img_guided);
title('guided');
subplot(2,4,6);
imshow(pert_img_lrp);
title('lrp_epsilon');
subplot(2,4,7);
imshow(normalize(pert_img_rand));
title('random');

