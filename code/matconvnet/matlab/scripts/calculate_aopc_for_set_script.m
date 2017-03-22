net = load('/home/ruthfong/packages/matconvnet/data/models/imagenet-caffe-alex.mat');
imdb_paths = load('/data/ruthfong/ILSVRC2012/val_imdb_paths.mat');
%mask_res_dir = '/home/ruthfong/neural_coding/results7/imagenet/L0/min_classlabel_5/reg_lambda_0.000000_tv_norm_0.001000_beta_3.000000_num_iters_500_rand_stop_-100.000000_trans_linear_a_50.000000_adam/';
%mask_res_dir = '/home/ruthfong/neural_coding/results8/imagenet/L0/min_classlabel_5/lr_100.000000_reg_lambda_0.000000_tv_norm_0.001000_beta_3.000000_num_iters_300_rand_trans_linear_adam/';
%mask_res_dir = '/home/ruthfong/neural_coding/results7/imagenet/L0/min_classlabel_5_superpixels/lr_10.000000_reg_lambda_0.000001_tv_norm_0.000100_beta_3.000000_num_iters_500_rand_trans_linear_adam/';
mask_res_dir = '/home/ruthfong/neural_coding/results8/imagenet/L0/min_classlabel_5/lr_10.000000_reg_lambda_0.000000_tv_norm_0.001000_beta_3.000000_num_iters_350_rand_trans_linear_adam/';
mask_dims = 2;
flip_mask = true;

img_idx = [1,2,5,8,3,6,7,20,57,12,14,18,21,27,37,41,61,70,76,91];
%img_idx=1:500;
opts = {};
opts.num_iters = 30;
opts.num_perturbs = 1;
opts.window_size = 9;
opts.show_fig = false;

norm_deg = 2;

save_res = true;
if save_res
    save_res_dir = strrep(mask_res_dir, 'lr', sprintf('aopc_iters_%d_perturbs_%d_with_pert_imgs_lr', ...
        opts.num_iters, opts.num_perturbs));
    prep_path(save_res_dir);
end

paths = imdb_paths.images.paths;
labels = imdb_paths.images.labels;
normalization = net.meta.normalization;

diff_scores_mask = zeros([opts.num_iters, length(img_idx)], 'single');
diff_scores_sal = zeros([opts.num_iters, length(img_idx)], 'single');
diff_scores_deconv = zeros([opts.num_iters, length(img_idx)], 'single');
diff_scores_guided = zeros([opts.num_iters, length(img_idx)], 'single');
diff_scores_lrp = zeros([opts.num_iters, length(img_idx)], 'single');
diff_scores_rand = zeros([opts.num_iters, length(img_idx)], 'single');

for i=1:length(img_idx)
    img_i = img_idx(i);
    img_path = paths{img_i};
    target_class = labels(img_i);

    img = imread(img_path);
    if ndims(img) == 2 % deal with b&w images
        img = repmat(img, [1 1 3]);
    end

    img_ = cnn_normalize(normalization, img, 1);

    % Fong and Vedaldi, 2017 (Optimized Mask)
    mask_path = fullfile(mask_res_dir, sprintf('%d_mask_dim_%d.mat', img_i, mask_dims));
    mask_res = load(mask_path);
    mask_res = mask_res.new_res;

    if flip_mask
        heatmap_mask = 1 - mask_res.mask;
    else
        heatmap_mask = mask_res.mask;
    end

    [~, diff_scores_mask(:,i), pert_img_mask] = calculate_aopc(net, ...
        img, target_class, heatmap_mask, opts);

    % Simonyan et al, 2014 (Gradient-based Saliency)
    heatmap_sal = compute_heatmap(net, img_, target_class, 'saliency', norm_deg);
    [~, diff_scores_sal(:,i), pert_img_sal] = calculate_aopc(net, ...
        img, target_class, heatmap_sal, opts);

    % Zeiler & Fergus, 2014 (Deconvolutional network)
    heatmap_deconv = compute_heatmap(net, img_, target_class, 'deconvnet', norm_deg);
    [~, diff_scores_deconv(:,i), pert_img_deconv] = calculate_aopc(net, ...
        img, target_class, heatmap_deconv, opts);

    % Springenberg et al., 2015, Mahendran and Vedaldi, 2015 (DeSalNet/Guided Backprop)
    heatmap_guided = compute_heatmap(net, img_, target_class, 'guided_backprop', norm_deg);
    [~, diff_scores_guided(:,i), pert_img_guided] = calculate_aopc(net, ...
        img, target_class, heatmap_guided, opts);

    % TBD (LRP-epsilon, eps = 100)
    heatmap_lrp = compute_heatmap(net, img_, target_class, 'lrp_epsilon', norm_deg);
    [~, diff_scores_lrp(:,i), pert_img_lrp] = calculate_aopc(net, ...
        img, target_class, heatmap_lrp, opts);

    % random baseline
    heatmap_random = rand(net.meta.normalization.imageSize(1:2), 'single');
    [~, diff_scores_rand(:,i), pert_img_rand] = calculate_aopc(net, ...
        img, target_class, heatmap_random, opts);
    
    if save_res
        res = struct();
        res.mask.flip = flip_mask;
        res.mask.path = mask_path;
        res.mask.heatmap = heatmap_mask;
        res.mask.pert_img = pert_img_mask;
        res.mask.diff_scores = diff_scores_mask(:,i);
        res.saliency.heatmap = heatmap_sal;
        res.saliency.pert_img = pert_img_sal;
        res.saliency.diff_scores = diff_scores_sal(:,i);
        res.deconvnet.heatmap = heatmap_deconv;
        res.deconvnet.pert_img = pert_img_deconv;
        res.deconvnet.diff_scores = diff_scores_deconv(:,i);
        res.guided.heatmap = heatmap_guided;
        res.guided.pert_img = pert_img_guided;
        res.guided.diff_scores = diff_scores_guided(:,i);
        res.lrp_epsilon.heatmap = heatmap_lrp;
        res.lrp_epsilon.pert_img = pert_img_lrp;
        res.lrp_epsilon.diff_scores = diff_scores_lrp(:,i);
        res.random.heatmap = heatmap_random;
        res.random.pert_img = pert_img_rand;
        res.random.diff_scores = diff_scores_rand(:,i);
        res.norm_deg = norm_deg;
        res.opts = opts;
        res.image.path = img_path;
        res.image.target_class = target_class;
        
        save(fullfile(save_res_dir, sprintf('%d.mat', img_i)), 'res');
    end
end

aopc_mask = transpose(mean(cumsum(diff_scores_mask,1),2))./(1+(1:opts.num_iters));
aopc_sal = transpose(mean(cumsum(diff_scores_sal,1),2))./(1+(1:opts.num_iters));
aopc_deconv = transpose(mean(cumsum(diff_scores_deconv,1),2))./(1+(1:opts.num_iters));
aopc_guided = transpose(mean(cumsum(diff_scores_guided,1),2))./(1+(1:opts.num_iters));
aopc_lrp = transpose(mean(cumsum(diff_scores_lrp,1),2))./(1+(1:opts.num_iters));
aopc_rand = transpose(mean(cumsum(diff_scores_rand,1),2))./(1+(1:opts.num_iters));

err_mask = (transpose(std(cumsum(diff_scores_mask,1),0,2))./(1+(1:opts.num_iters)))/sqrt(length(img_idx));
err_sal = (transpose(std(cumsum(diff_scores_sal,1),0,2))./(1+(1:opts.num_iters)))/sqrt(length(img_idx));
err_deconv = (transpose(std(cumsum(diff_scores_deconv,1),0,2))./(1+(1:opts.num_iters)))/sqrt(length(img_idx));
err_guided = (transpose(std(cumsum(diff_scores_guided,1),0,2))./(1+(1:opts.num_iters)))/sqrt(length(img_idx));
err_lrp = (transpose(std(cumsum(diff_scores_lrp,1),0,2))./(1+(1:opts.num_iters)))/sqrt(length(img_idx));
err_rand = (transpose(std(cumsum(diff_scores_rand,1),0,2))./(1+(1:opts.num_iters)))/sqrt(length(img_idx));


figure;
hold on;
errorbar(aopc_mask-aopc_rand, err_mask);
errorbar(aopc_sal-aopc_rand, err_sal);
errorbar(aopc_deconv-aopc_rand, err_deconv);
errorbar(aopc_guided-aopc_rand, err_guided);
errorbar(aopc_lrp-aopc_rand, err_lrp);
errorbar(zeros(size(aopc_rand)), err_rand);
hold off;
legend({'mask','sal','deconv','guided','lrp-epsilon','rand'});
%legend({'sal','deconv','rand'});
ylabel('AOPC relative to random');
xlabel('perturbation steps');

figure;
hold on;
plot(aopc_mask-aopc_rand);
plot(aopc_sal-aopc_rand);
plot(aopc_deconv-aopc_rand);
plot(aopc_guided-aopc_rand);
plot(aopc_lrp-aopc_rand);
plot(zeros(size(aopc_rand)));
hold off;
legend({'mask','sal','deconv','guided','lrp-epsilon','rand'});
%legend({'sal','deconv','rand'});
ylabel('AOPC relative to random');
xlabel('perturbation steps');


% figure;
% subplot(2,3,1);
% imshow(normalize(img_));
% title('orig');
% subplot(2,3,2);
% imshow(normalize(bsxfun(@times, img_, heatmap_mask)));
% title('mask');
% subplot(2,3,3);
% imshow(normalize(bsxfun(@times, img_, heatmap_sal)));
% title('saliency');
% subplot(2,3,4);
% imshow(normalize(bsxfun(@times, img_, heatmap_deconv)));
% title('deconv');
% subplot(2,3,5);
% imshow(normalize(bsxfun(@times, img_, heatmap_guided)));
% title('guided');
% 
% figure;
% subplot(2,3,1);
% imshow(normalize(img_));
% title('orig');
% subplot(2,3,2);
% imshow(normalize(pert_img_mask));
% title('mask');
% subplot(2,3,3);
% imshow(normalize(pert_img_sal));
% title('saliency');
% subplot(2,3,4);
% imshow(normalize(pert_img_deconv));
% title('deconv');
% subplot(2,3,5);
% imshow(normalize(pert_img_guided));
% title('guided');
% subplot(2,3,6);
% imshow(normalize(pert_img_rand));
% title('random');

