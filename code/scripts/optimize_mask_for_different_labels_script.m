net = load('/home/ruthfong/packages/matconvnet/data/models/imagenet-caffe-alex.mat');
imdb_paths = load('/data/ruthfong/ILSVRC2012/val_imdb_paths.mat');

opts = struct();
opts.l1_ideal = 1;

opts.learning_rate = 1e1;
opts.num_iters = 500;cc
opts.adam.beta1 = 0.999;
opts.adam.beta2 = 0.999;
opts.adam.epsilon = 1e-8;
opts.lambda = 5e-7;
opts.tv_lambda = 1e-3;
opts.beta = 2;

opts.noise.use = true;
opts.noise.mean = 0;
opts.noise.std = 1e-3;
opts.mask_params.type = 'direct';
opts.update_func = 'adam';

num_top = 5;
img_i = 57;
img = cnn_normalize(net.meta.normalization, ...
    imread(imdb_paths.images.paths{img_i}), true);
target_class = imdb_paths.images.labels(img_i);
res = vl_simplenn(net, img);
res_size = size(res(end).x);
[~,top_idx] = sort(squeeze(res(end).x), 'descend');

for i=1:num_top
    curr_opts = opts;
    gradient = zeros(res_size, 'single');
    gradient(top_idx(i)) = 1;
    
    curr_opts.null_img = imgaussfilt(img, 10);
    
%     curr_opts.save_fig_path = fullfile(sprintf(strcat('/home/ruthfong/neural_coding/figures9/', ...
%         'imagenet/L0/min_classlabel_5_%s_blur/lr_%f_reg_lambda_%f_tv_norm_%f_beta_%f_num_iters_%d_noise_%d_%s/', ...
%         '%d_mask_dim_2.jpg'), ...
%      curr_opts.mask_params.type, curr_opts.learning_rate, curr_opts.lambda, curr_opts.tv_lambda, ...
%      curr_opts.beta, curr_opts.num_iters, curr_opts.noise.use, curr_opts.update_func), num2str(img_i));
%     curr_opts.save_res_path = fullfile(sprintf(strcat('/home/ruthfong/neural_coding/results9/', ...
%         'imagenet/L0/min_classlabel_5_%s_blur/lr_%f_reg_lambda_%f_tv_norm_%f_beta_%f_num_iters_%d_noise_%d_%s/', ...
%         '%d_mask_dim_2mat'), ...
%      curr_opts.mask_params.type, curr_opts.learning_rate, curr_opts.lambda, curr_opts.tv_lambda, ...
%      curr_opts.beta, curr_opts.num_iters, curr_opts.noise.use, curr_opts.update_func), num2str(img_i));
    
    mask_res = optimize_mask(net, img, gradient, curr_opts);
end