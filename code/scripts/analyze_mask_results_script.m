net = load('/home/ruthfong/packages/matconvnet/data/models/imagenet-caffe-alex.mat');
imdb_paths = load('/data/ruthfong/ILSVRC2012/val_imdb_paths.mat');

if ~isequal(net.layers{end}.type, 'softmaxloss')
    net.layers{end+1} = struct('type', 'softmaxloss') ;
end

img_i = 1;
target_class = imdb_paths.images.labels(img_i);

res_path = fullfile('/home/ruthfong/neural_coding/results7/imagenet/L0/min_max_classlabel_0_superpixels/lr_10.000000_reg_lambda_0.000001_tv_norm_0.000100_beta_3.000000_num_iters_500_rand_trans_linear_adam/',...
    sprintf('%d_mask_dim_2.mat', img_i));

analyze_mask_results(net, res_path, target_class);
