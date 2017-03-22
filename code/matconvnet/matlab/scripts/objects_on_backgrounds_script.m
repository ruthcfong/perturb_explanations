net = load('/home/ruthfong/packages/matconvnet/data/models/imagenet-caffe-alex.mat');
imdb = load('/home/ruthfong/saliency/data/ferrari/imdb.mat');
data_parent_dir = '/home/ruthfong/saliency/';
background_img = imread('/home/ruthfong/neural_coding/background_images/grass/1.jpg');
%img_idx = [17];
img_idx = 49; % echidna
layer = 0;

use_superpixels = true;

opts = struct();
opts.num_iters = 500;
opts.plot_step = 50;
%opts.save_fig_path = '';
%opts.save_res_path = '';
opts.loss = 'min_classlabel';
opts.num_class_labels = 0;

if use_superpixels
    opts.learning_rate = 1e1;
    opts.lambda = 1e-6;
    opts.tv_lambda = 1e-4;
    opts.beta = 3;
else
    opts.learning_rate = 1e2;
    opts.lambda = 5e-7;
    opts.tv_lambda = 1e-3;
    opts.beta = 3;
end

for i=1:length(img_idx)
    img_i = img_idx(i);
    img_name = imdb.images.name{img_i};
    num_segs = imdb.images.numSegs(img_i);
    img_path = fullfile(data_parent_dir, sprintf(imdb.paths.image, img_name));
    seg_path = fullfile(data_parent_dir, sprintf(imdb.paths.seg, img_name, num_segs));
    img = imread(img_path);
    seg = imread(seg_path);
    
    
    img_ = imresize(img, size(seg));
    background_img_ = imresize(background_img, size(seg));
    composite_img = uint8(bsxfun(@times, single(background_img_), single(1-seg))) ...
        + uint8(bsxfun(@times, single(img_), single(seg)));

    b_img = cnn_normalize(net.meta.normalization, background_img_, 1);
    c_img = cnn_normalize(net.meta.normalization, composite_img, 1);
    res_b = vl_simplenn(net, b_img);
    res_c = vl_simplenn(net, c_img);
    
    [~,sorted_idx_b] = sort(squeeze(res_b(end).x), 'descend');
    [~,sorted_idx_c] = sort(squeeze(res_c(end).x), 'descend');
    
%     get_short_class_name(net, sorted_idx_b(1:5), false)
%     res_b(end).x(sorted_idx_b(1:5))
    get_short_class_name(net, sorted_idx_c(1:5), false)
    squeeze(res_c(end).x(sorted_idx_c(1:5)))

    figure;
    subplot(2,3,1);
    imshow(img);
    subplot(2,3,2);
    imshow(seg);
    subplot(2,3,3);
    imshow(uint8(bsxfun(@times, single(img_), single(seg))));
    subplot(2,3,4);
    imshow(background_img_);
    subplot(2,3,5);
    imshow(composite_img);
    subplot(2,3,6);
    bar(squeeze(res_c(end-1).x));
        
    target_class = sorted_idx_c(1);
    
    opts.null_img = b_img;
    
%     % add a softmaxloss layer if needed
%     if ~isequal(net.layers{end}.type, 'softmaxloss')
%         net.layers{end+1} = struct('type', 'softmaxloss') ;
%     end
% 
%     if use_superpixels
%         res_mask = optimize_layer_feats_superpixels(net, c_img, target_class, layer, opts);
%     else
%         res_mask = optimize_layer_feats(net, c_img, target_class, layer, opts);
%     end
end

