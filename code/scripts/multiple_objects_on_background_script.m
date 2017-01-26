function multiple_objects_on_background_script

net = load('/home/ruthfong/packages/matconvnet/data/models/imagenet-caffe-alex.mat');
imdb = load('/home/ruthfong/saliency/data/ferrari/imdb.mat');
data_parent_dir = '/home/ruthfong/saliency/';
%background_img = imread('/home/ruthfong/neural_coding/background_images/grass/1.jpg');
background_img = 255*ones([1000 1000 3]);
img_i_a = 17; % dog
img_i_b = 49; % echidna
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

img_name_a = imdb.images.name{img_i_a};
num_segs_a = imdb.images.numSegs(img_i_a);
img_path_a = fullfile(data_parent_dir, sprintf(imdb.paths.image, img_name_a));
seg_path_a = fullfile(data_parent_dir, sprintf(imdb.paths.seg, img_name_a, num_segs_a));
img_a = imread(img_path_a);
seg_a = imread(seg_path_a);
img_a_ = imresize(img_a, size(seg_a));

img_name_b = imdb.images.name{img_i_b};
num_segs_b = imdb.images.numSegs(img_i_b);
img_path_b = fullfile(data_parent_dir, sprintf(imdb.paths.image, img_name_b));
seg_path_b = fullfile(data_parent_dir, sprintf(imdb.paths.seg, img_name_b, num_segs_b));
img_b = imread(img_path_b);
seg_b = imread(seg_path_b);
img_b_ = imresize(img_b, size(seg_b));

background_img_a = imresize(background_img, size(seg_a));
background_img_b = imresize(background_img, size(seg_b));

composite_img_a = uint8(bsxfun(@times, single(background_img_a), single(1-seg_a))) ...
    + uint8(bsxfun(@times, single(img_a_), single(seg_a)));
composite_img_b = uint8(bsxfun(@times, single(background_img_b), single(1-seg_b))) ...
    + uint8(bsxfun(@times, single(img_b_), single(seg_b)));

n_img = cnn_normalize(net.meta.normalization, background_img_a, 1);
res_n = vl_simplenn(net, n_img);
a_img = cnn_normalize(net.meta.normalization, composite_img_a, 1);
res_a = vl_simplenn(net, a_img);
b_img = cnn_normalize(net.meta.normalization, composite_img_b, 1);
res_b = vl_simplenn(net, b_img);

[~,sorted_idx_n] = sort(squeeze(res_n(end).x), 'descend');
[~,sorted_idx_a] = sort(squeeze(res_a(end).x), 'descend');
[~,sorted_idx_b] = sort(squeeze(res_b(end).x), 'descend');

%     get_short_class_name(net, sorted_idx_n(1:5), false)
%     res_n(end).x(sorted_idx_n(1:5))
% get_short_class_name(net, sorted_idx_a(1:5), false)
% squeeze(res_a(end).x(sorted_idx_a(1:5)))
% get_short_class_name(net, sorted_idx_b(1:5), false)
% squeeze(res_b(end).x(sorted_idx_b(1:5)))

background_img_ = imresize(background_img, [1000 1000]);

[comp_img_a, object_img_a, seg_img_a] = get_new_img_components(background_img_, ...
    img_a_, seg_a, 1);
[comp_img_b, object_img_b, seg_img_b] = get_new_img_components(background_img_, ...
    img_b_, seg_b, 501);

seg_c = seg_img_a + seg_img_b;
comp_img_c = uint8(bsxfun(@times, single(background_img_), single(1-seg_c)) ...
    + bsxfun(@times, single(object_img_a), single(seg_img_a)) ...
    + bsxfun(@times, single(object_img_b), single(seg_img_b)));

res_a = vl_simplenn(net, cnn_normalize(net.meta.normalization, comp_img_a, 1));
res_b = vl_simplenn(net, cnn_normalize(net.meta.normalization, comp_img_b, 1));
res_c = vl_simplenn(net, cnn_normalize(net.meta.normalization, comp_img_c, 1));

figure;
subplot(2,3,1);
imshow(comp_img_a);
subplot(2,3,2);
imshow(comp_img_b);
subplot(2,3,3);
imshow(comp_img_c);
subplot(2,3,4);
bar(squeeze(res_a(end-1).x));
subplot(2,3,5);
bar(squeeze(res_b(end-1).x));
subplot(2,3,6);
bar(squeeze(res_c(end-1).x));

[~,sorted_idx_a] = sort(squeeze(res_a(end-1).x), 'descend');
[~,sorted_idx_b] = sort(squeeze(res_b(end-1).x), 'descend');
[~,sorted_idx_c] = sort(squeeze(res_c(end-1).x), 'descend');

get_short_class_name(net, sorted_idx_a(1:5), false)
squeeze(res_a(end-1).x(sorted_idx_a(1:5)))
get_short_class_name(net, sorted_idx_b(1:5), false)
squeeze(res_b(end-1).x(sorted_idx_b(1:5)))
get_short_class_name(net, sorted_idx_c(1:5), false)
squeeze(res_c(end-1).x(sorted_idx_c(1:5)))

% [~,c_start_a] = find(seg_a, 1, 'first');
% [~,c_end_a] = find(seg_a, 1, 'last');
% [~,r_start_a] = find(seg_a', 1, 'first');
% [~,r_end_a] = find(seg_a', 1, 'last');
% 
% figure;
% subplot(2,3,1);
% imshow(img_a);
% subplot(2,3,2);
% imshow(seg_a);
% subplot(2,3,3);
% imshow(uint8(bsxfun(@times, single(img_a_), single(seg_a))));
% subplot(2,3,4);
% imshow(background_img_a);
% subplot(2,3,5);
% imshow(composite_img_a);
% hold on;
% rectangle('Position', [c_start_a, r_start_a, c_end_a - c_start_a,r_end_a - r_start_a]);
% hold off;
% subplot(2,3,6);
% bar(squeeze(res_a(end-1).x));
% 
% 
% [~,c_start_b] = find(seg_b, 1, 'first');
% [~,c_end_b] = find(seg_b, 1, 'last');
% [~,r_start_b] = find(seg_b', 1, 'first');
% [~,r_end_b] = find(seg_b', 1, 'last');
% 
% figure;
% subplot(2,3,1);
% imshow(img_b);
% subplot(2,3,2);
% imshow(seg_b);
% subplot(2,3,3);
% imshow(uint8(bsxfun(@times, single(img_b_), single(seg_b))));
% subplot(2,3,4);
% imshow(background_img_b);
% subplot(2,3,5);
% imshow(composite_img_b);
% hold on;
% rectangle('Position', [c_start_b, r_start_b, c_end_b - c_start_b,r_end_b - r_start_b]);
% hold off;
% subplot(2,3,6);
% bar(squeeze(res_b(end-1).x));
% 
% target_class = sorted_idx_a(1);
% 
% opts.null_img = n_img;

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

function [comp_img, object_img, seg_img] = get_new_img_components(background_img, orig_img, orig_seg, col_start)

    h_background = size(background_img, 1);
    w_background = size(background_img, 2);

    [h_seg, w_seg] = size(orig_seg);
    h_pad = floor((h_background-h_seg)/2);
    object_img = zeros(size(background_img), 'single');
    object_img(h_pad:h_seg+h_pad-1,col_start:col_start+w_seg-1,:) = orig_img;

    seg_img = zeros([h_background w_background], 'single');
    seg_img(h_pad:h_seg+h_pad-1,col_start:col_start+w_seg-1,:) = orig_seg;
    comp_img = uint8(bsxfun(@times, single(background_img), single(1-seg_img))) ...
        + uint8(bsxfun(@times, single(object_img), single(seg_img)));
    %figure; imshow(comp_img);
end

