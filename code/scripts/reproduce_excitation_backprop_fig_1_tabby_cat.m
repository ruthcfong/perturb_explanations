%net = load('/home/ruthfong/packages/matconvnet/data/models/imagenet-vgg-verydeep-16.mat');
net = load('/home/ruthfong/packages/matconvnet/data/models/imagenet-caffe-alex.mat');
net_dag = dagnn.DagNN.fromSimpleNN(net);
img = cnn_normalize(net.meta.normalization, imread('~/neural_coding/images/tabby_cat_cropped.jpg'), 1);
target_class = 282;
disp(get_short_class_name(net, target_class, 0));

opts = struct();
opts.lrp_epsilon = 100;
opts.lrp_alpha = 1;
norm_deg = Inf;

[heatmap, res] = compute_heatmap(net_dag, img, target_class, 'lrp_alpha_beta', norm_deg, opts);
%[heatmap, res] = compute_heatmap(net, repmat(img, [1 1 1 2]), [target_class target_class], 'lrp_epsilon', norm_deg, opts);

layer_names = {'pool5', 'pool4', 'pool3', 'pool2', 'pool1'};
layer_idx = zeros(size(layer_names));
for i=1:length(layer_idx)
    layer_i = find(cellfun(@(x) strcmp(x.name, layer_names{i}), net.layers));
    assert(length(layer_i) == 1);
    layer_idx(i) = layer_i;
end

img_size = size(img);

figure;
subplot(2,3,1);
imshow(uint8(cnn_denormalize(net.meta.normalization, img)));
title(get_short_class_name(net,target_class,1));
for i=1:length(layer_idx)
    subplot(2,3,i+1);
    hm = map2jpg(normalize(sum(res(layer_idx(i)+1).dzdx, 3)));
    hm_resized = imresize(hm, img_size(1:2), 'nearest');
    imshow(hm_resized);
    title(layer_names{i});
end