visualize_different_layers = false;

object_net = load('/home/ruthfong/packages/matconvnet/data/models/imagenet-caffe-alex.mat');
img_i = 25260;
img = imread(imdb_paths.images.paths{img_i});
img = cnn_normalize(object_net.meta.normalization, img, true);
target_class = imdb_paths.images.labels(img_i);

% object_net = net;
% object_net.layers = object_net.layers(1:end-1);
% img_i = batch_range(13);
% img = single(imdb.images.data(:,:,:,img_i));
% 
% img = imresize(img, object_net.meta.normalization.imageSize(1:2));
% display_img = img + object_net.meta.normalization.averageImage;

%img = single(img) - single(object_net.meta.normalization.averageImage);

res_cam = vl_simplenn(object_net, img);

[~,sorted_class_idx] = sort(res_cam(end).x);

num_top = 5;

layer = 14;

img_size = size(img);
[H_l,W_l,K_l] = size(res_cam(layer+1).x);

softmax_norm = sum(exp(res_cam(end).x));

figure;

subplot(2,3,1);
imshow(normalize(img));
if (isfield(object_net.meta.classes, 'description'))
    title(sprintf('Orig Img: %s', object_net.meta.classes.description{...
        target_class}));
else
    title(sprintf('Orig Img: %s', object_net.meta.classes{...
        target_class}));
end


for i=1:num_top
    gradient = zeros(size(res_cam(end).x),'single');
    c = sorted_class_idx(end-i+1);
    gradient(c) = single(1);
    % score = exp(res(end).x(c))/softmax_norm;
    score = res_cam(end).x(c);
    new_res = vl_simplenn(object_net, img, gradient);
    weights = sum(sum(new_res(layer+1).dzdx,1),2) / (H_l*W_l);
    map = bsxfun(@max, sum(bsxfun(@times, new_res(layer+1).x, weights),3), 0);
    large_heatmap = map2jpg(im2double(imresize(map, img_size(1:2))));

    subplot(2,3,i+1);
    imshow(normalize(img)*0.3 + 0.7*large_heatmap);
    if isfield(object_net.meta.classes, 'description')
        title(sprintf('%f: %s', score, object_net.meta.classes.description{c}));
    else
        title(sprintf('%f: %s', score, object_net.meta.classes{c}));
    end
end

%% visualize different layers
if visualize_different_layers
    layers = [3,7,10,12,14];

    gradient = zeros(size(res_cam(end).x),'single');
    c = sorted_class_idx(end);
    gradient(c) = single(1);
    score = res_cam(end).x(c);
    new_res = vl_simplenn(object_net, img, gradient);

    figure;

    subplot(2,3,1);
    imshow(normalize(display_img));
    if isfield(object_net.meta.classes, 'description')
        title(sprintf('%f: %s', score, object_net.meta.classes.description{c}));
    else
        title(sprintf('%f: %s', score, object_net.meta.classes{c}));
    end

    for i=1:length(layers)
        layer = layers(i);
        weights = sum(sum(new_res(layer+1).dzdx,1),2) / (H_l*W_l);
        map = bsxfun(@max, sum(bsxfun(@times, new_res(layer+1).x, weights),3), 0);
        large_heatmap = map2jpg(im2double(imresize(map, img_size(1:2))));

        subplot(2,3,i+1);
        imshow(normalize(im2double(display_img))*0.3 + 0.7*large_heatmap);
        title(object_net.layers{layer}.name);
    end
end