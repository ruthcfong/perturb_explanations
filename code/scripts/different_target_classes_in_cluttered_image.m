net = load('/home/ruthfong/packages/matconvnet/data/models/imagenet-vgg-verydeep-16.mat');
%net.layers = net.layers(1:end-1); % eliminate the softmax layer
%net = load('/home/ruthfong/packages/matconvnet/data/models/imagenet-caffe-alex.mat');

img = cnn_normalize(net.meta.normalization, imread('/home/ruthfong/neural_coding/fnn_images/comic_book_cluttered_image.jpg'), 1);

res = vl_simplenn(net, img);

[~, sorted_idx] = sort(squeeze(res(end).x), 'descend');
get_short_class_name(net, sorted_idx(1:10), 0)

interested_classes = ...
    [293, ... % tiger
    367, ... % gorilla
    389, ... % giant_panda (388 = lesser_panda)
    387, ... % African_elephant (386 = Indian_elephant)
    292, ... % lion
    ];

masks = zeros([net.meta.normalization.imageSize(1:2) length(interested_classes)], 'single');

opts = struct();
opts.l1_ideal = 0;
opts.learning_rate = 1e0; %1e1;
opts.num_iters = 1000;
opts.plot_step = 100;
opts.adam.beta1 = 0.9; %0.999;
opts.adam.beta2 = 0.999;
opts.adam.epsilon = 1e-8;
% opts.lambda = 5e-7;
% opts.tv_lambda = 1e-3;
% opts.beta = 3;
opts.lambda = 5e-8; %1e-10;
opts.tv_lambda = 5e-6; %1e-8; 
opts.beta = 1.5; % 1.2;
opts.noise.use = true;
opts.noise.mean = 0;
opts.noise.std = 1e-3;
opts.mask_params.type = 'direct';
%opts.mask_params.type = 'superpixels';
opts.update_func = 'adam';
opts.null_img = imgaussfilt(img, 10);
opts.gpu = 1;

for i = 1:length(interested_classes)
    target_class = interested_classes(i);
    gradient = zeros(size(res(end).x), 'single');
    gradient(target_class) = -1;
    new_res = optimize_mask(net, img, gradient, opts);
    masks(:,:,i) = new_res.mask;
end

figure;
subplot(2,3,1);
imshow(uint8(cnn_denormalize(net.meta.normalization, img)));
title('Img');
for i=1:length(interested_classes)
    subplot(2,3,1+i);
    imshow(normalize(masks(:,:,:,i)));
    title(get_short_class_name(net, interested_classes(i), 1));
end
    