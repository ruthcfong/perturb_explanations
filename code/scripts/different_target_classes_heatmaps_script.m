% % load relevant imagenet synsets (TODO -- this is only the next layer)
% % http://image-net.org/synset?wnid=n02084071
% dog_synsets = {'n02084071', 'n01322604', 'n02112497', 'n02113335', 'n02111277', 'n02084732', 'n02111129', 'n02103406', 'n02112826', 'n02111626', 'n02110958', 'n02110806', 'n02085272', 'n02113978', 'n02087122', 'n02111500', 'n02110341', 'n02085374', 'n02084861'};
% 
% % http://image-net.org/synset?wnid=n02121808
% cat_synsets = {'n02121808', 'n02122430', 'n02122725', 'n02123478', 'n02122298', 'n02124313', 'n02122878', 'n02123045', 'n02122510', 'n02123597', 'n02124157', 'n02123394', 'n02123159', 'n02124484', 'n02123917', 'n02124075', 'n02123242'};
% 
% % http://image-net.org/synset?wnid=n02958343
% car_synsets = {'n02958343', 'n04516354', 'n02701002', 'n02814533', 'n02924554', 'n02930766', 'n03079136', 'n03100240', 'n03119396', 'n03141065', 'n03268790', 'n03421669', 'n03493219', 'n03498781', 'n03539103', 'n03543394', 'n03594945', 'n03670208', 'n03680512', 'n03770085', 'n03770679', 'n03777568', 'n03870105', 'n04037443', 'n04097373', 'n04166281', 'n04285008', 'n04285965', 'n04302988', 'n04322924', 'n04347119', 'n04459122'};
% 
% % http://image-net.org/synset?wnid=n02834778
% bike_synsets = {'n02834778', 'n02835271', 'n03792782', 'n03853924', 'n04026813', 'n04126066', 'n04524716'};
% 
%  % http://image-net.org/synset?wnid=n02391049
% zebra_synsets = {'n02391049', 'n02391234', 'n02391373', 'n02391508'};
% 
% % http://image-net.org/synset?wnid=n02503517
% elephant_synsets = {'n02503517', 'n02504013', 'n02506783', 'n02504770', 'n02504458', 'n02503756'};
% 
% synset = dog_synsets;
% synset2 = cat_synsets;
% 
% s_idx = zeros([1 length(synset)]);
% for i=1:length(synset)
%     res = find(cellfun(@(s) ~isempty(strfind(s, synset{i})), net.meta.classes.name));
%     if ~isempty(res), s_idx(i) = res; end
% end

%net = load('/home/ruthfong/packages/matconvnet/data/models/imagenet-caffe-alex.mat');
net = load('/home/ruthfong/packages/matconvnet/data/models/imagenet-vgg-verydeep-16.mat');

%img = cnn_normalize(net.meta.normalization, imread('dog-cat3.jpg'), 1);
img = cnn_normalize(net.meta.normalization, imread('zeb-ele1.jpg'), 1);
obj1_name = 'elephant';
obj2_name = 'zebra';
res = vl_simplenn(net, img);
[~,sorted_idx] = sort(res(end).x, 'descend');
get_short_class_name(net, sorted_idx(1:10), 0)

% dog-cat3.jpg
% obj1_i = 188; % Yorkshire_terrier (1)
% obj2_i = 282; % tabby (7?)

% bic-car1.jpg
% obj1_i = 818; % sports_car (1)
% obj2_i = 871; % tricycle (5)
% norm_deg = Inf;

% zeb-ele1.jpg
obj1_i = 387; % African_elephant (1)
obj2_i = 341; % zebra

norm_deg = Inf;
obj1_sal = compute_heatmap(net, img, obj1_i, 'saliency', norm_deg);
obj2_sal = compute_heatmap(net, img, obj2_i, 'saliency', norm_deg);
obj1_deconv = compute_heatmap(net, img, obj1_i, 'deconvnet', norm_deg);
obj2_deconv = compute_heatmap(net, img, obj2_i, 'deconvnet', norm_deg);
obj1_guided = compute_heatmap(net, img, obj1_i, 'guided_backprop', norm_deg);
obj2_guided = compute_heatmap(net, img, obj2_i, 'guided_backprop', norm_deg);

opts = struct();
opts.l1_ideal = 1;
opts.learning_rate = 1e1;
opts.num_iters = 300;
opts.adam.beta1 = 0.999;
opts.adam.beta2 = 0.999;
opts.adam.epsilon = 1e-8;
opts.lambda = 5e-7;
opts.tv_lambda = 1e-3;
opts.beta = 3;

opts.noise.use = true;
opts.noise.mean = 0;
opts.noise.std = 1e-3;
opts.mask_params.type = 'direct';
opts.update_func = 'adam';

opts.null_img = imgaussfilt(img, 10);

gradient = zeros(size(res(end).x), 'single');
gradient(obj1_i) = 1;
new_res = optimize_mask(net, img, gradient, opts);
dog_mask = 1 - new_res.mask;

%opts.mask_params.type = 'superpixels';
%opts.l1_ideal = 0;
gradient = zeros(size(res(end).x), 'single');
gradient(obj2_i) = 1;
%net.layers = net.layers(1:end-1);
new_res = optimize_mask(net, img, gradient, opts);
cat_mask = 1 - new_res.mask;

figure;
subplot(3,4,1);
imshow(normalize(obj1_sal));
title(sprintf('Saliency (%s)', obj1_name));
subplot(3,4,2);
imshow(normalize(obj1_deconv));
title('Deconvnet');
subplot(3,4,3);
imshow(normalize(obj1_guided));
title('Guided Backprop');
subplot(3,4,4);
imshow(normalize(dog_mask));
title('Learned Mask');
subplot(3,4,5);
imshow(normalize(obj2_sal));
title(sprintf('Saliency (%s)', obj2_name));
subplot(3,4,6);
imshow(normalize(obj2_deconv));
title('Deconvnet');
subplot(3,4,7);
imshow(normalize(obj2_guided));
title('Guided Backprop');
subplot(3,4,8);
imshow(normalize(cat_mask));
title('Learned Mask');
subplot(3,4,9);
imshow(normalize(img));
title('Original Image');