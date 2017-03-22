net = load('/home/ruthfong/packages/matconvnet/data/models/imagenet-caffe-alex.mat');
imdb_paths = load('/data/ruthfong/ILSVRC2012/val_imdb_paths.mat');

%img_idx = [1,2,5,8,3,6,7,20,57,12,14,18,21,27,37,41,61,70,76,91];
img_idx = [1];

opts = struct();

opts.learning_rate = 1e-1;
opts.num_iters = 500;
opts.plot_step = 20;
opts.noise.use = false;
opts.noise.mean = 0;
opts.noise.std = 1e-3;
opts.update_func = 'adam';

for i=1:length(img_idx)
    curr_opts = opts;
    img_i = img_idx(i);
    img = cnn_normalize(net.meta.normalization, ...
        imread(imdb_paths.images.paths{img_i}), true);
    
    target_class = imdb_paths.images.labels(img_i);
    res = vl_simplenn(net, img);
    gradient = zeros(size(res(end).x), 'single');
    gradient(target_class) = -1;
%     [~,top_idx] = sort(squeeze(res(end).x), 'descend');
%     gradient(top_idx(1:5)) = 1;
    
    %curr_opts.null_img = imgaussfilt(img, 10);
        
    zoom_res = optimize_zoom(net, img, gradient, target_class, curr_opts);
end