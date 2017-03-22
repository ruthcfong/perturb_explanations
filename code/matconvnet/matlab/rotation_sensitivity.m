net = load('/home/ruthfong/packages/matconvnet/data/models/imagenet-caffe-alex.mat');
imdb_paths = load('/data/ruthfong/ILSVRC2012/val_imdb_paths.mat');

opts = struct();
opts.layers = [2, 6, 10, 12, 14]; % relu 1-5 alexnet

% 818: sports car, sports car, 445: tandem bicycle, 672: mountain bike,
% 780: school bus, 655: minibus, 875: trolley bus, 467: bullet train, 
% 896: warplane
% 156, Shih-Tzu, 88: African grey, 622: lawn mower, 387: African elephant 549: entertainment center
class_idx = [445, 467, 672, 780, 655, 875, 896, 818, 156, 88, 622, 387, 549];

for class_i=class_idx
    disp(get_short_class_name(net, class_i, false));
    image_idx_for_class = find(imdb_paths.images.labels == class_i);
    for i=1:length(image_idx_for_class)
        disp(i);
        img = imread(imdb_paths.images.paths{image_idx_for_class(i)});
        norm_img = cnn_normalize(net.meta.normalization, img, true);
        
        opts.fig_path = fullfile('/home/ruthfong/neural_coding/figures5/imagenet/rotation_sensitivity',...
            sprintf('%d_%s', class_i, get_short_class_name(net, class_i, false)), ...
            sprintf('%d.jpg', image_idx_for_class(i)));
        run_rotation_sensitivity_experiment(net, norm_img, opts);
        close all;
    end
end