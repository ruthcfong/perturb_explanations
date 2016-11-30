load_network = false;
load_dataset = false;
is_local = false;

if load_network
    if is_local
        % TODO
        assert(false);
    else
        net = load('/home/ruthfong/packages/matconvnet/data/models/imagenet-caffe-alex.mat');
    end
end

if load_dataset
    if is_local
        assert(false); % TODO
    else
        imdb_paths = load('/data/ruthfong/ILSVRC2012/val_imdb_paths.mat');
    end
end

%% search imagenet images
% how to search for a class
present_idx = find(cellfun(@(s) ~isempty(strfind(s, 'lawn')), net.meta.classes.description));
class_i = present_idx(3); 
% 818: sports car, sports car, 445: tandem bicycle, 672: mountain bike,
% 780: school bus, 655: minibus, 875: trolley bus, 467: bullet train, 
% 896: warplane
disp(sprintf('%d: %s\n', class_i, net.meta.classes.description{class_i}));
image_idx_for_class = find(imdb_paths.images.labels == class_i);

%% show images
num_show = 50;
figure;
for i=1:num_show
    subplot(ceil(sqrt(num_show)),ceil(sqrt(num_show)),i)
    img = imread(imdb_paths.images.paths{image_idx_for_class(i)});
    imshow(normalize(cnn_normalize(net.meta.normalization, img, true)));
    title(sprintf('%d: %d', i, image_idx_for_class(i)));
end