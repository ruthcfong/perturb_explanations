%%
% addpath('/home/ruthfong/deep-goggle/helpers'); % for cnn_normalize.m

images_dir_path = '/data/datasets/ILSVRC2012/images/train';
% % save original CYMK/poorly formatted files in the below folder
% obs_images_dir_path = '/data/ruthfong/ILSVRC2012/images/val_obs_original';
% if ~exist(obs_images_dir_path,'dir')
%     mkdir(obs_images_dir_path);
% end
savepath = '/data/ruthfong/ILSVRC2012/train_imdb_paths.mat';

num_classes = 1000;

meta_imagenet = load('/data/datasets/ILSVRC2012/ILSVRC2014_devkit/data/meta_clsloc.mat');
net_imagenet = load('/home/ruthfong/packages/matconvnet/data/models/imagenet-caffe-alex.mat');

dir_paths = dir(images_dir_path);
dir_paths = dir_paths(3:end); 
assert(length(dir_paths) == num_classes);

wnid_to_net_class = zeros([1 num_classes]);
for i=1:num_classes
    wnid_to_net_class(i) = find(strcmp(net_imagenet.meta.classes.name, ...
        meta_imagenet.synsets(i).WNID));
end

%%
all_paths = {};
all_labels = {};

for i=1:num_classes,
    wnid = find(strcmp({meta_imagenet.synsets(1:num_classes).WNID}, dir_paths(i).name));
    assert(length(wnid) == 1);
    target_class = wnid_to_net_class(wnid);
    
    image_paths = dir(fullfile(images_dir_path, dir_paths(i).name));
    image_paths = image_paths(3:end);
    
    % save image paths
    paths = fullfile(images_dir_path, dir_paths(i).name, {image_paths(:).name});

    % save net target class labels (not wnid labels)
    labels = ones(size(image_paths), 'single')*target_class;

    images = struct();
    images.paths = paths;
    images.labels = labels;

    meta = struct();
    meta.classes = net_imagenet.meta.classes;

    save(sprintf('/data/ruthfong/ILSVRC2012/class_train_imdb_paths/%d_train_imdb_paths.mat', ...
        target_class), 'images', 'meta','-v7.3');
    
    all_paths = [all_paths paths];
    all_labels = [all_labels; labels];
end

images = struct();
images.paths = all_paths;
images.labels = all_labels;

meta = struct();
meta.classes = net_imagenet.meta.classes;
save(savepath, 'images', 'meta', '-v7.3');