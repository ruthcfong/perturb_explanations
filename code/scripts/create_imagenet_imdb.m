addpath('/home/ruthfong/deep-goggle/helpers'); % for cnn_normalize.m

% use my own copy so I can correct for the few CYMK files
images_dir_path = '/data/ruthfong/ILSVRC2012/images/val';%/data/datasets/ILSVRC2012/images/val';
% save original CYMK/poorly formatted files in the below folder
obs_images_dir_path = '/data/ruthfong/ILSVRC2012/images/val_obs_original';
if ~exist(obs_images_dir_path,'dir')
    mkdir(obs_images_dir_path);
end
wnid_labels_path = '/data/datasets/ILSVRC2012/ILSVRC2014_devkit/data/ILSVRC2014_clsloc_validation_ground_truth.txt';
savepath = '/data/ruthfong/ILSVRC2012/val_imdb.mat';

num_files = 50000;
num_classes = 1000;

wnid_labels_f = fopen(wnid_labels_path, 'r');
wnid_labels = fscanf(wnid_labels_f, '%d');

% assert(max(wnid_labels) == 1000 && min(wnid_labels) == 1);

meta_imagenet = load('/data/datasets/ILSVRC2012/ILSVRC2014_devkit/data/meta_clsloc.mat');
net_imagenet = load('/home/ruthfong/packages/matconvnet/data/models/imagenet-caffe-alex.mat');

image_paths = dir(images_dir_path);
image_paths = image_paths(3:end); % ignore '.' and '..'
assert(length(image_paths) == num_files);


wnid_to_net_class = zeros([1 num_classes]);
for i=1:num_classes
    wnid_to_net_class(i) = find(strcmp(net_imagenet.meta.classes.name, ...
        meta_imagenet.synsets(i).WNID));
end

% save normalized images without jitter
data = zeros([net_imagenet.meta.normalization.imageSize(1:3) length(wnid_labels)]);

% save net target class labels (not wnid labels)
labels = zeros(size(wnid_labels));

for i=1:num_files
    image_path = fullfile(images_dir_path, image_paths(i).name);
    try
        img = imread(image_path);
    catch
        % copy original file to backup directory and convert it to RGB
        [~,image_name, ext] = filepaths(image_path);
        backup_image_path = fullfile(obs_images_dir_path, strcat(image_name, ext));
        disp(fprintf('backing up %s to %s and converting it to a RGB file\n', ...
            image_path, backup_image_path));
        assert(system(sprintf('cp %s %s', image_path, backup_image_path)) == 0);
        assert(system(sprintf('convert %s -colorspace RGB %s', image_path, image_path)) == 0);
        
        img = imread(image_path);
    end
    norm_img = cnn_normalize(net_imagenet.meta.normalization, img, true);
    data(:,:,:,i) = norm_img;
    labels(i) = wnid_to_net_class(wnid_labels(i));
end

data_mean = mean(data,4);

images = struct();
images.data = data;
images.data_mean = data_mean;
images.labels = labels;

meta = struct();
meta.classes = net_imagenet.meta.classes;

save(savepath, 'images', 'meta','-v7.3');

%% save small version of validation set with 10 examples per class (in sorted order by class)
num_per_class = 10;
all_idx = zeros([1 num_per_class*num_classes]);
for i=1:num_classes
    idx = find(labels == i, num_per_class);
    all_idx((i-1)*num_per_class+1:i*num_per_class) = idx;
end

images.data = single(data(:,:,:,all_idx));
images.data_mean = single(mean(data(:,:,:,all_idx), 4));
images.idx = all_idx;
images.labels = labels(all_idx);
save('/data/ruthfong/ILSVRC2012/val_imdb_small.mat', 'images', 'meta', '-v7.3');