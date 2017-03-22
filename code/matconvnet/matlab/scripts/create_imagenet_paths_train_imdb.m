%%
% addpath('/home/ruthfong/deep-goggle/helpers'); % for cnn_normalize.m

images_dir_path = '/data/datasets/ILSVRC2012/images/train';
annotations_dir_path = '/data/ruthfong/ILSVRC2012/Annotation';
% % save original CYMK/poorly formatted files in the below folder
% obs_images_dir_path = '/data/ruthfong/ILSVRC2012/images/val_obs_original';
% if ~exist(obs_images_dir_path,'dir')
%     mkdir(obs_images_dir_path);
% end
savepath = '/data/ruthfong/ILSVRC2012/annotated_train_imdb_paths.mat';

num_classes = 1000;

meta_imagenet = load('/data/datasets/ILSVRC2012/ILSVRC2014_devkit/data/meta_clsloc.mat');
net_imagenet = load('/home/ruthfong/packages/matconvnet/data/models/imagenet-caffe-alex.mat');

dir_paths = dir(images_dir_path);
dir_paths = dir_paths(3:end);
ann_dir_paths = dir(annotations_dir_path);
ann_dir_paths = ann_dir_paths(3:end);

assert(length(dir_paths) == num_classes);

wnid_to_net_class = zeros([1 num_classes]);
for i=1:num_classes
    wnid_to_net_class(i) = find(strcmp(net_imagenet.meta.classes.name, ...
        meta_imagenet.synsets(i).WNID));
end

%%
all_paths = {};
all_labels = {};
all_annotation_paths = {};

for i=1:num_classes,
    wnid = find(strcmp({meta_imagenet.synsets(1:num_classes).WNID}, dir_paths(i).name));
    assert(length(wnid) == 1);
    target_class = wnid_to_net_class(wnid);
    
    if exist(fullfile(annotations_dir_path, dir_paths(i).name), 'dir') == 0
        continue;
    end
    
    image_paths = dir(fullfile(images_dir_path, dir_paths(i).name));
    image_paths = image_paths(3:end);
    
    % save image paths
    paths = fullfile(images_dir_path, dir_paths(i).name, {image_paths(:).name});
    ann_paths = {};
    has_ann_idx = zeros(size(image_paths), 'logical');
    for j=1:length(image_paths)
        [~,filename,~] = fileparts(paths{j});
        ann_path = fullfile(annotations_dir_path, dir_paths(i).name, [filename, '.xml']);
        if exist(ann_path, 'file')
            has_ann_idx(j) = 1;
            ann_paths = [ann_paths ann_path];
        end
    end
    
    % save net target class labels (not wnid labels)
    labels = ones([sum(has_ann_idx), 1], 'single')*target_class;

%     images = struct();
%     images.paths = paths(has_ann_idx);
%     images.annotation_paths = ann_paths;
%     images.labels = labels;
% 
%     meta = struct();
%     meta.classes = net_imagenet.meta.classes;

%     save(sprintf('/data/ruthfong/ILSVRC2012/class_train_imdb_paths/%d_train_imdb_paths.mat', ...
%         target_class), 'images', 'meta','-v7.3');
    
    all_paths = [all_paths paths(has_ann_idx)];
    all_labels = [all_labels; labels];
    all_annotation_paths = [all_annotation_paths ann_paths];
end

images = struct();
images.paths = all_paths;
images.labels = vertcat(all_labels{:});
images.annotation_paths = all_annotation_paths;

meta = struct();
meta.classes = net_imagenet.meta.classes;
save(savepath, 'images', 'meta', '-v7.3');