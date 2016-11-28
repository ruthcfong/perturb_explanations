is_local = false;
load_network = false;

if load_network
    if is_local
        % TODO
        assert(false);
    else
        net = load('/home/ruthfong/packages/matconvnet/data/models/imagenet-caffe-alex.mat');
    end
end

target_class = 11;
class_imdb_paths = load(sprintf('/data/ruthfong/ILSVRC2012/class_train_imdb_paths/%d_train_imdb_paths.mat', ...
        target_class));
class_imdb = build_imagenet_class_imdb(class_imdb_paths, net.meta.normalization);

if ~isequal(net.layers{end}.type, 'softmaxloss')
    net.layers{end+1} = struct('type', 'softmaxloss') ;
end

net.layers{end}.class = class_imdb.images.labels;
cres = vl_simplenn(net, class_imdb.images.data, 1);
