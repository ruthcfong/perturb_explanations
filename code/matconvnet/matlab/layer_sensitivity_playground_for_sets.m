load_network = true;
load_dataset = true;
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

% class_i = 2; % 2 - goldfish; 818 - sportscar
% 818: sports car, sports car, 445: tandem bicycle, 672: mountain bike,
% 780: school bus, 655: minibus, 875: trolley bus, 467: bullet train, 
% 896: warplane
%class_idx = [445, 467, 672, 780, 655, 875, 896, 818];
class_idx = [467, 672, 780, 655, 875, 896, 818];

layers = [2, 6, 10, 12, 14]; % relu 1-5 layers for alexnet
for class_i=class_idx
    fprintf('running color sensitivity experiment for %s\n', ...
        get_short_class_name(net, class_i, false));
    run_color_sensitivity_experiment(net,imdb_paths,class_i, layers);
    close all;
end
