% net = load('/home/ruthfong/packages/matconvnet/data/models/imagenet-caffe-alex.mat');
net = load('/home/ruthfong/packages/matconvnet/data/models/imagenet-matconvnet-vgg-verydeep-16.mat');

% 818: sports car, sports car, 445: tandem bicycle, 672: mountain bike,
% 780: school bus, 655: minibus, 875: trolley bus, 467: bullet train, 
% 896: warplane
classes = [445, 467, 672, 780, 655, 875, 896, 818];
% layers = [7, 10, 12, 15]; % norm2, relu3, relu4, pool5 for alexnet
layers = [10, 17, 24, 31]; % pool 2, 3, 4, and 5
Ks = [24, 32, 48, 64, 96];

opts = struct();
opts.use_norm = true;
for target_class=classes
    for layer=layers
        for K=Ks
            opts.save_fig_dir = sprintf('~/neural_coding/figures5/imagenet/population_encoding_vgg_very_deep_16_norm_%d/%d_%s/%s/K_%d', ...
                opts.use_norm, target_class, get_short_class_name(net, target_class, false), ...
                net.layers{layer}.name, K);
            run_population_encoding_imagenet_experiment(net, layer, ...
                target_class, K, opts)
        end
    end
end


