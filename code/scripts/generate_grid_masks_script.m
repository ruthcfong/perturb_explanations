net = load('/home/ruthfong/packages/matconvnet/data/models/imagenet-vgg-verydeep-16.mat');
imdb_paths = load('/data/ruthfong/ILSVRC2012/val_imdb_paths.mat');
img_i = 1;
img = cnn_normalize(net.meta.normalization, imread(imdb_paths.images.paths{img_i}), 1);
target_class = imdb_paths.images.labels(img_i);
generate_grid_masks(net, img, target_class);