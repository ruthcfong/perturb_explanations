net_type = 'googlenet';
caffe_model_dir = '/home/ruthfong/packages/caffe/models';
switch net_type
    case 'alexnet'
        model_dir = fullfile(caffe_model_dir, 'bvlc_reference_caffenet');
        net_model = fullfile(model_dir,'deploy_force_backward.prototxt');
        net_weights = fullfile(model_dir,'bvlc_reference_caffenet.caffemodel');
    case 'vgg16'
        model_dir = fullfile(caffe_model_dir, 'vgg16');
        net_model = fullfile(model_dir, 'VGG_ILSVRC_16_layers_deploy_force_backward.prototxt');
        net_weights = fullfile(model_dir, 'VGG_ILSVRC_16_layers.caffemodel');
    case 'googlenet'
        model_dir = fullfile(caffe_model_dir, 'bvlc_googlenet');
        net_model = fullfile(model_dir, 'deploy_force_backward.prototxt');
        net_weights = fullfile(model_dir, 'bvlc_googlenet.caffemodel');
    otherwise
        error('%s net type is not supported', net_type);
end
net = caffe.Net(net_model, net_weights, 'test');

mean_data = caffe.io.read_mean('/home/ruthfong/packages/caffe/data/ilsvrc12/imagenet_mean.binaryproto');
im = caffe.io.load_image('~/packages/caffe/examples/images/cat.jpg');
im_ = normalize_img(im, mean_data, get_net_default_img_size(net_type));
target_class = 282; % tabby cat

opts = struct();
opts.gpu = 0;
opts.end_layer = 'pool2/3x3_s2';
opts.norm_deg = -1;
heatmap = convert_im_order(compute_heatmap(net, im_, target_class, 'excitation_backprop', opts));

caffe.reset_all();