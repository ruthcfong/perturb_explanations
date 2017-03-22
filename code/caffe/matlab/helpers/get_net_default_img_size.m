function net_size = get_net_default_img_size(net_type)
    switch net_type
        case 'alexnet'
            net_size = [227 227 3 1];
        case {'vgg16', 'googlenet'}
            net_size = [224 224 3 1];
        otherwise
            error('%s net type not supported', net_type);
    end         
end