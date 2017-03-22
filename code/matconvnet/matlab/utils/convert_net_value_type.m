function net = convert_net_value_type(net, type_fh)
    num_layers = length(net.layers);
    for i=1:num_layers
        l = net.layers{i};
        switch l.type
            case 'conv'
                net.layers{i}.weights{1} = type_fh(net.layers{i}.weights{1});
                net.layers{i}.weights{2} = type_fh(net.layers{i}.weights{2});
            otherwise
                continue;
        end
    end
end