function net = create_scaling_net(net_path)
    net = load(net_path);
    num_layers = length(net.layers);
    for l=1:num_layers
        switch net.layers{l}.type
            case 'pool'
                net.layers{l}.method = 'avg';
            case 'conv' % W and bias
                W_old = net.layers{l}.weights{1};
                W_new = ones(size(W_old),'single');
                b_old = net.layers{l}.weights{2};
                b_new = zeros(size(b_old),'single');
                net.layers{l}.weights = {W_new b_new};
        end
    end
end