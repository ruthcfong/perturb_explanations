function res = pass_through_scaling_net(net, im, layernum, gradient)
    % check that network is ready for scaling
    num_layers = length(net.layers);
    for l=1:num_layers
        switch length(net.layers{l}.weights)
            case 0 % no weights for layer
                % do nothing
            case 2 % W and bias
                W = net.layers{l}.weights{1};                
                b = net.layers{l}.weights{2};
                % TODO remind to call create_scale_net
                assert(all(W(:) == 1) && all(b(:) == 0));
            otherwise % error handling: TODO
                % do nothing
        end
    end
    
    all_layers = net.layers;
    net.layers = net.layers(1:layernum);
    %opts = {};
    %opts.skipForward = true;
    %res = vl_simplenn(net, im, gradient, opts);
    res = vl_simplenn(net, im, gradient);
    
    % restore scaling net
    net.layers = all_layers;
end