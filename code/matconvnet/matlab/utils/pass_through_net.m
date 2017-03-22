function res = pass_through_net(net, im, layernum, gradient)    
    all_layers = net.layers;
    net.layers = net.layers(1:layernum);
    res = vl_simplenn(net, im, gradient);
    
    % restore net
    net.layers = all_layers;
end