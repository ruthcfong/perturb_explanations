function layer = create_excitation_backprop_conv_layer(conv_layer)    
    assert(strcmp(conv_layer.type, 'conv'));
    layer = conv_layer;
    layer.type = 'custom';
    layer.forward = @excitation_backprop_forward;
    layer.backward = @excitation_backprop_backward;
end
    
function res_ = excitation_backprop_forward(l, res, res_)
% Same forward function as normal conv layer

    res_.x = vl_nnconv(res.x, l.weights{1}, l.weights{2}, ...
        'pad', l.pad, ...
        'stride', l.stride, ...
        'dilate', l.dilate, ...
        l.opts{:});
end

function res = excitation_backprop_backward(l, res, res_)
% Implemented according to Alg. 1 description in Zhang et al., 2016
% and based on Caffe implementation here: 
% https://github.com/jimmie33/Caffe-ExcitationBP/blob/master/src/caffe/layers/conv_layer.cpp

    Wp = l.weights{1};
    Wp(Wp < 0) = 0;
    
    X = vl_nnconv(res.x, Wp, [], ...
        'pad', l.pad, ...
        'stride', l.stride, ...
        'dilate', l.dilate, ...
        l.opts{:});
    
    Y = res_.dzdx ./ X;
    if length(find(isnan(Y))) > 0
        warning(sprintf('Some NaNs in layer %s', l.name));
        Y(isnan(Y)) = 0;
    end
    [Z,~,~] = vl_nnconv(res.x, Wp, [], Y, ...
        'pad', l.pad, ...
        'stride', l.stride, ...
        'dilate', l.dilate, ...
        l.opts{:});
    
    res.dzdx = res.x .* Z;
end