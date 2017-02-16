function max_diff_layer = create_max_diff_layer()    
    max_diff_layer = struct();
    max_diff_layer.type = 'custom';
    max_diff_layer.forward = @max_diff_forward;
    max_diff_layer.backward = @max_diff_backward;
end
    
function res_ = max_diff_forward(l, res, res_)
    res_.x = max(res.x)-min(res.x);
end

function res = max_diff_backward(l, res, res_)
    [~,max_idx] = max(res.x);
    [~,min_idx] = min(res.x);
    max_der = zeros(size(res.x), 'like', res.x);
    min_der = zeros(size(res.x), 'like', res.x);
    max_der(max_idx) = res_.dzdx;
    min_der(min_idx) = res_.dzdx;
    res.dzdx = max_der - min_der;
end