function [start_coord,end_coord] = get_patch_coordinates(net, layer_in, ...
    layer_out, r_in, c_in)
    debug = true;
    
    assert(layer_out < layer_in);
    
    stride_rf = -1;
    pad_rf = -1;
    field_size_rf = -1;
    r_center = -1;
    c_center = -1;
    
    seen_first_rf_layer = false;
    for l=layer_in:-1:layer_out
        layer = net.layers{l};
        type_l = layer.type;
        switch type_l
            case 'pool'
                % check that stride, pad, and field size are equal
                % in all directions (unnecessarily tight but simplifying
                % constraint)
                assert(all(layer.pool == layer.pool(1)));
                assert(all(layer.pad == layer.pad(1)));
                assert(all(layer.stride == layer.stride(1)));
                
                field_size = layer.pool(1);
                pad = layer.pad(1); 
                stride = layer.stride(1);
            case 'conv'
                weights_size = size(layer.weights{1});
                
                assert(weights_size(1) == weights_size(2));
                assert(all(layer.pad == layer.pad(1)));
                assert(all(layer.stride == layer.stride(1)));
                
                field_size = weights_size(1);
                pad = layer.pad(1);
                stride = layer.stride(1);
            otherwise
                continue
        end
        
        r_end = (r_end - 1)*stride + field_size - 2*pad;
        c_end = (c_end - 1)*stride + field_size - 2*pad;
        r_start = (r_start - 1)*stride + 1;
        c_start = (c_start - 1)*stride + 1;
        
        if ~seen_first_rf_layer
            stride_rf = stride;
            pad_rf = pad;
            field_size_rf = field_size;
            r_center = r_in;
            c_center = c_in;
        else
            pad_rf = stride_rf * (pad - 1) + pad_rf;
            field_size_rf = stride_rf(field_size - 1) + field_size_rf;
            stride_rf = stride_rf * stride;
            r_center = 
        end

        
        if debug
            disp(fprintf('layer %d: start = (%.f, %.f), end = (%.f, %.f)', ...
                l, r_start, c_start, r_end, c_end));
        end
    end
    
      % operator applied to the input image
%   info.receptiveFieldSize(1:2,l) = 1 + ...
%       sum(cumprod([[1;1], info.stride(1:2,1:l-1)],2) .* ...
%           (info.support(1:2,1:l)-1),2) ;
%   info.receptiveFieldOffset(1:2,l) = 1 + ...
%       sum(cumprod([[1;1], info.stride(1:2,1:l-1)],2) .* ...
%           ((info.support(1:2,1:l)-1)/2 - info.pad([1 3],1:l)),2) ;
%   info.receptiveFieldStride = cumprod(info.stride,2) ;
% 
  
    start_coord = [r_start c_start];
    end_coord = [r_end c_end];
    
end