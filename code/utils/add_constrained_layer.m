function net = add_constrained_layer(net, layernum, num_top_filters, constraint_type)        
%     res_prev = vl_simplenn(net, im);
%     res_curr = struct('x',[],'dzdx', [], 'dzdw',{}, 'aux', [], 'stats', [], ...
%         'time', 0, 'backwardTime', 0);
%     res_curr.x = 
    
    % define constrained layer
    ly.type = 'custom' ;
    ly.num_top_filters = num_top_filters;
    ly.constraint = constraint_type ;
    ly.forward = @constraint_forward ;
    ly.backward = @constraint_backward ;

    net.layers = {net.layers{1:layernum} ly net.layers{layernum+1:end}};
end

function res_ = constraint_forward(ly, res, res_)
    res_.x = res.x;
    switch ly.constraint
        case 'mask'
            switch ndims(res.x)
                case 3
                    size_x = size(res.x);
                    x = reshape(res.x, [prod(size_x(1:2)), size_x(3)]);
                    norm_2 = sqrt(sum(abs(x).^2,1));
                    [~, sorted_idx] = sort(norm_2);
                    y = zeros(size_x);
                    top_idx = sorted_idx(end-ly.num_top_filters+1:end);
                    y(:,:,top_idx) = res.x(:,:,top_idx);
                    res_.x = y;
                case 4
                    size_x = size(res.x);
                    x = reshape(res.x, [prod(size_x(1:2)), size_x(3), size_x(4)]);
                    norm_2 = sqrt(sum(abs(x).^2,1));
                    [~, sorted_idx] = sort(norm_2,2);
                    y = zeros(size_x, 'single');
                    top_idx = squeeze(sorted_idx(:,end-ly.num_top_filters+1:end,:));
                    for d=1:size_x(4) % Q: Is there a way to do this without a loop?
                        y(:,:,top_idx(:,d),d) = res.x(:,:,top_idx(:,d),d);
                    end
                    res_.x = y;
                otherwise
                    assert(false);
            end
        case 'binary'
            assert(false);
        case 'quantize'
            assert(false);
        otherwise
            assert(false);
    end
end

function res = constraint_backward(ly, res, res_)
    res.dzdx = res_.dzdx; % not sure if this would work
end