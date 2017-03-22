function lrp_layer = create_lrp_alpha_beta_conv_layer(conv_layer, alpha, include_bias)    
    assert(strcmp(conv_layer.type, 'conv'));
    lrp_layer = conv_layer;
    lrp_layer.type = 'custom';
    lrp_layer.alpha = alpha;
    lrp_layer.include_bias = include_bias;
    lrp_layer.forward = @lrp_alpha_beta_forward;
    lrp_layer.backward = @lrp_alpha_beta_backward;
end
    
function res_ = lrp_alpha_beta_forward(l, res, res_)
    res_.x = vl_nnconv(res.x, l.weights{1}, l.weights{2}, ...
        'pad', l.pad, ...
        'stride', l.stride, ...
        'dilate', l.dilate, ...
        l.opts{:});
end

function res = lrp_alpha_beta_backward_quick(l, res, res_)
    alpha = l.alpha;
    beta = 1 - alpha;
    W = l.weights{1};
    Z = vl_nnconv(res.x, W, [], ...
        'pad', l.pad, ...
        'stride', l.stride, ...
        'dilate', l.dilate, ...
        l.opts{:});
    [R,~,~] = vl_nnconv(res.x, W, [], res_.dzdx, ...
            'pad', l.pad, ...
            'stride', l.stride, ...
            'dilate', l.dilate, ...
            l.opts{:});

    if ~(alpha == 0)
        Zp = Z .* (Z > 0);
        Yp = 1 ./ Zp;
        if ~isempty(find(isnan(Yp),1))
            warning('Some NaNs in Yp at layer %s', l.name);
            Yp(isnan(Yp)) = 0;
        end
        [Np,~,~] = vl_nnconv(res.x, W, [], Yp, ...
            'pad', l.pad, ...
            'stride', l.stride, ...
            'dilate', l.dilate, ...
            l.opts{:});
    else
        Zp = 0;
    end
    
    if ~(beta == 0)
        Zn = Z .* (Z < 0);
        Yn = res_.dzdx ./ Zn;
        if ~isempty(find(isnan(Yn),1))
            warning('Some NaNs in Yn at layer %s', l.name);
            Yn(isnan(Yn)) = 0;
        end
        [Rn,~,~] = vl_nnconv(res.x, W, [], Yn, ...
            'pad', l.pad, ...
            'stride', l.stride, ...
            'dilate', l.dilate, ...
            l.opts{:});
    else
        Zn = 0;
    end
    
    res.dzdx = R .* (alpha*Zp.*res.x);
end

function res = lrp_alpha_beta_backward(l, res, res_)
    alpha = l.alpha;
    beta = 1 - l.alpha;
    W = l.weights{1};
    b = l.weights{2};
    hstride = l.stride(1);
    wstride = l.stride(2);

    [h_in,w_in,d_in,N] = size(res.x);
    size_in = size(res.x);
    [h_out,w_out,~,~] = size(res_.x);
    [hf, wf, df, nf] = size(W);
    
    % deal with parallel streams scenario
    if d_in ~= df
        assert(mod(d_in, df) == 0);
        W = repmat(W, [1 1 d_in/df 1 N]);
        [hf, wf, df, nf, ~] = size(W);
    end

    % add padding if necessary
    has_padding = sum(l.pad) > 0;
    if has_padding
        pad_dims = length(l.pad);
        switch pad_dims
            case 1
                X = zeros([h_in + 2*l.pad, w_in + 2*l.pad, size_in(3:end)], 'like', res.x);
                X(l.pad+1:l.pad+h_in,l.pad+1:l.pad+w_in, :, :) = res.x;
                relevance = zeros([h_in + 2*l.pad, w_in + 2*l.pad, size_in(3:end)], 'like', res.x);
            case 4
                X = zeros([h_in + sum(l.pad(1:2)), w_in + sum(l.pad(3:4)), size_in(3:end)], 'like', res.x);
                X(l.pad(1)+1:l.pad(1)+h_in,l.pad(3)+1:l.pad(3)+w_in, :, :) = res.x;
                relevance = zeros([h_in + sum(l.pad(1:2)), w_in + sum(l.pad(3:4)), size_in(3:end)], 'like', res.x);
            otherwise
                assert(false);
        end
        
    else
        X = res.x;
        relevance = zeros(size(X), 'like', X);
    end
    next_relevance = res_.dzdx;

    for h=1:h_out
        for w=1:w_out
            x = X((h-1)*hstride+1:(h-1)*hstride+hf,(w-1)*wstride+1:(w-1)*wstride+wf,:,:); % [hf, wf, df, N]
            x = permute(repmat(x, [1 1 1 1 nf]), [1 2 3 5 4]); % [hf, wf, d_in, nf, N]
            rr = repmat(reshape(next_relevance(h,w,:,:), [1 1 1 nf N]), [hf, wf, df, 1, 1]); % [hf, wf, df, nf, N]
            Z = bsxfun(@times, x, W); % [nh, wf, df, nf N]
            
            if ~(alpha == 0)
                Zp = Z .* (Z > 0);

                Zsp = sum(sum(sum(Zp,1),2),3);
                if l.include_bias
                    Brp = b .* (b > 0);
                    size_Zsp = size(Zsp);
                    Zsp = bsxfun(@plus, Zsp, reshape(Brp, size_Zsp(1:4)));
                    %Zsp = Zsp + reshape(Brp, size(Zsp)) ; % 1 x 1 x 1 x nf
                end

                Zsp = repmat(reshape(Zsp,[1 1 1 nf N]),[hf wf df 1 1]); %  [hf x wf x df x nf x N]

                Ralpha = reshape(alpha .* sum(Zp ./ Zsp .* rr,4), [hf wf df N]);
            else
                Ralpha = 0;
            end

            if ~(beta == 0)
                Zn = Z .* (Z < 0);

                Zsn = sum(sum(sum(Zn,1),2),3);
                
                if l.include_bias
                    Brn = b .* (b < 0);
                    size_Zsn = size(Zsn);
                    Zsn = bsxfun(@plus, Zsn, reshape(Brn, size_Zsn(1:4)));
                    %Zsn = Zsn + reshape(Brn, size(Zsn)) ; % N x Nf
                end

                Zsn = repmat(reshape(Zsn,[1 1 1 nf N]),[hf wf df 1 1]); % [hf x wf x df x Nf x N]

                Rbeta = reshape(beta .* sum(Zn ./ Zsn .* rr,4), [hf wf df N]);
            else
                Rbeta = 0;
            end
            
            rx = relevance((h-1)*hstride+1:(h-1)*hstride+hf,(w-1)*wstride+1:(w-1)*wstride+wf,:,:);
            relevance((h-1)*hstride+1:(h-1)*hstride+hf,(w-1)*wstride+1:(w-1)*wstride+wf,:,:) = ...
                rx + Ralpha + Rbeta;
        end
    end
    
    if has_padding
        switch pad_dims
            case 1
                 relevance = relevance(l.pad+1:l.pad+h_in, l.pad+1:l.pad+w_in, :, :);
            case 4
                 relevance = relevance(l.pad(1)+1:l.pad(1)+h_in, l.pad(3)+1:l.pad(3)+w_in, :, :);
            otherwise
                assert(false);
        end
    end
    
    res.dzdx = relevance;
    assert(isequal(size(res.dzdx),size(res.x)));
end