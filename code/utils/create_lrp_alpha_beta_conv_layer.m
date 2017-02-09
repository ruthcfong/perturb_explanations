function lrp_layer = create_lrp_alpha_beta_conv_layer(conv_layer, alpha)    
    assert(strcmp(conv_layer.type, 'conv'));
    lrp_layer = conv_layer;
    lrp_layer.type = 'custom';
    lrp_layer.alpha = alpha;
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

function res = lrp_alpha_beta_backward(l, res, res_)
    alpha = l.alpha;
    beta = 1 - l.alpha;
    W = l.weights{1};
    %b = l.weights{2};
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
                X = zeros([h_in + 2*l.pad, w_in + 2*l.pad, size_in(3:end)], 'single');
                X(l.pad+1:l.pad+h_in,l.pad+1:l.pad+w_in, :, :) = res.x;
                relevance = zeros([h_in + 2*l.pad, w_in + 2*l.pad, size_in(3:end)], 'single');
            case 4
                X = zeros([h_in + sum(l.pad(1:2)), w_in + sum(l.pad(3:4)), size_in(3:end)], 'single');
                X(l.pad(1)+1:l.pad(1)+h_in,l.pad(3)+1:l.pad(3)+w_in, :, :) = res.x;
                relevance = zeros([h_in + sum(l.pad(1:2)), w_in + sum(l.pad(3:4)), size_in(3:end)], 'single');
            otherwise
                assert(false);
        end
        
    else
        X = double(res.x);
        %relevance = zeros(size(res.x), 'single');
        relevance = zeros(size(res.x), 'double');
    end
    next_relevance = res_.dzdx;
    %W(:,:,end+1,:) = b;

    for h=1:h_out
        for w=1:w_out
            x = X((h-1)*hstride+1:(h-1)*hstride+hf,(w-1)*wstride+1:(w-1)*wstride+wf,:,:); % [hf, wf, df, N]
            x = permute(repmat(x, [1 1 1 1 nf]), [1 2 3 5 4]); % [hf, wf, d_in, nf, N]
            %x(:,:,end+1,:) = ones(size(b),'like',x);
            %rr = repmat(reshape(R(:,i,j,:),[N 1 1 1 Nf]),[1 hf wf df 1]); % N x hf x wf x df x Nf
            %rr = repmat(reshape(next_relevance(h,w,:), [1 1 1 nf]), [hf, wf, df+1, 1]);
            try
                rr = repmat(reshape(next_relevance(h,w,:,:), [1 1 1 nf N]), [hf, wf, df, 1, 1]); % [hf, wf, df, nf, N]
            catch
                assert(false);
            end
            %Z = double(W .* x); % [hf, wf, df, nf]
            Z = double(bsxfun(@times, x, W)); % [nh, wf, df, nf N]
%             Zs = sum(sum(sum(Z,1),2),3); % [1 1 1 nf] (convolution summing here)
%             Zs = Zs + reshape(b, size(Zs));
%             Zs = Zs + l.epsilon*sign(Zs);
%             Zs = repmat(Zs, [hf, wf, df, 1]);
% 
%             zz = Z ./ Zs;

%             rr = repmat(reshape(next_relevance(h,w,:), [1 1 1 nf]), [hf, wf, df, 1]); % [hf, wf, df, nf]
%             rx = relevance((h-1)*hstride+1:(h-1)*hstride+hf,(w-1)*wstride+1:(w-1)*wstride+wf,:);
%             relevance((h-1)*hstride+1:(h-1)*hstride+hf,(w-1)*wstride+1:(w-1)*wstride+wf,:) = ...
%                 rx + sum(zz .* rr, 4);
            
            if ~(alpha == 0)
                Zp = Z .* (Z > 0);
                %Brp = b .* (b > 0);

                Zsp = sum(sum(sum(Zp,1),2),3);
                %Zsp = Zsp + reshape(Brp, size(Zsp)) ; % 1 x 1 x 1 x nf
                %Zsp = repmat(reshape(Zsp,[1 1 1 nf]),[hf wf df+1 1]); %  hf x wf x df x nf
                %Zsp = repmat(reshape(Zsp,[1 1 1 nf]),[hf wf df 1]); %  hf x wf x df x nf
                Zsp = repmat(reshape(Zsp,[1 1 1 nf N]),[hf wf df 1 1]); %  [hf x wf x df x nf x N]

                Ralpha = reshape(alpha .* sum(Zp ./ Zsp .* rr,4), [hf wf df N]);
            else
                Ralpha = 0;
            end

            if ~(beta == 0)
                Zn = Z .* (Z < 0);
                %Brn = b .* (b < 0);

                Zsn = sum(sum(sum(Zn,1),2),3);
                %Zsn = Zsn + reshape(Brn, size(Zsn)) ; % N x Nf
                %Zsn = repmat(reshape(Zsn,[1 1 1 nf]),[hf wf df 1]); % N x hf x wf x df x Nf
                Zsn = repmat(reshape(Zsn,[1 1 1 nf N]),[hf wf df 1 1]); % [hf x wf x df x Nf x N]

                Rbeta = reshape(beta .* sum(Zn ./ Zsn .* rr,4), [hf wf df N]);
            else
                Rbeta = 0;
            end
            
            rx = relevance((h-1)*hstride+1:(h-1)*hstride+hf,(w-1)*wstride+1:(w-1)*wstride+wf,:,:);
            relevance((h-1)*hstride+1:(h-1)*hstride+hf,(w-1)*wstride+1:(w-1)*wstride+wf,:,:) = ...
                rx + Ralpha + Rbeta;
            %rx = Rx(:,(i-1)*hstride+1:(i-1)*hstride+hf,(j-1)*wstride+1:(j-1)*wstride+wf,:); % N x hf x wf x df
            %Rx(:,(i-1)*hstride+1:(i-1)*hstride+hf,(j-1)*wstride+1:(j-1)*wstride+wf,:) = rx + Ralpha + Rbeta;
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
    res.dzdx = single(relevance);
    try
        assert(isequal(size(res.dzdx),size(res.x)));
    catch
        assert(false);
    end
end