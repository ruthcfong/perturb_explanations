function lrp_layer = create_lrn_taylor_layer(norm_layer, epsilon)    
    assert(strcmp(norm_layer.type, 'lrn')  || strcmp(norm_layer.type, 'normalize'));
    lrp_layer = norm_layer;
    lrp_layer.type = 'custom';
    lrp_layer.epsilon = epsilon;
    lrp_layer.forward = @lrn_taylor_forward;
    lrp_layer.backward = @lrn_taylor_backward;
end
    
function res_ = lrn_taylor_forward(l, res, res_)
    res_.x = vl_nnormalize(res.x, l.param);
end

function res = lrn_taylor_backward(l, res, res_)
    N = l.param(1);
    kappa = l.param(2);
    alpha = l.param(3);
    beta = l.param(4);
    
    X = res.x;
    [H,W,D,NN] = size(X); % D = number of feature channels in X
    norm0 = (kappa + alpha*((X).^2)).^beta;
    
    Z = res.x./res_.x;
    norm1 = (Z.^(1/beta)).^(beta+1);
    
    taylor1 = zeros(size(X), 'like', X);
    
    for r=1:H
        for c=1:W
            for k=1:D
                for n=1:NN
                    x_k = X(r,c,k,n);
                    for j=max(1, k - floor((N-1)/2)):min(D, k + ceil((N-1)/2)) 
                        if j == k
                            continue;
                        end
                        x_j = X(r,c,j,n);
                        taylor1(r,c,k,n) = taylor1(r,c,k,n) + x_k*x_j/norm1(r,c,k,n);
                    end
                end
            end
        end
    end
    
    taylor1(isnan(taylor1)) = 0;
    t_approx = X./norm0 - 2*alpha*beta*taylor1;
    res.dzdx = t_approx.*res_.dzdx;
end