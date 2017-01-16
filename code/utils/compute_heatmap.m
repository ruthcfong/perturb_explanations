function heatmap = compute_heatmap(net, img, target_class, heatmap_type, norm_deg, varargin)
    opts.lrp_epsilon = 100;
%     opts.lrp_alpha = 2;
%     opts.lrp_beta = 1;
    opts = vl_argparse(opts, varargin);

    
%     if ~isequal(net.layers{end}.type, 'softmaxloss')
%         net.layers{end+1} = struct('type', 'softmaxloss') ;
%     end
%     net.layers{end}.class = target_class;

    % truncate to pre-softmax network (TODO: make more robust by searching
    % for type 'softmax')
    net = truncate_net(net, 1, length(net.layers)-1);   

    % modify network based on heatmap technique
    switch heatmap_type
        case 'saliency' % Simonyan et al., 2014 (Gradient-based Saliency)
            % do nothing (no modification necessary)
        case 'deconvnet' % Zeiler & Fergus, 2014 (Deconvolutional Network)
            for i=1:length(net.layers)
                switch net.layers{i}.type
                    case 'relu'
                        net.layers{i}.type = 'relu_deconvnet';
                    case 'normalize'
                        net.layers{i}.type = 'normalize_nobackprop';
                    case 'lrn'
                        net.layers{i}.type = 'lrn_nobackprop';
                    otherwise
                        continue;
                end
            end
        case 'guided_backprop'  % Springenberg et al., 2015, Mahendran and Vedaldi, 2015 (DeSalNet/Guided Backprop)
            for i=1:length(net.layers)
                switch net.layers{i}.type
                    case 'relu'
                        net.layers{i}.type = 'relu_eccv16';
                    case 'normalize'
                        net.layers{i}.type = 'normalize_nobackprop';
                    case 'lrn'
                        net.layers{i}.type = 'lrn_nobackprop';
                    otherwise
                        continue;
                end
            end
        case 'lrp_epsilon'
            for i=1:length(net.layers)
                switch net.layers{i}.type
                    case 'conv'
                        net.layers{i} = create_lrp_epsilon_conv_layer(...
                            net.layers{i}, opts.lrp_epsilon);
                    case 'normalize'
                        net.layers{i}.type = 'normalize_nobackprop';
                    case 'lrn'
                        net.layers{i}.type = 'lrn_nobackprop';

                    otherwise
                        continue;
                end
            end
        case 'lrp_alpha_beta'
            assert(false);
        otherwise
            assert(false);
    end
    
    res = vl_simplenn(net, img);
    gradient = zeros(size(res(end).x), 'single');
    gradient(target_class) = 1;
    res = vl_simplenn(net, img, gradient);
    
    if isinf(norm_deg)
        if norm_deg == inf
            heatmap = max(abs(res(1).dzdx),[],3);
        else
            heatmap = min(abs(res(1).dzdx),[],3);
        end
    else
        if norm_deg == 0
            heatmap = res(1).dzdx;
        else
            heatmap = sum(abs(res(1).dzdx).^ norm_deg, 3).^(1/norm_deg);
        end
    end
end