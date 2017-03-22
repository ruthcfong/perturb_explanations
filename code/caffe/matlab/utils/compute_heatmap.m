function heatmap = compute_heatmap(net, imgs, target_classes, heatmap_type, varargin)
    softmax_i = find(cellfun(@(l) strcmp(l, 'prob'), net.layer_names));
    if ~isempty(softmax_i)
        opts.start_layer = net.layer_names{softmax_i-1};
    else
        opts.start_layer = net.layer_names{end};
    end
    opts.end_layer = net.layer_names{1};
    opts.gpu = NaN;
    opts.norm_deg = NaN;
    
    opts = vl_argparse(opts, varargin);
    
    start_ind = find(cellfun(@(l) strcmp(l, opts.start_layer), net.layer_names));
    end_ind = find(cellfun(@(l) strcmp(l, opts.end_layer), net.layer_names));
    assert(~isempty(start_ind) && ~isempty(end_ind));
    start_ind = start_ind - 1;
    end_ind = end_ind - 1;
    
    if isnan(opts.gpu)
        switch heatmap_type
            case 'saliency' % Simonyan et al., 2014 (Gradient-based Saliency)
                apply_mode = @() caffe.set_mode_cpu();
            case 'guided_backprop'  % Springenberg et al., 2015, Mahendran and Vedaldi, 2015 (DeSalNet/Guided Backprop)
                apply_mode = @() caffe.set_mode_dc_cpu();
            case 'excitation_backprop'
                apply_mode = @() caffe.set_mode_eb_cpu();
            case {'deconvnet','lrp_epsilon','lrp_alpha_beta'}
                error('%s heatmap type is not supported in caffe; it is implemented in matconvnet');
            otherwise
                error('%s heatmap type is not supported.');
        end
        revert_mode = @() caffe.set_mode_cpu();
    else
        caffe.set_device(opts.gpu);
        switch heatmap_type
            case 'saliency'
                apply_mode = @() caffe.set_mode_gpu();
            case 'guided_backprop'
                apply_mode = @() caffe.set_mode_dc_gpu();
            case 'excitation_backprop'
                apply_mode = @() caffe.set_mode_eb_gpu();
            case {'deconvnet','lrp_epsilon','lrp_alpha_beta'}
                error('%s heatmap type is not supported in caffe; it is implemented in matconvnet');
            otherwise
                error('%s heatmap type is not supported.');
        end
        revert_mode = @() caffe.set_mode_gpu();
    end
    
    revert_mode();
    
    if isnan(opts.norm_deg)
        switch heatmap_type
            case {'saliency', 'guided_backprop'}
                opts.norm_deg = Inf;
            case 'excitation_backprop'
                opts.norm_deg = -1;
        end
    end
    
    if ndims(imgs) <= 3
        assert(ndims(imgs) == 3);
        net.blobs('data').reshape([size(imgs) 1]);
    else
        net.blobs('data').reshape(size(imgs));
    end
    
    net.blobs('data').set_data(imgs);
    net.forward_from_to(0, length(net.layer_names)-1);
    %net.forward_prefilled();
    
    diff = zeros(net.blobs(opts.start_layer).shape);
    if ndims(imgs) <= 3
        diff(target_classes) = 1;
    else
        for i=1:size(imgs, 4)
            diff(target_classes(i),i) = 1;
        end
    end
    %net.blobs('prob').set_diff(diff);
    %net.backward_prefilled();
    net.blobs(opts.start_layer).set_diff(diff);
    apply_mode();
    net.backward_from_to(start_ind, end_ind);
    revert_mode();
    volume = net.blobs(opts.end_layer).get_diff();
    
    if isinf(opts.norm_deg)
        if opts.norm_deg == Inf
            heatmap = max(abs(volume),[],3);
        else
            heatmap = min(abs(volume),[],3);
        end
    else
        if opts.norm_deg == 0
            heatmap = volume;
        elseif opts.norm_deg == -1
            heatmap = sum(volume, 3);
        else
            heatmap = sum(abs(volume).^ opts.norm_deg, 3).^(1/opts.norm_deg);
        end
    end
end