function [heatmap, res] = compute_heatmap(net, imgs, target_classes, heatmap_type, norm_deg, varargin)
    opts.lrp_epsilon = 100;
    opts.lrp_alpha = 1;
    opts.layer_name = '';
    opts.gpu = NaN;
    opts = vl_argparse(opts, varargin);
    
    isDag = isfield(net, 'params') || isprop(net, 'params');

    if isDag
        [heatmap, res] = compute_heatmap_dag(net, imgs, target_classes, heatmap_type, norm_deg, opts);
    else
        [heatmap, res] = compute_heatmap_simplenn(net, imgs, target_classes, heatmap_type, norm_deg, opts);
    end
end

function [heatmap, net] = compute_heatmap_dag(net, imgs, target_classes, heatmap_type, norm_deg, opts)
    net = copy(dagnn.DagNN.loadobj(net));
    net.reset();
    softmax_i = find(cellfun(@(x) isa(x, 'dagnn.SoftMax'), {net.layers.block}));
    assert(length(softmax_i) <= 1);
    if ~isempty(softmax_i)
        net.removeLayer(net.layers(softmax_i).name);
    end
    net.mode = 'test';
    %net.vars(1).precious = 1; % save data value & derivs
    %net.conserveMemory = 0;
    
    % modify network based on heatmap technique
    switch heatmap_type
        case 'saliency' % Simonyan et al., 2014 (Gradient-based Saliency)
            % do nothing (no modification necessary)
        case 'deconvnet' % Zeiler & Fergus, 2014 (Deconvolutional Network)
            interested_idx = cellfun(@(x) isa(x, 'dagnn.ReLU') || isa(x,'dagnn.LRN') ...
                , {net.layers.block});
            interested_names = {net.layers(interested_idx).name};
            for i=1:length(interested_names)
                l = net.getLayer(interested_names{i});
                switch class(l.block)
                    case 'dagnn.ReLU'
                        relu_block = ReLU_custom('custom_type', 'deconvnet', ...
                            'useShortCircuit', l.block.useShortCircuit, ...
                            'leak', l.block.leak, ...
                            'opts', l.block.opts);
                        net.removeLayer(l.name);
                        net.addLayer(l.name, relu_block, l.inputs, ...
                            l.outputs, l.params);
                    case 'dagnn.LRN'
                        lrn_block = LRN_custom('custom_type', 'nobackprop', ...
                            'param', l.block.param);
                        net.removeLayer(interested_names{i});
                        net.addLayer(l.name, lrn_block, l.inputs, ...
                            l.outputs, l.params);
                    otherwise
                        continue;
                end
            end
        case 'guided_backprop'  % Springenberg et al., 2015, Mahendran and Vedaldi, 2015 (DeSalNet/Guided Backprop)
            interested_idx = cellfun(@(x) isa(x, 'dagnn.ReLU') || isa(x,'dagnn.LRN') ...
                , {net.layers.block});
            interested_names = {net.layers(interested_idx).name};
            for i=1:length(interested_names)
                l = net.getLayer(interested_names{i});
                switch class(l.block)
                    case 'dagnn.ReLU'
                        relu_block = ReLU_custom('custom_type', 'guidedbackprop', ...
                            'useShortCircuit', l.block.useShortCircuit, ...
                            'leak', l.block.leak, ...
                            'opts', l.block.opts);
                        net.removeLayer(l.name);
                        net.addLayer(l.name, relu_block, l.inputs, ...
                            l.outputs, l.params);
                    case 'dagnn.LRN'
                        lrn_block = LRN_custom('custom_type', 'nobackprop', ...
                            'param', l.block.param);
                        net.removeLayer(interested_names{i});
                        net.addLayer(l.name, lrn_block, l.inputs, ...
                            l.outputs, l.params);
                    otherwise
                        continue;
                end
            end
        case 'lrp_epsilon'
            interested_idx = cellfun(@(x) isa(x, 'dagnn.Conv') || isa(x,'dagnn.LRN') ...
                , {net.layers.block});
            interested_names = {net.layers(interested_idx).name};
            for i=1:length(interested_names)
                l = net.getLayer(interested_names{i});
                switch class(l.block)
                    case 'dagnn.Conv'
                        conv_block = Conv_LRP_epsilon('epsilon', opts.lrp_epsilon, ...
                            'size', l.block.size, ...
                            'hasBias', l.block.hasBias, ...
                            'opts', l.block.opts, ...
                            'pad', l.block.pad, ...
                            'stride', l.block.stride, ...
                            'dilate', l.block.dilate);
                        params = net.params(l.paramIndexes);
                        net.removeLayer(l.name);
                        net.addLayer(l.name, conv_block, l.inputs, ...
                            l.outputs, l.params);
                        net.params(net.getLayer(l.name).paramIndexes) = params;
                    case 'dagnn.LRN'
                        lrn_block = LRN_custom('custom_type', 'nobackprop', ...
                            'param', l.block.param);
                        net.removeLayer(interested_names{i});
                        net.addLayer(l.name, lrn_block, l.inputs, ...
                            l.outputs, l.params);
                    otherwise
                        continue;
                end
            end
        otherwise % TODO: implement lrp_alpha_beta
            error('Heatmap type %s is not implemented', heatmap_type);
    end
    
    net.rebuild();
    %net = dagnn.DagNN.loadobj(net);
    
    order = net.getLayerExecutionOrder();
    
    if ~isempty(opts.layer_name)
        layer_i = find(cellfun(@(x) strcmp(x, opts.layer_name), {net.layers.name}));
        assert(length(layer_i) == 1);
        assert(length(net.layers(layer_i).outputIndexes) == 1);
        net.vars(net.layers(layer_i).outputIndexes).precious = 1;
    else
        net.vars(net.layers(order(1)).inputIndexes).precious = 1;
        layer_i = order(1);
    end
    
    input_var_i = net.layers(order(1)).inputIndexes;
    output_var_i = net.layers(order(end)).outputIndexes;
    if ndims(imgs) <= 3
        inputs = {net.vars(input_var_i).name, imgs};
        net.eval(inputs);
        gradient = zeros(size(net.vars(output_var_i).value), 'single');
        gradient(target_classes) = 1;
        derOutputs = {net.vars(output_var_i).name, gradient};
        net.eval(inputs, derOutputs);
    else
        assert(ndims(imgs) == 4);
        inputs = {net.vars(input_var_i).name, imgs(:,:,:,1)};
        net.eval(inputs);
        gradient = zeros([size(net.vars(output_var_i).value) size(imgs, 4)], 'single');
        if ~isnan(opts.gpu)
            g = gpuDevice(opts.gpu+1);
            net.move('gpu');
            imgs = gpuArray(imgs);
            gradient = gpuArray(gradient);
            target_classes = gpuArray(target_classes);
        end
        for i=1:size(imgs,4)
            gradient(1,1,target_classes(i),i) = 1;
        end
        inputs = {net.vars(input_var_i).name, imgs};
        derOutputs = {net.vars(output_var_i).name, gradient};
        net.eval(inputs, derOutputs);
    end
    
    volume = net.vars(net.layers(layer_i).outputIndexes).der;

    if isinf(norm_deg)
        if norm_deg == inf
            heatmap = max(abs(volume),[],3);
        else
            heatmap = min(abs(volume),[],3);
        end
    else
        if norm_deg == 0
            heatmap = volume;
        elseif norm_deg == -1
            heatmap = sum(volume, 3);
        else
            heatmap = sum(abs(volume).^ norm_deg, 3).^(1/norm_deg);
        end
    end
    
    heatmap = gather(heatmap);
end

function [heatmap, res] = compute_heatmap_simplenn(net, imgs, target_classes, heatmap_type, norm_deg, opts)
    % truncate to pre-softmax network (assuming that this is the last
    % layer)
    assert(strcmp(net.layers{end}.type, 'softmax'));
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
            for i=1:length(net.layers)
                switch net.layers{i}.type
                    case 'conv'
                        net.layers{i} = create_lrp_alpha_beta_conv_layer(...
                            net.layers{i}, opts.lrp_alpha);
                    case 'normalize'
                        net.layers{i}.type = 'normalize_nobackprop';
                    case 'lrn'
                        net.layers{i}.type = 'lrn_nobackprop';

                    otherwise
                        continue;
                end
            end
        otherwise
            assert(false);
    end
    
    if ndims(imgs) <= 3
        res = vl_simplenn(net, imgs);
        gradient = zeros(size(res(end).x), 'single');
        gradient(target_classes) = 1;
        res = vl_simplenn(net, imgs, gradient);
    else
        res = vl_simplenn(net, imgs(:,:,:,1));
        gradient = zeros([size(res(end).x) size(imgs,4)], 'single');
        if ~isnan(opts.gpu)
            g = gpuDevice(opts.gpu+1);
            net = vl_simplenn_move(net, 'gpu');
            imgs = gpuArray(imgs);
            gradient = gpuArray(gradient);
            target_classes = gpuArray(target_classes);
        end
        for i=1:size(imgs,4)
            gradient(1,1,target_classes(i),i) = 1;
        end
        res = vl_simplenn(net, imgs, gradient);
    end
    
    if ~isempty(opts.layer_name)
        layer_i = find(cellfun(@(x) strcmp(x.name, opts.layer_name), net.layers));
        assert(length(layer_i) == 1);
        volume = res(layer_i+1).dzdx;
    else
        volume = res(1).dzdx;
    end

    if isinf(norm_deg)
        if norm_deg == Inf
            heatmap = max(abs(volume),[],3);
        else
            heatmap = min(abs(volume),[],3);
        end
    else
        if norm_deg == 0
            heatmap = volume;
        elseif norm_deg == -1
            heatmap = sum(volume, 3);
        else
            heatmap = sum(abs(volume).^ norm_deg, 3).^(1/norm_deg);
        end
    end
    
    heatmap = gather(heatmap);
end