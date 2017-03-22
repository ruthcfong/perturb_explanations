function scores = get_region_score_curve(net, img, target_class, heatmap, varargin)
    opts.quantile_range = 0.99:-0.01:0.5;
    opts.window_size = 9;
    
    opts = vl_argparse(opts, varargin);
    
    % truncate to pre-softmax network (TODO: make more robust by searching
    % for type 'softmax')
    tnet = truncate_net(net, 1, length(net.layers)-1);
    
    res = vl_simplenn(tnet, img);
    
    %target_score = res(end).x(target_class);
    
    conv_layer = struct('name','conv1',...
        'type','conv',...
        'weights',{{ones([opts.window_size opts.window_size 1], 'single'), zeros([1 1], 'single')}},...
        'pad',0, ...
        'stride', 1, ...
        'dilate',1, ...
        'opts',{{}});
    nnet = struct('layers', {{conv_layer}});

    res = vl_simplenn(nnet, heatmap);
    region_scores = res(end).x;
    %figure; imshow(normalize(bsxfun(@times, img, clip_map(imresize(...
    %    region_scores > quantile(region_scores(:),0.95), net.meta.normalization.imageSize(1:2))))));
    scores = zeros([1 length(opts.quantile_range)], 'single');
    
    for i=1:length(opts.quantile_range)
        qt = opts.quantile_range(i);
        threshold = quantile(region_scores(:), qt);
        mask = clip_map(imresize(region_scores > threshold, net.meta.normalization.imageSize(1:2)));

        res = vl_simplenn(tnet, bsxfun(@times, img, mask));
        
        scores(i) = res(end).x(target_class);
    end
end