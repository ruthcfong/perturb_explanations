function [aopc, diff_scores, x] = calculate_aopc(net, img, target_class, heatmap, varargin)
% function [apoc, diff_scores, x] = calculate_aopc(net, img, target_class,
% heatmap, varargin)
%
% img - original img (before normalization)
    img = imresize(img, net.meta.normalization.imageSize(1:2));
    opts.num_iters = 100;
    opts.num_perturbs = 10;
    opts.window_size = 9;
    opts.show_fig = false;
    
    opts = vl_argparse(opts, varargin);
        
    % truncate to pre-softmax network (TODO: make more robust by searching
    % for type 'softmax')
    tnet = truncate_net(net, 1, length(net.layers)-1);
    
    res = vl_simplenn(tnet, cnn_normalize(net.meta.normalization, img, 1));
    
    target_score = res(end).x(target_class);
    
    conv_layer = struct('name','conv1',...
        'type','conv',...
        'weights',{{ones([9 9 1], 'single'), zeros([1 1], 'single')}},...
        'pad',0, ...
        'stride', 1, ...
        'dilate',1, ...
        'opts',{{}});
    nnet = struct('layers', {{conv_layer}});

    res = vl_simplenn(nnet, heatmap);
    region_scores = res(end).x;
    
    perb_mask = zeros(size(heatmap), 'single');
    k = floor(opts.window_size-1)/2;
    
%     min_pixel = min(img(:));
%     max_pixel = max(img(:));
    
    x = img;
    
    diff_scores = zeros([1 opts.num_iters], 'single');
    
    counter = 1;
    i = 1;
    
    [~,sorted_idx] = sort(region_scores(:), 'descend'); 
    while counter <= opts.num_iters && i <= numel(region_scores)
        [r,c] = ind2sub(size(region_scores), sorted_idx(i));
        i = i + 1;
        
        % ensure non-overlapping regions
        if ~isempty(nonzeros(perb_mask(r:r+opts.window_size-1,c:c+opts.window_size-1)))
            continue;
        end
                
        x_ = repmat(x, 1,1,1, opts.num_perturbs);
%         rand_perbs = min_pixel + (max_pixel-min_pixel)...
%             *rand([opts.window_size,opts.window_size,3,opts.num_perturbs], 'single');
        rand_perbs = 255*rand([opts.window_size,opts.window_size,3,opts.num_perturbs], 'single');
        try
            x_(r:r+opts.window_size-1,c:c+opts.window_size-1,:,:) = rand_perbs;
        catch
            assert(false);
        end
%         x_in = bsxfun(@minus, single(x_), single(net.meta.normalization.averageImage));
        res = vl_simplenn(tnet, cnn_normalize(net.meta.normalization, x_, 1));
        rand_scores = res(end).x(1,1,target_class,:);
        
        avg_score = mean(rand_scores);
        diff_scores(counter) = target_score - avg_score;
        median_score = median(rand_scores);
        [~,median_i] = min(abs(rand_scores-median_score));
        x(r:r+opts.window_size-1,c:c+opts.window_size-1,:) = rand_perbs(:,:,:,median_i);
        
        perb_mask(r:r+opts.window_size-1,c:c+opts.window_size-1) = 1;
        counter = counter + 1;
    end
    
    aopc = cumsum(diff_scores)./(1:opts.num_iters);
    
    if opts.show_fig
        figure;
        subplot(1,2,1);
        imshow(x);
        subplot(1,2,2);
        plot(1:num_iters, aopc);
    end
end