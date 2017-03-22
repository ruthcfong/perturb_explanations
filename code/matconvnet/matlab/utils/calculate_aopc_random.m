function [aopc, diff_scores, x] = calculate_aopc_random(net, img, target_class, varargin)
    opts.num_iters = 100;
    opts.num_perturbs = 10;
    opts.window_size = 9;
    opts.show_fig = false;
    
    opts = vl_argparse(opts, varargin);
        
    % truncate to pre-softmax network (TODO: make more robust by searching
    % for type 'softmax')
    tnet = truncate_net(net, 1, length(net.layers)-1);
    
    res = vl_simplenn(tnet, img);
    
    target_score = res(end).x(target_class);
    perb_mask = zeros(net.meta.normalization.imageSize(1:2), 'single');
    k = floor(opts.window_size-1)/2;
    
    min_pixel = min(img(:));
    max_pixel = max(img(:));
    
    x = img;
    
    diff_scores = zeros([1 opts.num_iters], 'single');
    
    counter = 1;
    i = 1;
    
    % randomly select indices to perturb
    sorted_idx = randperm(prod(net.meta.normalization.imageSize(1:2)));
    
    while counter <= opts.num_iters && i <= prod(net.meta.normalization.imageSize(1:2))
        [r,c] = ind2sub(net.meta.normalization.imageSize(1:2), sorted_idx(i));
        i = i + 1;
        
        % ensure region is sufficiently away from the border
        if r <= k || c <= k || r > net.meta.normalization.imageSize(1)-1-k ...
                || c > net.meta.normalization.imageSize(2)-1-k
            continue;
        end
        
        try
            % ensure non-overlapping regions
            if ~isempty(nonzeros(perb_mask(r-k:r+k,c-k:c+k)))
                continue;
            end
        catch
            warning('(%d,%d) is out of bounds', r, c);
            continue;
        end
        
        x_ = repmat(x, 1,1,1, opts.num_perturbs);
        rand_perbs = min_pixel + (max_pixel-min_pixel)*rand([2*k+1,2*k+1,3,opts.num_perturbs], 'single');
        x_(r-k:r+k,c-k:c+k,:,:) = rand_perbs;
        res = vl_simplenn(tnet, x_);
        rand_scores = res(end).x(1,1,target_class,:);
        
        avg_score = mean(rand_scores);
        diff_scores(counter) = target_score - avg_score;
        median_score = median(rand_scores);
        [~,median_i] = min(abs(rand_scores-median_score));
        x(r-k:r+k,c-k:c+k,:) = rand_perbs(:,:,:,median_i);
        
        perb_mask(r-k:r+k,c-k:c+k) = 1;
        counter = counter + 1;
    end
    
    aopc = cumsum(diff_scores)./(1:opts.num_iters);
    
    if opts.show_fig
        figure;
        subplot(1,2,1);
        imshow(normalize(x));
        subplot(1,2,2);
        plot(1:num_iters, aopc);
    end
end