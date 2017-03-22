function [res, heatmap] = getbb_from_heatmap(heatmap, alpha, resize, varargin)
    opts.thres_first = true;
    opts = vl_argparse(opts, varargin);
    
    if ~opts.thres_first && ~isempty(resize)
        heatmap = imresize(heatmap, resize);
    end
    
    threshold = alpha*mean(heatmap(:));
    heatmap(heatmap < threshold) = 0;
    
    if opts.thres_first && ~isempty(resize)
        heatmap = imresize(heatmap, resize);
    end
    
    if isempty(find(heatmap,1)) % if nothing survives the threshold, use the whole image
        if ~isempty(resize)
            res = [1 1 resize(2) resize(1)];
        else
            res = [1 1 size(heatmap,1) size(heatmap,2)];
        end
        return;
    end
    
    x1 = find(sum(heatmap,1), 1, 'first');
    x2 = find(sum(heatmap,1), 1, 'last');
    y1 = find(sum(heatmap,2), 1, 'first');
    y2 = find(sum(heatmap,2), 1, 'last');
    res = [x1 y1 x2 y2];
end
