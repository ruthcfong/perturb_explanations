function [res, heatmap] = getbb_from_heatmap(heatmap, resize, alpha)
    threshold = alpha*mean(heatmap(:));
    heatmap(heatmap < threshold) = 0;
    heatmap = imresize(heatmap, resize);
    if isempty(find(heatmap,1)) % if nothing survives the threshold, use the whole image
        res = [1 1 resize(2) resize(1)];
        return;
    end
    x1 = find(sum(heatmap,1), 1, 'first');
    x2 = find(sum(heatmap,1), 1, 'last');
    y1 = find(sum(heatmap,2), 1, 'first');
    y2 = find(sum(heatmap,2), 1, 'last');
    res = [x1 y1 x2 y2];
end
