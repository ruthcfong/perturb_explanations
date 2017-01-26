function analyze_mask_results(net, res_path, target_class)
    load(res_path);
    img = new_res.actual_feats;
    assert(isequal(size(img), net.meta.normalization.imageSize(1:3)));
    mask = new_res.mask;
    threshold = 0.1;
    
    if new_res.opts.num_iters > size(new_res.error, 2)
        prefix = 'Avg ';
    else
        prefix = '';
    end
    binary_mask = normalize(mask);
    binary_mask(binary_mask < threshold) = 0;
    binary_mask(binary_mask >= threshold) = 1;
    
    masked_img = bsxfun(@times, img, mask);
    binary_masked_img = bsxfun(@times, img, binary_mask);
    
    net.layers{end}.class = target_class;
    
    orig_res = vl_simplenn(net, img, 1);
    masked_res = vl_simplenn(net, masked_img, 1);
    binary_masked_res = vl_simplenn(net, binary_masked_img, 1);
   
    figure;
    
    subplot(2,3,1);
    imshow(normalize(img));
    [top_score, top_class] = max(orig_res(end-1).x);
    title(sprintf('%.3f: %s', top_score, ...
        get_short_class_name(net, top_class, true)));
    
    subplot(2,3,2);
    imagesc(mask);
    colorbar;
    title(strcat(prefix, 'Final Mask'));
    
    subplot(2,3,3);
    imagesc(binary_mask);
    colorbar;
    title(strcat(prefix, 'Binarized Mask'));
    
    subplot(2,3,5);
    imshow(normalize(masked_img));
    [top_score, top_class] = max(masked_res(end-1).x);
    title(sprintf('%.3f: %s', top_score, ...
        get_short_class_name(net, top_class, true)));

    subplot(2,3,6);
    imshow(normalize(bsxfun(@times, img, binary_mask)));
    [top_score, top_class] = max(binary_masked_res(end-1).x);
    title(sprintf('%.3f: %s', top_score, ...
        get_short_class_name(net, top_class, true)));
end