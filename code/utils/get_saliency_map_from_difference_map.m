function saliency_map = get_saliency_map_from_difference_map(layer_map, layer, rf_info, img_size)
    saliency_map = zeros([img_size(1), img_size(2)]);
    average_map = zeros([img_size(1), img_size(2)]);
    for r=1:size(layer_map,1)
        for c=1:size(layer_map,2)
            [r_start,r_end, c_start, c_end] = get_patch_coordinates(...
                rf_info, layer, img_size, [r,c]);
            saliency_map(r_start:r_end,c_start:c_end) = ...
                saliency_map(r_start:r_end,c_start:c_end) + layer_map(r,c);
            average_map(r_start:r_end,c_start:c_end) = ...
                average_map(r_start:r_end,c_start:c_end) + 1;
        end
    end
    saliency_map = saliency_map ./ average_map;
end