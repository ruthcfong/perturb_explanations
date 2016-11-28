function [r_start,r_end, c_start, c_end] = get_patch_coordinates(...
    rf_info, layernum, img_size, r_in, c_in)
    assert(length(r_in) == length(c_in));
    c_start = max(1,rf_info.stride(layernum)  * c_in + rf_info.offset(layernum) ...
        - (rf_info.size(layernum) - 1)/2);
    r_start = max(1,rf_info.stride(layernum) * r_in + rf_info.offset(layernum) ...
        - (rf_info.size(layernum) - 1)/2);
    c_end = min(img_size(2), c_start + rf_info.size(layernum) - 1);
    r_end = min(img_size(1), r_start + rf_info.size(layernum) - 1);
end