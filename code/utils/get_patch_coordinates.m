function [r_start,r_end, c_start, c_end] = get_patch_coordinates(...
    rf_info, layernum, img_size, in_coords)
    r_in = in_coords(1);
    c_in = in_coords(2);
    c_start = max(1,rf_info.stride(layernum)  * c_in + rf_info.offset(layernum) ...
        - (rf_info.size(layernum) - 1)/2);
    r_start = max(1,rf_info.stride(layernum) * r_in + rf_info.offset(layernum) ...
        - (rf_info.size(layernum) - 1)/2);
    c_end = min(img_size(2), c_start + rf_info.size(layernum) - 1);
    r_end = min(img_size(1), r_start + rf_info.size(layernum) - 1);
end