function res = compute_occlusion_map(net, img, layer, max_feature_idx, orig_feature_map, varargin)
    img_size = size(img);
    assert(img_size(1) == img_size(2));
    opts.step_size = ceil(img_size(1)/50);
    opts.num_steps = floor(img_size(1)/opts.step_size);
    opts = vl_argparse(opts, varargin);
    step_size = opts.step_size;
    num_steps = opts.num_steps;

    map = zeros([size(orig_feature_map,1),size(orig_feature_map,2),num_steps*num_steps]);
    parfor n=1:num_steps*num_steps
        r = 1 + step_size*ceil(n/num_steps);
        c = 1 + step_size*(mod(n-1,num_steps)+1);
        occ_img = img;
        occ_img(r:r+step_size-1, c:c+step_size-1, :) = 0;
        res_occ = vl_simplenn(net, occ_img, 1);
        diff = orig_feature_map - res_occ(layer+1).x(:,:,max_feature_idx);
        map(:,:,n) = diff;
    end
    res.map = map;
    res.step_size = step_size;
    res.num_steps = num_steps;
    res.max_feature_idx = max_feature_idx;
    res.layer = layer;
end