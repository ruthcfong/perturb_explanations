function localization_with_heatmaps(net, imdb_paths, all_img_idx, alpha, heatmap_type, out_file, varargin)
opts.meta = load('/data/ruthfong/ILSVRC2012/ILSVRC2014_devkit/data/meta_clsloc.mat');
opts.batch_size = 200;
opts.gpu = NaN;
opts.norm_deg = Inf;
opts.mask_dir = {};
opts.mask_flip = false;
opts.layer_name = '';

opts = vl_argparse(opts, varargin);
%out_file = '/data/ruthfong/ILSVRC2012/saliency_loc_predictions_alpha_5_v4.txt';
heatmap_opts = struct();
heatmap_opts.gpu = opts.gpu;
heatmap_opts.layer_name = opts.layer_name;

wnid_to_im_id = cellfun(@(net_out) find(cellfun(@(s) ~isempty(strfind(s, net_out)), ...
    {opts.meta.synsets.WNID})), net.meta.classes.name);

isDag = isfield(net, 'params') || isprop(net, 'params');

if ~isnan(opts.gpu)
    g = gpuDevice(opts.gpu+1);
    if isDag
        net = dagnn.DagNN.loadobj(net);
        net.move('gpu');
    else
        net = vl_simplenn_move(net, 'gpu');
    end
end

for j=1:ceil(length(all_img_idx)/opts.batch_size)
    if j*opts.batch_size <= length(all_img_idx)
        img_idx = all_img_idx((j-1)*opts.batch_size+1:j*opts.batch_size);
    else
        img_idx = all_img_idx((j-1)*opts.batch_size+1:end);
    end
    
    target_classes = imdb_paths.images.labels(img_idx);

    imgs = zeros(net.meta.normalization.imageSize(1:3), 'single');
    for i=1:length(img_idx)
        imgs(:,:,:,i) = cnn_normalize(net.meta.normalization, ...
            imread(imdb_paths.images.paths{img_idx(i)}), 1);
    end

    
    if isDag
        order = net.getLayerExecutionOrder();
        if ~isnan(opts.gpu)
            imgs = gpuArray(imgs);
        end
        
        inputs = {net.vars(net.layers(order(1)).inputIndexes).name, imgs};
        net.eval(inputs);
        [~,max_idx] = max(net.vars(net.layers(order(end)).outputIndexes).value, [], 3);
        warning('localization_with_heatmaps.m has not been tested yet for DagNNs');
    else
        if ~isnan(opts.gpu)
            imgs = gpuArray(imgs);
        end
        res = vl_simplenn(net, imgs);
        [~,max_idx] = max(res(end).x, [], 3);
    end
    max_idx = wnid_to_im_id(squeeze(max_idx));

    switch heatmap_type
        case 'mask'
            heatmaps = zeros([net.meta.normalization.imageSize(1:2) length(img_idx)], 'single');
            for i=1:length(img_idx);
                mask_res = load(fullfile(opts.mask_dir, [num2str(img_idx(i)) '.mat']));
                mask = mask_res.new_res.mask;
                if opts.mask_flip, mask = 1-mask; end
                heatmaps(:,:,i) = mask;
            end
        otherwise
            if ~isnan(opts.gpu)
                if isDag
                    net.move('cpu');
                else
                    net = vl_simplenn_move(net, 'cpu');
                end
                imgs = gather(imgs);
            end
            heatmaps = compute_heatmap(net, imgs, target_classes, heatmap_type, opts.norm_deg, heatmap_opts);
            if ~isnan(opts.gpu)
                if isDag
                    net.move('gpu');
                else
                    net = vl_simplenn_move(net, 'gpu');
                end
            end
    end
    
    bb_coords = zeros([4 length(img_idx)], 'single');

    for i=1:length(img_idx)
        img_size = size(imread(imdb_paths.images.paths{img_idx(i)}));
        bb_coords(:,i) = getbb_from_heatmap(heatmaps(:,:,i), img_size(1:2), alpha);
    end

    prep_path(out_file);
    fprintf('writing to %s\n', out_file);
    fid = fopen(out_file, 'a');
    fprintf(fid, '%d %d %d %d %d\n', [max_idx; bb_coords]);
    fclose(fid);
    fprintf('finished writing\n');
end

end

function [res, heatmap] = getbb_from_heatmap(heatmap, resize, alpha)
    heatmap = imresize(heatmap, resize);
    threshold = alpha*mean(heatmap(:));
    heatmap(heatmap < threshold) = 0;
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
