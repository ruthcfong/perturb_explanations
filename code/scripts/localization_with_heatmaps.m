function localization_with_heatmaps(net, imdb_paths, all_img_idx, alpha, heatmap_type, out_file, varargin)
opts.meta = load('/data/ruthfong/ILSVRC2012/ILSVRC2014_devkit/data/meta_clsloc.mat');
opts.batch_size = 200;
opts.gpu = NaN;
opts.norm_deg = Inf;
opts.mask_dir = {};
opts.mask_flip = false;
opts.layer_name = '';
opts.resize_one_side = -1;

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

    if opts.resize_one_side == -1
        imgs = zeros(net.meta.normalization.imageSize(1:3), 'single');
        for i=1:length(img_idx)
            imgs(:,:,:,i) = cnn_normalize(net.meta.normalization, ...
                imread(imdb_paths.images.paths{img_idx(i)}), 1);
        end
    else
        imgs = {};
        for i=1:length(img_idx);
            img = imread(imdb_paths.images.paths{img_idx(i)});
            [H,W,~] = size(img);
            scale = opts.resize_one_side/min(H,W);
            img = imresize(img, scale);
            if ismatrix(img) % handle BW images
                img = repmat(img, [1 1 3]);
            end
            avg_img = imresize(net.meta.normalization.averageImage, ...
                [size(img,1) size(img,2)]); % TODO -- redo for vgg16, where they just give the average color channels
            imgs{i} = single(img) - single(avg_img);
        end
    end
    
%     if isDag
%         order = net.getLayerExecutionOrder();
%         if ~isnan(opts.gpu)
%             imgs = gpuArray(imgs);
%         end
%         
%         inputs = {net.vars(net.layers(order(1)).inputIndexes).name, imgs};
%         net.eval(inputs);
%         [~,max_idx] = max(net.vars(net.layers(order(end)).outputIndexes).value, [], 3);
%         warning('localization_with_heatmaps.m has not been tested yet for DagNNs');
%     else
%         if ~isnan(opts.gpu)
%             imgs = gpuArray(imgs);
%         end
%         res = vl_simplenn(net, imgs);
%         [~,max_idx] = max(res(end).x, [], 3);
%     end
    %max_idx = wnid_to_im_id(squeeze(max_idx));
    
    gt_idx = imdb_paths.images.labels(img_idx);
    gt_idx = wnid_to_im_id(squeeze(gt_idx));

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
            if opts.resize_one_side == -1
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
            else
                heatmaps = {};
                for i=1:length(img_idx)
                    heatmaps{i} = compute_heatmap(net, imgs{i}, target_classes(i), heatmap_type, opts.norm_deg, heatmap_opts);
                end
            end
    end
    
    bb_coords = zeros([4 length(img_idx)], 'single');

    for i=1:length(img_idx)
        img_size = size(imread(imdb_paths.images.paths{img_idx(i)}));
        if opts.resize_one_side == -1
            bb_coords(:,i) = getbb_from_heatmap(heatmaps(:,:,i), img_size(1:2), alpha);
        else
            bb_coords(:,i) = getbb_from_heatmap(heatmaps{i}, img_size(1:2), alpha);
        end
    end

    prep_path(out_file);
    fprintf('writing to %s\n', out_file);
    fid = fopen(out_file, 'a');
    fprintf(fid, '%d %d %d %d %d\n', [gt_idx; bb_coords]);
    fclose(fid);
    fprintf('finished writing\n');
end

end