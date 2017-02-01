function localization_with_heatmaps(net, imdb_paths, all_img_idx, alpha, heatmap_type, out_file, varargin)

opts.meta = load('/data/ruthfong/ILSVRC2012/ILSVRC2014_devkit/data/meta_clsloc.mat');
opts.batch_size = 200;
opts.gpu = NaN;
opts.norm_deg = Inf;

opts = vl_argparse(opts, varargin);
%out_file = '/data/ruthfong/ILSVRC2012/saliency_loc_predictions_alpha_5_v4.txt';
heatmap_opts = struct();
heatmap_opts.gpu = opts.gpu;

wnid_to_im_id = cellfun(@(net_out) find(cellfun(@(s) ~isempty(strfind(s, net_out)), ...
    {opts.meta.synsets.WNID})), net.meta.classes.name);

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

    res = vl_simplenn(net, imgs);
    [~,max_idx] = max(res(end).x, [], 3);
    max_idx = wnid_to_im_id(squeeze(max_idx));

    heatmaps = compute_heatmap(net, imgs, target_classes, heatmap_type, opts.norm_deg, heatmap_opts);

    bb_coords = zeros([4 length(img_idx)], 'single');

    for i=1:length(img_idx)
        img_size = size(imread(imdb_paths.images.paths{img_idx(i)}));
        bb_coords(:,i) = getbb_from_heatmap(heatmaps(:,:,:,i), img_size(1:2), alpha);
    end

    fprintf('writing to %s\n', out_file);
    fid = fopen(out_file, 'a');
    fprintf(fid, '%d %d %d %d %d\n', [max_idx; bb_coords]);
    fclose(fid);
    fprintf('finished writing\n');
end


% [~,filename,~] = fileparts(imdb_paths.images.paths{1});
% 
% bb_path = fullfile(bb_dir, strcat(filename, '.xml')); 
% rec = VOCreadxml(bb_path);
% bb = rec.annotation.object.bndbox;
% bb_disp = [str2double(bb.xmin)+1, str2double(bb.ymin)+1, str2double(bb.xmax)-str2double(bb.xmin), ...
%     str2double(bb.ymax)-str2double(bb.ymin)];
% 
% figure;
% subplot(3,4,1);
% imshow(normalize(imgs(:,:,:,1)));
% subplot(3,4,2);
% imshow(normalize(heatmaps_sal(:,:,:,1)));
% subplot(3,4,3);
% imshow(normalize(heatmaps_deconv(:,:,:,1)));
% subplot(3,4,4);
% imshow(normalize(heatmaps_guided(:,:,:,1)));
% subplot(3,4,5);
% imshow(imread(imdb_paths.images.paths{1}));
% hold on;
% rectangle('Position', bb_disp, 'EdgeColor', 'r');
% hold off;
% subplot(3,4,6);
% [x1,x2,y1,y2] = getbb_from_heatmap(heatmaps_sal(:,:,:,1), sal_alpha);
% imshow(normalize(imgs(:,:,:,1)));
% hold on;
% rectangle('Position', [x1 y1 (x2-x1) (y2-y1)], 'EdgeColor', 'r');
% hold off;
% subplot(3,4,7);
% [x1,x2,y1,y2] = getbb_from_heatmap(heatmaps_deconv(:,:,:,1), deconvnet_alpha);
% imshow(normalize(imgs(:,:,:,1)));
% hold on;
% rectangle('Position', [x1 y1 (x2-x1) (y2-y1)], 'EdgeColor', 'r');
% hold off;

end

function res = getbb_from_heatmap(heatmap, resize, alpha)
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
