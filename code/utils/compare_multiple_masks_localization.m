function compare_multiple_masks_localization(nets, mask_paths, img_paths, alphas, varargin)
    opts.flip = false;
    opts.annotation_files = [];
    opts.gt_labels = [];
    opts.title_prefix = [];
    opts.save_fig_path = '';
    opts.meta_file = '/data/ruthfong/ILSVRC2012/ILSVRC2014_devkit/data/meta_clsloc.mat';
    
    opts = vl_argparse(opts, varargin);

    load(opts.meta_file);
    hash = make_hash(synsets);

    num_masks = length(mask_paths);

    bb_coords = zeros([4 num_masks], 'single');

    fig = figure();
    for i=1:num_masks
        mask_res = load(mask_paths{i});
        mask = mask_res.new_res.mask;
        if opts.flip, mask = 1-mask; end
        heatmap = mask;
        if ischar(img_paths)
            img = imread(img_paths);
        else
            img = imread(img_paths{i});
        end
        img_size = size(img);
        bb_coords(:,i) = getbb_from_heatmap(heatmap, img_size(1:2), alphas(i));
        subplot(1,1+num_masks,i+1);
        imshow(normalize(bsxfun(@times, single(img), single(imresize(heatmap, img_size(1:2))))));
        hold on;
        rectangle('Position', [bb_coords(1:2,i)' (bb_coords(3,i) - bb_coords(1,i)) (bb_coords(4,i) - bb_coords(2,i))], ...
            'EdgeColor', 'r');
        
        if ~isempty(opts.annotation_files) && ~isempty(opts.gt_labels)
            if ischar(opts.annotation_files)
                %r = VOCreadrecxml(sprintf('%s/%s',opts.annotation_dir,filename),hash);
                r = VOCreadrecxml(opts.annotation_files, hash);
            else
                r = VOCreadrecxml(opts.annotation_files{i}, hash);
            end
            objs = rmfield(r.objects,{'class','bndbox'});
            rec(i).objects = objs;
            ov_vector = compute_overlap(bb_coords(:,i),rec(i),opts.gt_labels(i));
            if isempty(opts.title_prefix)
                title(ov_vector);
            else
                title(strcat(opts.title_prefix{i}, num2str(mean(ov_vector))));
            end
        end
    end
    
    if (~isempty(opts.annotation_files) && ~isempty(opts.gt_labels) ...
            && ischar(opts.annotation_files) && ischar(img_paths))
        subplot(1,1+num_masks, 1);
        imshow(img);
        hold on;
        for j=1:length(objs)
            rectangle('Position', [objs(j).bbox(1:2) ...
                (objs(j).bbox(3)-objs(j).bbox(1)) ...
                (objs(j).bbox(4)-objs(j).bbox(2))], 'EdgeColor', 'r');
        end
        hold off;
        title('Orig Img');
    end
    
    if ~isempty(opts.save_fig_path)
        prep_path(opts.save_fig_path);
        print(fig, opts.save_fig_path, '-djpeg');
    end
end