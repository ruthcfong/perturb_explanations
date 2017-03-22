function localization_with_heatmaps(net, paths, labels, alpha, heatmap_type, out_file, varargin)
    softmax_i = find(cellfun(@(l) strcmp(l, 'prob'), net.layer_names));
    if ~isempty(softmax_i)
        opts.start_layer = net.layer_names{softmax_i-1};
    else
        opts.start_layer = net.layer_names{end};
    end
    opts.end_layer = net.layer_names{1};

    opts.indexing = importdata('/home/ruthfong/packages/caffe/data/ilsvrc12/ascii_order_to_synset_order.txt');
    opts.batch_size = 200;
    opts.default_img_size = get_net_default_img_size('googlenet');
    opts.mean_img = caffe.io.read_mean('/home/ruthfong/packages/caffe/data/ilsvrc12/imagenet_mean.binaryproto');
    opts.gpu = NaN;
    opts.norm_deg = Inf;

    % opts.mask_dir = {};
    % opts.mask_flip = false;
    % opts.layer_name = '';
    % opts.resize_one_side = -1;

    opts = vl_argparse(opts, varargin);

    heatmap_opts = struct();
    heatmap_opts.gpu = opts.gpu;
    heatmap_opts.norm_deg = opts.norm_deg;
    heatmap_opts.start_layer = opts.start_layer;
    heatmap_opts.end_layer = opts.end_layer;

    num_tot_imgs = length(paths);
    assert(num_tot_imgs == length(labels));

    for j=1:ceil(num_tot_imgs/opts.batch_size)
        if j*opts.batch_size <= num_tot_imgs
            img_idx = ((j-1)*opts.batch_size+1):(j*opts.batch_size);
        else
            img_idx = ((j-1)*opts.batch_size+1):num_tot_imgs;
        end

        num_batch_imgs = length(img_idx);
        target_classes = labels(img_idx);

        imgs = zeros([opts.default_img_size(1:3) num_batch_imgs]);
        for i=1:num_batch_imgs
            imgs(:,:,:,i) = normalize_img(caffe.io.load_image(paths{img_idx(i)}), ...
                opts.mean_img, opts.default_img_size);
        end

        gt_labels = opts.indexing(squeeze(target_classes))';

        heatmaps = compute_heatmap(net, imgs, target_classes, heatmap_type, heatmap_opts);
        bb_coords = zeros([4 length(img_idx)]);

        heatmaps = squeeze(convert_im_order(heatmaps)); % change from caffe [W,H,(bgr)] to [H,W,(rgb)]
        for i=1:length(img_idx)
            img_size = size(imread(paths{img_idx(i)}));
            bb_coords(:,i) = getbb_from_heatmap(heatmaps(:,:,i), alpha, img_size(1:2));
        end

        prep_path(out_file);
        fprintf('writing to %s (%d-%d out of %d)\n', out_file, img_idx(1), ...
            img_idx(end), num_tot_imgs);
        fid = fopen(out_file, 'a');
        fprintf(fid, '%d %d %d %d %d\n', [gt_labels; bb_coords]);
        fclose(fid);
    end
end