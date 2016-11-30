function run_rotation_sensitivity_experiment(net, norm_img, varargin)

    opts.layers = [2, 6, 10, 12, 14]; % relu 1-5 alexnet
    opts.angles = 5:5:355;
    opts.fig_path = '';
    % opts.res_path = '';

    opts = vl_argparse(opts, varargin);

    layers = opts.layers;
    angles = opts.angles;

    res_ref = vl_simplenn(net, norm_img);

    num_layers = length(layers);
    assert(num_layers <= 6); % for subplots

    f = figure('units','normalized','outerposition',[0 0 1 1]); % open a maxed out figure
    subplot(4,4,1);
    imshow(normalize(norm_img));
    title('Img');

    for j=1:length(layers)
        layer = layers(j);
        mean_changes = zeros([size(res_ref(layer+1).x, 3) length(angles)]);
        for i=1:length(angles)
            angle = angles(i);
            rotated_img = imrotate(norm_img, angle, 'bilinear');
            [rot_width, rot_height, ~] = size(rotated_img);
            crop_img = imcrop(rotated_img, [(rot_width-227)/2, (rot_height-227)/2, ...
                227, 227]);
            
            res_a = vl_simplenn(net, crop_img);

            diff_vol = res_ref(layer+1).x - res_a(layer+1).x;
            size_vol = size(diff_vol);

            assert(length(size_vol) == 3);
            diff_feats = reshape(diff_vol, [prod(size_vol(1:2)), size_vol(3)]);

            mean_abs_act_diffs = mean(abs(diff_feats), 1);
    %         std_act_diffs = std(diff_feats, 1);
    %         [~, sorted_idx] = sort(mean_abs_act_diffs);

            mean_changes(:,i) = mean_abs_act_diffs;
        end
        subplot(4,4,j+1);
        plot(angles, mean_changes');
        xlabel('Angle');
        ylabel('Mean Abs Diff');
        title(net.layers{layer}.name);
        
        subplot(4,4,j+1+num_layers);
        bar(sort(mean(mean_changes, 2)));
        title(net.layers{layer}.name);
        xlabel('Sorted HUs');
        ylabel('Mean Mean Abs Diff');
        
        subplot(4,4,j+1+2*num_layers);
        hist(mean(mean_changes, 2));
        title(net.layers{layer}.name);
        xlabel('Mean Mean Abs Diff');
        ylabel('Num HUs');
    end

    if ~isempty(opts.fig_path)
        prep_path(opts.fig_path);
        print(f, opts.fig_path, '-djpeg');
    end
    
end