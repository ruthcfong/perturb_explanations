%net_path = '../models/places-caffe-ref-upgraded-tidy.mat';
%net = load(net_path);

im = imread('../data/amusement_park/gsun_fff7c12aaf006e684f249cb2633b89da.jpg');
im_ = single(im);
im_ = imresize(im_, net.meta.normalization.imageSize(1:2));
im_ = im_ - net.meta.normalization.averageImage;

target_y = zeros([1 1 205]);
target_y(5) = single(1);

layernum = 13; % 13 = conv5
gradient_idx = [1,1];
bounding_box = test_patch(net, im_, target_y, layernum, gradient_idx);

function bounding_box = test_patch(net, x, y, layernum, gradient_idx, show_graph)
    res = vl_simplenn(net, x);
    [~, dzdy] = mse_loss(res(end).x, y);
    res = vl_simplenn(net, x, dzdy);

    x_l = res(layernum+1).x;

    gradient = zeros(size(x_l),'single');
    gradient(gradient_idx(1),gradient_idx(2),1) = single(1);
    gradient_gap = mean(gradient, 3);
    gradient_norm = normalize(gradient_gap);

    res = pass_through_net(net, x, layernum, gradient);

    dzdx_im = res(1).dzdx;
    dzdx_im_norm = normalize(dzdx_im);
    dzdx_im_gap = mean(dzdx_im_norm, 3);
    dzdx_mode = mode(dzdx_im_gap(:));
    
    %mask = abs(dzdx_im_gap - dzdx_mode) > 0.5*(max(dzdx_im_gap) - dzdx_mode);
    mask = dzdx_im_gap ~= dzdx_mode;
    [r_start,c_start] = find(mask,1,'first');
    [r_end,c_end] = find(mask,1,'last');
    bounding_box = [r_start, c_start, r_end-r_start, c_end-c_start];
    
    if show_graph,
        figure;
        subplot(2,2,1);
        imshow(normalize(x + net.meta.normalization.averageImage));
        hold on;
        rectangle('Position', bounding_box, 'EdgeColor', 'r', 'LineWidth', 2);
        hold off;
        title('Img + BB');
        subplot(2,2,2);
        imshow(gradient_norm);
        title(['Gradient for layer ', num2str(layernum)]);
        subplot(2,2,3);
        imshow(dzdx_im_norm);
        hold on;
        rectangle('Position', bounding_box, 'EdgeColor', 'r', 'LineWidth', 2);
        hold off;
        title('Gradient wrt  + BB');
        subplot(2,2,4);
        imshow(mask);
        hold on;
        rectangle('Position', bounding_box, 'EdgeColor', 'r', 'LineWidth', 2);
        hold off;
        title('Theoretical RF');
    end
end