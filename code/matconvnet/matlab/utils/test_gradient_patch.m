function bounding_box = test_gradient_patch(net, x, y, layernum, gradient_idx, show_graph)
    net.layers{end}.class = y;
    res = vl_simplenn(net, x, 1);

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
    bounding_box = [r_start, c_start, r_end-r_start+1, c_end-c_start+1];
    
    if show_graph,
        figure;
        subplot(2,2,1);
        if isfield(net.meta,'normalization') && isfield(net.meta.normalization, 'averageImage')
            imshow(normalize(x + net.meta.normalization.averageImage));
        else
            imshow(x);
        end
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