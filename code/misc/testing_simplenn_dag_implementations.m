net_simple = load('/Users/brian/matconvnet-1.0-beta21/data/models/imagenet-caffe-alex.mat');
net_dag = dagnn.DagNN.fromSimpleNN(net_simple);

img = imread('/Users/brian/neural_coding/images/imagenet/water_snake.jpg');
img_ = cnn_normalize(net_simple.meta.normalization, img, 1);

target_class = 59;
gradient = zeros([1 1 1000], 'single');
gradient(target_class) = 1;

%%
res = vl_simplenn(net_simple, img_, gradient);
inputs = {net_dag.vars(1).name, img_};
derOutputs = {net_dag.vars(end).name, gradient};
net_dag = dagnn.DagNN.loadobj(net_dag);
net_dag.conserveMemory = 0;
net_dag.eval(inputs, derOutputs);

for i=length(res):-1:1
    if ~isequal(res(i).dzdx, net_dag.vars(i).der)
        fprintf('layer %d: %s derivatives are not equal\n', i-1, net_simple.layers{i-1}.name);
        break;
    end
end

figure;
subplot(1,3,1);
imshow(normalize(res(1).dzdx));
title('simple der');
subplot(1,3,2);
imshow(normalize(net_dag.vars(1).der));
title('dag der');
subplot(1,3,3);
imshow(normalize(res(1).dzdx - net_dag.vars(1).der));
title('simple - dag der');

%%
[heatmap_deconv_simple, res_simple] = compute_heatmap(net_simple, img_, target_class, 'deconvnet', Inf);
[heatmap_deconv_dag, res_dag] = compute_heatmap(net_dag, img_, target_class, 'deconvnet', Inf);

% disp('completed deconvnet');
% for i=length(res_simple):-1:1
%     if ~isequal(res_simple(i).dzdx, res_dag.vars(i).der)
%         fprintf('layer %d: %s derivatives are not equal\n', (i-1), net_simple.layers{i-1}.name);
%         break;
%     end
% end

[heatmap_sal_simple, res_simple] = compute_heatmap(net_simple, img_, target_class, 'saliency', Inf);
[heatmap_sal_dag, res_dag] = compute_heatmap(net_dag, img_, target_class, 'saliency', Inf);

% disp('completed saliency');
% for i=length(res_simple):-1:1
%     if ~isequal(res_simple(i).dzdx, res_dag.vars(i).der)
%         fprintf('layer %d: %s derivatives are not equal\n', (i-1), net_simple.layers{i-1}.name);
%         break;
%     end
% end

[heatmap_guided_simple, res_simple] = compute_heatmap(net_simple, img_, target_class, 'guided_backprop', Inf);
[heatmap_guided_dag, res_dag] = compute_heatmap(net_dag, img_, target_class, 'guided_backprop', Inf);

% disp('completed guided');
% for i=length(res_simple):-1:1
%     if ~isequal(res_simple(i).dzdx, res_dag.vars(i).der)
%         fprintf('layer %d: %s derivatives are not equal\n', (i-1), net_simple.layers{i-1}.name);
%         break;
%     end
% end

[heatmap_lrp_epsilon_simple, res_simple] = compute_heatmap(net_simple, img_, target_class, 'lrp_epsilon', Inf);
[heatmap_lrp_epsilon_dag, res_dag] = compute_heatmap(net_dag, img_, target_class, 'lrp_epsilon', Inf);

% disp('completed lrp_epsilon');
% order = res_dag.getLayerExecutionOrder;
% for i=length(res_simple):-1:1
%     if ~isequal(res_simple(i).dzdx, res_dag.vars(i).der)
%         fprintf('layer %d: %s derivatives are not equal\n', (i-1), net_simple.layers{i-1}.name);
%         break;
%     end
% end

assert(isequal(heatmap_deconvnet_simple, heatmap_deconvnet_dag));
assert(isequal(heatmap_sal_simple, heatmap_sal_dag));
assert(isequal(heatmap_guided_simple, heatmap_guided_dag));
assert(isequal(heatmap_lrp_epsilon_simple, heatmap_lrp_epsilon_dag));
%%
figure;
subplot(3,4,1); 
imshow(normalize(heatmap_sal_simple));
title('simple saliency');
subplot(3,4,2); 
imshow(normalize(heatmap_deconv_simple)); 
title('simple deconv');
subplot(3,4,3); 
imshow(normalize(heatmap_guided_simple));
title('simple guided');
subplot(3,4,4); 
imshow(normalize(heatmap_lrp_epsilon_simple));
title('simple lrp-epsilon');
subplot(3,4,5); 
imshow(normalize(heatmap_sal_dag)); 
title('dag saliency');
subplot(3,4,6); 
imshow(normalize(heatmap_deconv_dag)); 
title('dag deconv');
subplot(3,4,7); 
imshow(normalize(heatmap_guided_dag));
title('dag guided');
subplot(3,4,8); 
imshow(normalize(heatmap_lrp_epsilon_dag));
title('dag lrp-epsilon');
subplot(3,4,9); 
imagesc(heatmap_sal_simple-heatmap_sal_dag);
axis square;
colorbar;
title('simple-dag saliency diff');
subplot(3,4,10); 
imagesc(heatmap_deconv_simple-heatmap_deconv_dag);
axis square;
colorbar;
title('simple-dag deconv diff');
subplot(3,4,11); 
imagesc(heatmap_guided_simple-heatmap_guided_dag);
axis square;
colorbar;
title('simple-dag guided diff');
subplot(3,4,12); 
imagesc(heatmap_lrp_epsilon_simple-heatmap_lrp_epsilon_dag);
axis square;
colorbar;
title('simple-dag lrp-epsilon diff');