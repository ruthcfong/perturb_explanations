net_type = 'vgg16';
switch net_type
    case 'alexnet'
        net = load('/home/ruthfong/packages/matconvnet/data/models/imagenet-caffe-alex.mat');
    case 'vgg16'
        net = load('/home/ruthfong/packages/matconvnet/data/models/imagenet-vgg-verydeep-16.mat');
    otherwise
        assert(false);
end
fprintf('using %s\n\n', net_type);

img = cnn_normalize(net.meta.normalization, imread('~/neural_coding/images/tabby_cat_cropped.jpg'), 1);
target_class = 282;

res = vl_simplenn(net, img);
gradient = zeros(size(res(end).x), 'like', res(end).x);
gradient(target_class) = 1;

res_1 = vl_simplenn(net, img, gradient);
res_2 = vl_simplenn(net, repmat(img, [1 1 1 2]), repmat(gradient, [1 1 1 2]));

%%
disp('check simplenn 1-image input vs 2-image input');

first_unequal_l = -1;
for i=2:length(res_1)
    if ~isequal(res_1(i).x, res_2(i).x(:,:,:,1))
        fprintf('%s output not equal\n', net.layers{i-1}.name);
        if first_unequal_l == -1
            first_unequal_l = i-1;
        end
    end
end

for i=length(res_1):-1:2
    if ~isequal(res_1(i).dzdx, res_2(i).dzdx(:,:,:,1))
        fprintf('%s deriv not equal\n', net.layers{i-1}.name);
    end
end

%%
net_dag_1 = dagnn.DagNN.fromSimpleNN(net);
net_dag_2 = dagnn.DagNN.fromSimpleNN(net);
net_dag_1.conserveMemory = false;
net_dag_2.conserveMemory = false;
order = net_dag_1.getLayerExecutionOrder();
inputs_1 = {net_dag_1.vars(net_dag_1.layers(order(1)).inputIndexes).name, img};
inputs_2 = {net_dag_1.vars(net_dag_1.layers(order(1)).inputIndexes).name, repmat(img, [1 1 1 2])};
ders_1 = {net_dag_1.vars(net_dag_1.layers(order(end)).outputIndexes).name, gradient};
ders_2 = {net_dag_1.vars(net_dag_1.layers(order(end)).outputIndexes).name, repmat(gradient, [1 1 1 2])};

net_dag_1.eval(inputs_1, ders_1);
net_dag_2.eval(inputs_2, ders_2);

disp('check dagnn 1-image input vs 2-image input');
for i=1:length(order)
    if ~isequal(net_dag_1.vars(net_dag_1.layers(order(i)).outputIndexes).value, ...
            net_dag_2.vars(net_dag_2.layers(order(i)).outputIndexes).value(:,:,:,1))
        fprintf('%s output not equal\n', net_dag_1.layers(order(i)).name);
    end
end

for i=length(order):-1:1
    if ~isequal(net_dag_1.vars(net_dag_1.layers(order(i)).outputIndexes).der, ...
            net_dag_2.vars(net_dag_2.layers(order(i)).outputIndexes).der(:,:,:,1))
        fprintf('%s der not equal\n', net_dag_1.layers(order(i)).name);
    end
end

disp('check between the two outputs from the 2-image run');
for i=1:length(order)
    if ~isequal(net_dag_2.vars(net_dag_2.layers(order(i)).outputIndexes).value(:,:,:,2), ...
            net_dag_2.vars(net_dag_2.layers(order(i)).outputIndexes).value(:,:,:,1))
        fprintf('%s output not equal\n', net_dag_2.layers(order(i)).name);
    end
end

for i=length(order):-1:1
    if ~isequal(net_dag_2.vars(net_dag_2.layers(order(i)).outputIndexes).der(:,:,:,1), ...
            net_dag_2.vars(net_dag_2.layers(order(i)).outputIndexes).der(:,:,:,2))
        fprintf('%s der not equal\n', net_dag_2.layers(order(i)).name);
    end
end

diff_x_simple = res_1(first_unequal_l+1).x-res_2(first_unequal_l+1).x(:,:,:,1);
diff_dzdx_simple = res_1(first_unequal_l+1).dzdx-res_2(first_unequal_l+1).dzdx(:,:,:,1);
fprintf('log10(max value diff) in simplenn = %f\n', log10(max(diff_x_simple)));
fprintf('log10(max deriv diff) in simplenn = %f\n', log10(max(diff_dzdx_simple)));

diff_x_dag = net_dag_1.vars(net_dag_1.layers(first_unequal_l).outputIndexes).value ...
    - net_dag_2.vars(net_dag_2.layers(first_unequal_l).outputIndexes).value(:,:,:,1);
diff_dzdx_dag = net_dag_1.vars(net_dag_1.layers(first_unequal_l).outputIndexes).der ...
    - net_dag_2.vars(net_dag_2.layers(first_unequal_l).outputIndexes).der(:,:,:,1);
fprintf('log10(max value diff) in dagnn = %f\n', log10(max(diff_x_dag)));
fprintf('log10(max deriv diff) in dagnn = %f\n', log10(max(diff_dzdx_dag)));
