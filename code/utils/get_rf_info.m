function rf_info = get_rf_info(net)
    num_layers = length(net.layers);
    rf_strides = zeros([1 num_layers]);
    rf_offsets = zeros([1 num_layers]);
    rf_sizes = zeros([1 num_layers]);

    for i=1:num_layers
        l = net.layers{i};
        if ~strcmp(l.type,'conv') && ~strcmp(l.type,'pool')
            rf_strides(i) = rf_strides(i-1);
            rf_offsets(i) = rf_offsets(i-1);
            rf_sizes(i) = rf_sizes(i-1);
            continue;
        end

        assert(length(l.stride) == 1 || isequal(l.stride(1), l.stride(2)));
        if strcmp(l.type,'conv')
            weight_dims = size(l.weights{1});
            assert(weight_dims(1) == weight_dims(2));
            assert(length(l.pad) == 1|| all(l.pad == l.pad(1)));
            rf_size = weight_dims(1);
        else
            assert(l.pool(1) == l.pool(2));
            rf_size = l.pool(1);
        end

        if i == 1
            rf_strides(i) = l.stride(1);
            rf_sizes(i) = rf_size;
            rf_offsets(i) = (rf_size-1)/2+1;
            continue;
        end

        rf_strides(i) = l.stride(1) * rf_strides(i-1);
        if l.stride(1) == 1 && l.pad(1) == (rf_size - 1)/2
            rf_offsets(i) = rf_offsets(i-1);
        else
            rf_offsets(i) = rf_offsets(i-1) + rf_strides(i-1) * ((rf_size - 1)/2);
        end
        rf_sizes(i) = rf_sizes(i-1) + rf_strides(i-1) * (rf_size - 1);
    end

    rf_info.stride = rf_strides;
    rf_info.offset = rf_offsets;
    rf_info.size = rf_sizes;
end