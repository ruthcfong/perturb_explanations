function x_norm = normalize(x)
    x_norm = (x - min(x(:)))/(max(x(:)) - min(x(:)));
end