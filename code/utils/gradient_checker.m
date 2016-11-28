function gradient_checker(fn, gradfn, inputSize, delta_x)
% fn is a function for which you want to check gradients
% output = fn(input) will compute the function
% dzdinput = gradfn(input, dzdoutput) will compute the gradient
% inputSize is the size of input that should be passed to f
% delta_x is the step size for computing numerical gradients

input = randn(inputSize, 'double', 'gpuArray');
out = fn(input);
dzdy = randn(size(out), 'double', 'gpuArray');

dzdx_layer = gradfn(input, dzdy);

for i=1:numel(input)
    input(i) = input(i) + delta_x;
    out = fn(input);
    f_x_plus_h = out(:)' * dzdy(:);
    
    input(i) = input(i) - 2*delta_x;
    out = fn(input);
    f_x_minus_h = out(:)' * dzdy(:);
   
    dzdx_numerical = (f_x_plus_h - f_x_minus_h) / (2 * delta_x);

    input(i) = input(i) + delta_x;
    dzdx_layer = dzdx_layer(i);

    fprintf(1, '%f, %f, %f, %f\n', dzdx_layer, dzdx_numerical, norm(dzdx_layer - dzdx_numerical),...
        abs(dzdx_layer - dzdx_numerical) / abs(dzdx_layer + dzdx_numerical) );
    
    errorlog(i) = abs(dzdx_layer - dzdx_numerical) / abs(dzdx_layer + dzdx_numerical) ;
    
    if(errorlog(i) > 0.01 && norm(dzdx_layer - dzdx_numerical) > 1e-4)
        'what?'
        keyboard;
    end
end