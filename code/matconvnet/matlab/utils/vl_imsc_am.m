function img = vl_imsc_am(img_in)
% See footnote 4 in page 7.

a = quantile(img_in(:), 0.005);
if(abs(a) < 1e-5)
    a = min(img_in(:));
end
fprintf(1, 'Viz quantile is %f\n', a);
%img = 0.5 * (1 - img_in / a);
img = 1./(1 + exp(-(img_in / a * -log(99))));
end
