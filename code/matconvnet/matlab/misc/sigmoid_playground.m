figure;
xx = 0:0.01:1;
plot(xx, xx);
hold on;
gen_sig = @(x, c, a) (1./(1+exp(-a*(x-c))));
d_gen_sig = @(x, c, a) (a*gen_sig(x,c,a)*(1-gen_sig(x,c,a)));

c = 0.5;
a = 50;

plot(xx, gen_sig(xx, c, a));

for x=xx
    y = gen_sig(x, c, a);
    dzdx = d_gen_sig(x, c, a);
    tt = (x-0.01):0.01:(x+0.01);
    plot(tt, y+dzdx*(-0.01:0.01:0.01));
    fprintf('x=%f y=%f dydx=%f\n', x, y, dzdx);
end
