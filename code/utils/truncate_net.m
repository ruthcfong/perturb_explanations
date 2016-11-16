function tnet = truncate_net(net, start_l, end_l)
    tnet = net;
    tnet.layers = tnet.layers(start_l:end_l);
end