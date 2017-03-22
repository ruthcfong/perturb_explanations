function map = clip_map(map)
    map(map > 1) = 1;
    map(map < 0) = 0;
end