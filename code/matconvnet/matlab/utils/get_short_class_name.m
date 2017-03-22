function s = get_short_class_name(net, classes, escape)
    s = {};
    for i=1:length(classes)
        class_split = strsplit(net.meta.classes.description{classes(i)},',');
        s_curr = strrep(class_split{1}, ' ', '_');
        if escape
            s_curr = strrep(s_curr, '_', '\_');
        end
        s{i} = s_curr;
    end
    if length(classes) == 1
        s = s{1};
    end
end