function s = get_short_class_name(net, target_class, escape)
    class_split = strsplit(net.meta.classes.description{target_class},',');
    s = strrep(class_split{1}, ' ', '_');
    if escape
        s = strrep(s, '_', '\_');
    end
end