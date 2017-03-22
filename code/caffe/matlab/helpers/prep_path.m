function prep_path(file_path)
    [folder, ~, ~] = fileparts(file_path);
    if ~exist(folder, 'dir')
        mkdir(folder);
    end
end