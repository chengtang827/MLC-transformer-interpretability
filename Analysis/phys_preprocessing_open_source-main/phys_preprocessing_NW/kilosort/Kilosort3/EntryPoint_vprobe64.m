function EntryPoint_neuropxl(data_dir, sort_option)
    if nargin < 2
        sort_option = 1;
    end
    EntryPoint(data_dir, 0, sort_option);
end
