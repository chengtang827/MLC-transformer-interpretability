
function find_pre_filter_data(open_ephys_dir)

disp(['open_ephys_dir: ' open_ephys_dir]);

record_nodes = dir(open_ephys_dir);

done = false;

for i=1:length(record_nodes)
    node = record_nodes(i).name;
    disp(['node: ' node]);
    if node(1) == '.'  % Filter out hidden files
        continue
    end
    if strcmp(node, 'unfiltered_node.txt')  % Filter out special file
        continue
    end
    disp(['record node: ' node]);
    filename = fullfile(open_ephys_dir, node, '100_15.continuous');
    disp(['filename: ' filename]);
    [neurdata, ~, ~] = load_open_ephys_data(filename);
    neurdata_slice = neurdata(1:20000);
    kernel = ones(1, 1000) / 1000;
    conv_output = conv(neurdata_slice, kernel);

    var_raw = var(neurdata_slice);
    var_conv = var(conv_output);
    variance_ratio = var_raw / var_conv;
    disp(['var_raw: ' num2str(var_raw)]);
    disp(['var_conv: ' num2str(var_conv)]);
    disp(['variance_ratio: ' num2str(variance_ratio)]);
    
    if variance_ratio < 10  % Not filtered
        write_file = fullfile(open_ephys_dir, 'unfiltered_node.txt');
        disp(['node to write : ' node]);
        disp(['write_file : ' write_file]);
        fid = fopen(write_file, 'wt');
        fprintf(fid, node);
        fclose(fid);
        done = true;
        break
    end
end

if done
    disp('Found un-filtered data')
else
    disp('Did not find un-filtered data.')
    disp(['Treating ', node, ' as random un-filtered record node.'])
    write_file = fullfile(open_ephys_dir, 'unfiltered_node.txt');
    disp(['node to write : ' node]);
    disp(['write_file : ' write_file]);
    fid = fopen(write_file, 'wt');
    fprintf(fid, node);
    fclose(fid);
end
    
disp('Finished find_pre_filter_data')

end
