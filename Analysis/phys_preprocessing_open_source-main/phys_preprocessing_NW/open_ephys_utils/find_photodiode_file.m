function find_photodiode_file(om4_open_ephys_dir)
    % Extract and write photodiode filename.
    % Args:
    %   om4_open_ephys_dir: Full (optionally relative) path to the OpenEphys
    %       data directory on om4.
    
    fprintf('om4_open_ephys_dir: %s \n', om4_open_ephys_dir);

    disp('Loading data')
    filename_adc = '100_ADC2.continuous';
    filepath_adc = fullfile(om4_open_ephys_dir, filename_adc);
    if isfile(filepath_adc)
        % Old Open Ephys version
        filepath = filepath_adc;
        filename = filename_adc;
    else
        if length(dir([om4_open_ephys_dir, '/*.continuous'])) == 40
            % 32-channel headstage
            filename = '100_34.continuous';
        elseif length(dir([om4_open_ephys_dir, '/*.continuous'])) == 72
            % 64-channel headstage
            filename = '100_66.continuous';
        elseif length(dir([om4_open_ephys_dir, '/*.continuous'])) == 144
            % 64-channel headstage, but two runs
            filename = '100_66_2.continuous';
        elseif length(dir([om4_open_ephys_dir, '/*.continuous'])) == 136
            % double 64-channel headstage
            filename = '100_130.continuous';
        end
        filepath = fullfile(om4_open_ephys_dir, filename);
    end

    split_str = split(om4_open_ephys_dir, 'raw_data');
    raw_data_dir = split_str{1};
    write_file = fullfile(...
        raw_data_dir, ...
        'raw_data/paths_to_task_data/open_ephys_photodiode_filename');
    disp(['photodiode_filename : ' filename]);
    disp(['write_file : ' write_file]);
    fid = fopen(write_file, 'wt');
    fprintf(fid, filename);
    fclose(fid);
    disp('Finished find_photodiode_file.m.')
    
end