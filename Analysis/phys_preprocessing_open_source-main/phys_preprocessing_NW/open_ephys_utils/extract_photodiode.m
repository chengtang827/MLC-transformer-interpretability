function extract_photodiode(open_ephys_dir)
    %% Extract and write photodiode values and times.
    % Args:
    %   open_ephys_dir: Full (optionally relative) path to the OpenEphys
    %   data directory.
    
    fprintf('open_ephys_dir: %s \n', open_ephys_dir);

    filepath = fullfile(open_ephys_dir, 'photodiode.continuous');
    [data, times, ~] = load_open_ephys_data(filepath);
    
    % 30KHz sampling, so sub-sample
    data = data(1:30:end);
    times = times(1:30:end);

    % Write data
    write_dir = fullfile(...
        open_ephys_dir, '../../trial_structure/sync_events/open_ephys');
    disp('Writing values')
    writematrix(data, [write_dir, '/photodiode_values.csv'])
    disp('Writing times')
    writematrix(times, [write_dir, '/photodiode_times.csv'])

end