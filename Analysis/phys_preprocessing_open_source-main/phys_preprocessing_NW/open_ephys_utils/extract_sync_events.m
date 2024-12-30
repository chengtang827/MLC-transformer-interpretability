function event_times = extract_sync_events(open_ephys_dir)
    %% Extract and write digital sync signal event values and times.
    % Args:
    %   open_ephys_dir: A path to the OpenEphys data directory with events file.

    % Assume OpenEphys was run with sample rate 30000
    sample_rate = 30000;

    fprintf('open_ephys_dir: %s \n', open_ephys_dir);
    write_dir = fullfile(...
        open_ephys_dir, '../../trial_structure/sync_events/open_ephys');
    fprintf('write_dir: %s \n', write_dir);
    
    all_event_files = dir(open_ephys_dir + "/all_channels*");
    if length(all_event_files) == 1
        event_file = fullfile(open_ephys_dir, 'all_channels.events');
    elseif length(all_event_files) == 2
        event_file = fullfile(open_ephys_dir, 'all_channels_2.events');
    else
        fprintf(all_event_files)
        disp('Unable to find events file, exiting');
        exit;
    end
    fprintf('event_file: %s \n', event_file);

    [evdata, evtn, evinfo] = load_open_ephys_data(event_file);
    event_times = get_ttl_time(evdata, evtn, evinfo);
    
    % Write sample rate
    read_sample_rate_path = fullfile(write_dir, 'sample_rate');
    fprintf('Writing sample rate %d to %s', sample_rate, read_sample_rate_path);
    read_sample_rate_fid = fopen(read_sample_rate_path,'wt');
    fprintf(read_sample_rate_fid, num2str(sample_rate));
    fclose(read_sample_rate_fid);
    
    % Write event data
    disp('Writing data')
    writematrix(event_times{2}.on, [write_dir, '/sync_trial_num_zero_on.csv'])
    writematrix(event_times{2}.off, [write_dir, '/sync_trial_num_zero_off.csv'])
    writematrix(event_times{3}.on, [write_dir, '/sync_phase_on.csv'])
    writematrix(event_times{3}.off, [write_dir, '/sync_phase_off.csv'])
    writematrix(event_times{4}.on, [write_dir, '/sync_trial_num_one_on.csv'])
    writematrix(event_times{4}.off, [write_dir, '/sync_trial_num_one_off.csv'])
    writematrix(event_times{5}.on, [write_dir, '/sync_trial_start_on.csv'])
    writematrix(event_times{5}.off, [write_dir, '/sync_trial_start_off.csv'])

end


function [event_times_per_channel] = get_ttl_time(evdata_, evtn_, evinfo_)
    %% Get ttl on and off times for each ttl signal
    num_channels = max(evdata_) + 1;
    event_times_per_channel = cell(1, num_channels);
    inds_on = evinfo_.eventId == 1;
    inds_off = evinfo_.eventId == 0;
    for event_channel = 1 : num_channels
        inds_channel = (evdata_ == event_channel - 1);
        times_on = evtn_(inds_channel & inds_on);
        times_off = evtn_(inds_channel & inds_off);
        times_channel = struct('on', times_on, 'off', times_off);
        event_times_per_channel{1, event_channel} = times_channel;
    end
end


function extract_photodiode(open_ephys_dir)
    % Extract and write photodiode values and times.
    % Args:
    %   open_ephys_dir: Full (optionally relative) path to the OpenEphys
    %       data directory.

    % Assume OpenEphys was run with sample rate 30000
    sample_rate = 30000;

    fprintf('open_ephys_dir: %s \n', open_ephys_dir);
    write_dir = fullfile(...
        open_ephys_dir, '../../trial_structure/sync_events/open_ephys');
    fprintf('write_dir: %s \n', write_dir);

    photodiode_filepath = fullfile(open_ephys_dir, 'photodiode');
    [data, times, ~] = load_open_ephys_data(photodiode_filepath);
    
    stride = sample_rate / 1000;
    data = data(1:stride:end);
    times = times(1:stride:end);

    disp('Writing data')
    disp('Writing values')
    writematrix(data, [write_dir, '/photodiode_values.csv'])
    disp('Writing times')
    writematrix(times, [write_dir, '/photodiode_times.csv'])
end