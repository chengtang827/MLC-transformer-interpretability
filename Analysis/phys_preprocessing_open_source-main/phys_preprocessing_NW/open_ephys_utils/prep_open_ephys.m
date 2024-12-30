
function prep_open_ephys(open_ephys_dir)
%% Prepare OpenEphys data for KiloSort.

% function prepOpenEphys_kilosort for 64ch laminar V-probes
% convert open ephys .continuous files into binary dat for kilosort
% input: foldername for one session

% Written by Nick Watters (2022), based on code by Sujaya Neupane (2019)

% Note: matlab unpacking code for open-ephys is not consistent with python
% version in that the ttl channel is stored in evdata, the timestamps are
% converted to seconds rather than stored as samples, and spikedata
% contains all timestamps rather than just 1st of block.

%% Setting up paths

disp(['open_ephys_dir: ' open_ephys_dir]);
addpath(pwd);
init_dir = pwd;

%% Get probes and their channel maps

cd (open_ephys_dir);
channel_files = dir('*.continuous');
disp(channel_files);
disp(length(channel_files));

chMap = [8:-1:1 32:-1:25 9:24];  % For 32-channel headstage
two_runs = false;
if length(channel_files) == 32  % 32-channel V-probe
    disp('32-channel V-probe')
    channel_maps = {chMap};
elseif length(channel_files) == 40  % 16-channel V-probe
    disp('16-channel V-probe')
    channel_maps = {[9:24]};
elseif length(channel_files) == 64  % 64-channel V-probe
    disp('64-channel V-probe')
    channel_maps = {[chMap + 32 chMap]};
elseif length(channel_files) == 72  % 64-channel V-probe
    disp('64-channel V-probe')
    channel_maps = {[chMap + 32 chMap]};
elseif length(channel_files) == 144  % 64-channel V-probe, two runs
    disp('64-channel V-probe')
    channel_maps = {[chMap + 32 chMap]};
    two_runs = true;
elseif length(channel_files) == 136  % Two 64-channel V-probes
    disp('Two 64-channel V-probes')
    channel_maps = {
        [chMap + 32 chMap],
        [chMap + 96 chMap + 64],
    };
    % channel_maps = {
    %     [1:1:64],
    %     [65:1:128],
    % };
else
    error(['Invalid number of channels ' length(channel_files)])
end

disp('channel_maps')
disp(channel_maps)

%% Get number of timesteps

% filename = '100_CH1.continuous';
if two_runs
    filename = '100_9_2.continuous';
else
    filename = '100_9.continuous';
end
[neurdata, ~, ~] = load_open_ephys_data(filename);
nT = length(neurdata);

%% Loop through probes

disp(['Start binarizing ' open_ephys_dir])

for probe_num=1:length(channel_maps)
    probe_num_str = num2str(probe_num - 1);
    disp(['begin processing probe ' probe_num_str])

    % Get data

    channel_map = channel_maps{probe_num}
    disp(channel_map)
    num_channels = size(channel_map, 2);
    data = {};

    for ch=1:length(channel_map)
        channel_num = channel_map(ch);
        if two_runs
            filename = strcat('100_', num2str(channel_num), '_2.continuous');
        else
            filename = strcat('100_', num2str(channel_num), '.continuous');
        end
        filename = convertStringsToChars(filename);
        [neurdata, ~, ~] = load_open_ephys_data(filename);
        neurdata = cast(neurdata', 'int16');
        data{ch} = neurdata;
        disp(strcat('chan ', num2str(channel_num), '; depth', num2str(ch), ...
            ' in memory'));
    end

    % Truncate so all channels have the same length
    disp('Truncating channels to have the same length')
    min_length = min(cellfun(@(x) size(x, 2), data));
    max_length = max(cellfun(@(x) size(x, 2), data));
    disp(['min channel data length ' num2str(min_length)])
    disp(['max channel data length ' num2str(max_length)])
    for ch=1:length(data)
        data{ch} = data{ch}(1:min_length);
    end
    data = vertcat(data{:});

    % Write data

    write_dir = strcat('../v_probe_', probe_num_str);
    mkdir(write_dir);
    write_file = strcat(write_dir, '/raw_data.dat');
    disp(strcat('writing to file: ', write_file))
    fid = fopen(write_file, 'w');
    fwrite(fid, data, 'int16');
    fclose(fid);
    disp(strcat('done processing probe ', probe_num_str))

disp('Finished prep_open_ephys')

end
