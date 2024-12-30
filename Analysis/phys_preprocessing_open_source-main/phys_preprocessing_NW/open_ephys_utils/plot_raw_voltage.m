function plot_raw_voltage(open_ephys_dir)
    %% Make and save plots of the raw voltage for each channel.
    % Assumes sample rate of 30kHz
    % Args:
    %   open_ephys_dir: Full (optionally relative) path to the OpenEphys
    %   data directory.
    
    sample_rate = 30000;
    
    sample_len = 1 + sample_rate ;  % 2 seconds
    x_axis = 0:1 / sample_rate:1;

    % open_ephys_dir = '/om4/group/jazlab/nwatters/phys_data/Perle/2022-04-27/2022-04-27_13-56-17';
    % write_dir = '/om2/user/nwatters/phys_data/Perle/2022-04-27_raw_voltage';
    
    fprintf('open_ephys_dir: %s \n', open_ephys_dir);
    write_dir = fullfile(open_ephys_dir, 'raw_voltage_plots');
    fprintf('write_dir: %s \n', write_dir);
    mkdir(write_dir)
    
    channel_files = dir([open_ephys_dir, '/*.continuous']);
    disp(dir(open_ephys_dir));
    disp(dir([open_ephys_dir, '/*.continuous']));
    disp(numel(channel_files));
    for i=1:numel(channel_files)
        f = fullfile(open_ephys_dir, channel_files(i).name);
        [neurdata, ~, ~] = load_open_ephys_data(f);
        
        subplot(1, 3, 1);
        plot(x_axis, neurdata(1:sample_len));
        title('Early')
        
        subplot(1, 3, 2);
        start = floor(numel(neurdata) / 2);
        x = x_axis + start / sample_rate;
        plot(x, neurdata(start:start + sample_len - 1));
        title('Middle')
        
        subplot(1, 3, 3);
        start = numel(neurdata) - sample_len + 1;
        x = x_axis + start / sample_rate;
        plot(x, neurdata(end - sample_len + 1:end));
        title('Late')
        
        fprintf('Writing channel %s \n', i)
        
        saveas(gcf, fullfile(write_dir, [channel_files(i).name '.png']));
        close;
        
    end
end