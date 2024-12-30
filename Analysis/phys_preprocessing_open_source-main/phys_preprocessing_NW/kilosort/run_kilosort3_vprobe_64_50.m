function run_kilosort3_vprobe_64_50(data_dir)

    fprintf('data_dir %s \n', data_dir);
    [~, probe_name, ~] = fileparts(data_dir);
    fprintf('probe_name %s \n', probe_name);

    base_dir = '/om4/group/jazlab/nwatters/multi_prediction/phys_preprocessing/kilosort/';
    kilosort_path = 'Kilosort3';
    
    %% Get Kilosort config
    config_file = 'ks_3_vprobe_64_50_v6.m';
    config_file_path = fullfile([base_dir, 'configFiles/'], config_file);

    %% These lines may be unnecessary
    % mex -setup C++
    % run([base_dir, '/', kilosort_path, '/CUDA/mexGPUall.m']);

    fprintf('Data directory %s \n', data_dir);
    rootZ = data_dir; % the raw data binary file is in this folder
    
    rootKS = sprintf(...
        '%s/../../spike_sorting/%s/ks_3_output_pre_v6', rootZ, probe_name);
    fprintf('rootKS %s \n', rootKS);
    if isfolder(rootKS) == 0
        mkdir(rootKS);
    end
    
    addpath(genpath([base_dir, kilosort_path])) % path to kilosort folder
    addpath(genpath([base_dir, 'npy-matlab'])) % for converting to Phy
    rootH = data_dir; % path to temporary binary file (same size as data, should be on fast SSD)
    
    %% Create ops
    ops = struct();
    opt_datashift = struct();
    
    run(config_file_path)
    ops.fproc = fullfile(rootH, 'ks_data.dat'); % proc file on a fast SSD
    ops.trange = [0 Inf]; % time range to sort

    %% this block runs all the steps of the algorithm
    fprintf('Looking for data inside %s \n', rootZ)
    
    % find the binary file
    fs = [dir(fullfile(rootZ, '*.bin')) dir(fullfile(rootZ, '*raw_data.dat'))];
    ops.fbinary = fullfile(rootZ, fs(1).name);
    fprintf('Binary file: %s \n', ops.fbinary);
    
    % preprocess data to create temp_wh.dat
    rez = preprocessDataSub(ops);  %preProcess/preprocessDataSub
    rez = datashift2(rez, 1, opt_datashift);  %preProcess/datashift2
    [rez, st3, tF] = extract_spikes(rez);  %clustering/extract_spikes
    rez = template_learning(rez, tF, st3);
    [rez, st3, tF] = trackAndSort(rez);
    rez = final_clustering(rez, tF, st3);
    rez = find_merges(rez, 1);
    fprintf('found %d good units \n', sum(rez.good>0))


    % write to Phy
    fprintf('Saving results to Phy  \n')
    rezToPhy2(rez, rootKS);
    clean_up();

    % discard features in final rez file (too slow to save)
    rez.cProj = [];
    rez.cProjPC = [];
    
    function clean_up()
        % diary to keep track of option specs
        diary_filename = sprintf('%s/diary_%s.txt', rootKS,  datestr(now, 'mm-dd-yy'));
        diary(diary_filename);
        disp(ops);
        disp(opt_datashift);
        diary off;

        % delete temp_wh.dat, its just a duplicate
        delete(ops.fproc)
    end

end
