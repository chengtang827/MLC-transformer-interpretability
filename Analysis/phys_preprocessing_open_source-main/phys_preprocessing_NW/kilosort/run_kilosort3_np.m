function run_kilosort3_np(data_path)
    
    fprintf('Data path %s \n', data_path);
    [data_dir, ~, ~] = fileparts(data_path);
    fprintf('data_dir %s \n', data_dir);

    base_dir = '/om2/user/nwatters/multi_prediction/phys_preprocessing/kilosort/';
    kilosort_path = 'Kilosort3';
    
    %% Get Kilosort config
    config_file = 'ks_3_np_v2.m';
    config_file_path = fullfile([base_dir, 'configFiles/'], config_file);

    %% These lines may be unnecessary
    % mex -setup C++
    % run([base_dir, '/', kilosort_path, '/CUDA/mexGPUall.m']);

    rootKS = sprintf('%s/../../spike_sorting/np_0/ks_3_output_v2', data_dir);
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
    ops.fproc = fullfile(rootH, 'temp_wh.dat'); % proc file on a fast SSD
    ops.trange = [0 Inf]; % time range to sort

    % Set the binary file
    ops.fbinary = data_path;
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
