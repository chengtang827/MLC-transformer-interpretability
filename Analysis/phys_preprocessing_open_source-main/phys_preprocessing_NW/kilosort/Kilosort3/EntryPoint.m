function EntryPoint(data_dir, is_neuropxl, sort_option)

    %% init
    mex -setup C++
    
    homepath = '/om/user/rishir/lib/';
    kilosort_suffix = 'Kilosort';
    
    addpath(genpath([homepath,kilosort_suffix])) % path to kilosort folder
    addpath(genpath([homepath,'npy-matlab'])) % for converting to Phy
    
    ops = struct();
    opt_datashift = struct();

    %%
    [rootZ, rootH, rootKS] = make_directories();
    set_probe_config(); % this will overwrite ops
    set_sort_options();

    %% this block runs all the steps of the algorithm
    fprintf('Looking for data inside %s \n', rootZ)
    
    % find the binary file
    fs          = [dir(fullfile(rootZ, '*.ap.bin')) dir(fullfile(rootZ, '*.dat'))];
    ops.fbinary = fullfile(rootZ, fs(1).name);
    fprintf(1, '%s', ops.fbinary);
    
    rez                = preprocessDataSub(ops);                        %preProcess/preprocessDataSub
    rez                = datashift2(rez, 1, opt_datashift);             %preProcess/datashift2
    
    [rez, st3, tF]     = extract_spikes(rez);                           %clustering/extract_spikes
    
    rez                = template_learning(rez, tF, st3);
    
    [rez, st3, tF]     = trackAndSort(rez);
    
    rez                = final_clustering(rez, tF, st3);
    
    rez                = find_merges(rez, 1);
    
    
    rezToPhy2(rez, rootKS);
    clean_up();

    function [rootZ, rootH, rootKS] = make_directories()
        rootZ = data_dir; % the raw data binary file is in this folder
        rootH = strrep(data_dir,'/om4/group/jazlab/rishir/data/', '/om/user/rishir/data/'); % path to temporary binary file (same size as data, should be on fast SSD)
        if isfolder(rootH) == 0
            mkdir(rootH);
        end

        ks_output_dname = sprintf('/ks3_output_%d/', sort_option);
        rootKS = strcat(rootZ, ks_output_dname);
        if isfolder(rootKS) == 0
            mkdir(rootKS);
        end

        ops.fproc   = fullfile(rootH, 'temp_wh.dat'); % proc file on a fast SSD
    end

    function set_probe_config()
        pathToYourConfigFile = [homepath, kilosort_suffix, '/configFiles/']; % take from Github folder and put it somewhere else (together with the master_file)
        if is_neuropxl
            chanMapFile = 'neuropixPhase3B2_kilosortChanMap.mat';
            config_suffix = 'config_rr_neuropxl.m';
            NchanTOT = 385;
            nblocks = 5;
            intercontact_dist = 20;
        else
            chanMapFile = 'chanMap_vprobe64.mat';
            config_suffix =  'config_rr_vprobe64.m';
            NchanTOT = 64;
            nblocks = 0;
            intercontact_dist = 50;
        end

        %run([homepath, kilosort_suffix, '/CUDA/mexGPUall.m']);
        run(fullfile(pathToYourConfigFile, config_suffix))
        ops.chanMap = fullfile(pathToYourConfigFile, chanMapFile);
        ops.NchanTOT  = NchanTOT;
        ops.nblocks    = nblocks; % blocks for registration. 0 turns it off, 1 does rigid registration. Replaces "datashift" option.
        opt_datashift.dd  = intercontact_dist/4; % binning width across Y (um)
    end

    function set_sort_options()
        ops.trange    = [0 Inf]; % time range to sort
        % main parameter changes from Kilosort2 to v2.5
        ops.sig        = 20;  % spatial smoothness constant for registration
        ops.fshigh     = 300; % high-pass more aggresively
        % rishi edit: nblocks was 5

        switch sort_option
            case 1
                ops_th_2 = 4;
                auc = 0.8;
            case 2
                ops_th_2 = 9;
                auc = 0.8;
            case 3
                ops_th_2 = 4;
                auc = 0.9;
           case 4
                ops_th_2 = 9;
                auc = 0.9;
            otherwise
                ops_th_2 = 4;
                auc = 0.8;
        end

        ops.Th       = [9 ops_th_2]; % rishi edit: was [9 9]
        ops.AUCsplit = auc;

        opt_datashift.NrankPC = 6;
        opt_datashift.spkTh = 10; % same as the usual "template amplitude", but for the generic templates

    end
    
    function clean_up()
        % rishi: diary to keep track of option specs
        diary_filename = sprintf('%s/diary_%s.txt', rootKS,  datestr(now, 'mm-dd-yy'));
        diary(diary_filename);
        disp(ops);
        disp(opt_datashift);
        diary off;

        % delete temp_wh.dat, its just a duplicate
        delete(ops.fproc)
    end

end

