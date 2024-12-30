function [ops, opt_datashift, ks_output_dname] = update_ops(ops, is_neuropxl, option_number)



    if is_neuropxl
        opt_datashift.NrankPC = 6;
        opt_datashift.dd  = 50/4; % binning width across Y (um)
        opt_datashift.spkTh = 10; % same as the usual "template amplitude", but for the generic templates

        ops.NchanTOT  = 385;

    else
        opt_datashift.NrankPC = 6;
        opt_datashift.dd  = 50/4; % binning width across Y (um)
        opt_datashift.spkTh = 10; % same as the usual "template amplitude", but for the generic templates

        ops.NchanTOT  = 64;
    end


    ks_output_dname = sprintf('/ks3_output_%d/', option_number);

    switch option_number
        case 1
            ops_th_2 = 4;
        case 2  % high_cutoff = 0
            ops_th_2 = 9;


        otherwise
    end


end