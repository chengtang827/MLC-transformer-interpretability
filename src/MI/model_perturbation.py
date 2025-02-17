import MLC_utils as MLC_utils
from MI import hook_functions
import numpy as np
import matplotlib.pyplot as plt
import einops
import torch
import shutil
from MI.directed_graph import DirectedGraph
linear = torch._C._nn.linear
import copy
import MI.model_backtrack as backtrack


class Analyzer:
    def __init__(self, dataset, net, plot_dir='None'):
        self.dataloader = dataset['dataloader']
        self.net = net
        self.plot_dir = plot_dir
        self.langs = dataset['langs']
        self.null_dataset_path = dataset['null_dataset_path']

        self.backtrack_analysis = backtrack.Analyzer(dataset=dataset, net=net, plot_dir=plot_dir)
        self.backtrack_analysis.build_graph(bias=False)
        self.data_batch = next(iter(self.dataloader))

        self.cache = self.backtrack_analysis.cache
        self.output = self.backtrack_analysis.output




    def perturb_dec_cross_1_5_k(self, block, layer, type, head):
        if self.plot_dir != 'None':
            save_dir = self.plot_dir/'dec_cross_1_5'
            save_dir.mkdir(exist_ok=True, parents=True)
            perturbation_plot_dir = save_dir / 'perturbation_plot'
            perturbation_plot_dir.mkdir(exist_ok=True, parents=True)

        attn_dec_1_5_name = MLC_utils.get_module_names_by_regex(self.net, [{'module':'*dec*1*multi*attn_weight*', 'head':'5'}])
        attn_dec_1_5 = self.cache.cache[str(attn_dec_1_5_name[0])]
        k_dec_1_5_name = MLC_utils.get_module_names_by_regex(self.net, [{'module':'*dec*1*multi*k_hook*', 'head':'5'}])

        v_enc_1_1_name = MLC_utils.get_module_names_by_regex(self.net, [{'module':'*enc*1*self*v_hook*', 'head':'1'}])
        v_enc_1_1 = self.cache.cache[str(v_enc_1_1_name[0])]
        v_enc_1_1_swap = copy.deepcopy(v_enc_1_1)

        v_enc_1_0_name = MLC_utils.get_module_names_by_regex(self.net, [{'module':'*enc*1*self*v_hook*', 'head':'0'}])[0]
        v_enc_1_0 = self.cache.cache[str(v_enc_1_0_name)]
        v_enc_1_0_swap = copy.deepcopy(v_enc_1_0)




        v_enc_0_5_name = MLC_utils.get_module_names_by_regex(self.net, [{'module':'*enc*0*self*v_hook*', 'head':'5'}])


        resid_pre_name = MLC_utils.get_module_names_by_regex(self.net, [{'module':'*enc*0*resid_pre_hook*', 'head':'*'}])
        token_embedding = self.backtrack_analysis.graph['encoder_token'].output_to_resid
        pos_embedding = self.backtrack_analysis.graph['encoder_pos'].output_to_resid
        pos_embedding_swap = copy.deepcopy(pos_embedding)


        # v_enc_
        output = self.output
        n_batch = len(output['yq_predict'])
        arg_pred = np.arange(2)
        # arg_max_k = attention[np.arange(n_batch),0,:len(arg_pred),:].argmax(-1)
        acc_arg_pred = []
        batch_ids = []
        attn_1st = []
        attn_2nd = []
        for b in range(n_batch):
            xq_context = np.array(output['xq_context'][b])
            grammar_str = output['grammar'][b]['aux']['grammar_str']
            grammar_dict = MLC_utils.grammar_to_dict(grammar_str)
            yq = np.array(output['yq'][b])
            yq_predict = np.array(output['yq_predict'][b])
            correct = np.all(yq[arg_pred]==yq_predict[arg_pred+1])

            if (not correct) or (yq[0]==yq[1]):
                continue

            batch_ids.append(b)

            first_color = yq[0]
            first_color_poses = [i for i, token in enumerate(xq_context) if token==first_color]

            first_color_symbol = grammar_dict[1][first_color]
            first_symbol_poses = [i for i, token in enumerate(xq_context) if token==first_color_symbol]
            second_color = yq[1]
            second_color_poses = [i for i, token in enumerate(xq_context) if token==second_color]
            second_color_symbol = grammar_dict[1][second_color]
            second_symbol_poses = [i for i, token in enumerate(xq_context) if token==second_color_symbol]

            attn_1st.append([attn_dec_1_5[b,0,0,first_color_poses].sum(0),
                             attn_dec_1_5[b,0,0,second_color_poses].sum(0)])
            attn_2nd.append([attn_dec_1_5[b,0,1,second_color_poses].sum(0),
                             attn_dec_1_5[b,0,1,first_color_poses].sum(0)])

            # swap the v between those two symbols
            v_enc_1_1_swap[b,0,first_symbol_poses,:] = v_enc_1_1[b,0,second_symbol_poses,:].mean(0)
            v_enc_1_1_swap[b,0,second_symbol_poses,:] = v_enc_1_1[b,0,first_symbol_poses,:].mean(0)

            v_enc_1_0_swap[b,0,first_color_poses,:] = v_enc_1_0[b,0,second_color_poses,:].mean(0)
            v_enc_1_0_swap[b,0,second_color_poses,:] = v_enc_1_0[b,0,first_color_poses,:].mean(0)

            # swapt the positional embedding between two symbols
            pos_embedding_swap[b, first_symbol_poses[0],:] = pos_embedding[b, second_symbol_poses[0],:]
            pos_embedding_swap[b, second_symbol_poses[0],:] = pos_embedding[b, first_symbol_poses[0],:]
        
        # patch enc_1_1 v_hook and run again
        attn_1st = np.array(attn_1st)
        attn_2nd = np.array(attn_2nd)
        print(attn_1st.mean(0))
        print(attn_2nd.mean(0))
        """
        [0.95475894 0.02097722]
        [0.9313862  0.00860984]
        """

        if False:
            # only keep [enc_1_0, enc_1_1]
            circuit = {'sender_names':[
                # v_enc_1_1_name
                ], 
                    'receiver_names':[k_dec_1_5_name],
                    'knockout_names':                  
                        MLC_utils.get_module_names_by_regex(self.net, [{'module':'*enc*1*self*z_hook*', 'head':'2|3|4|5|6|7'}])+\
                        MLC_utils.get_module_names_by_regex(self.net, [{'module':'*enc*0*self*z_hook*', 'head':'*'}])+\
                        MLC_utils.get_module_names_by_regex(self.net, [{'module':'*enc*0*resid_pre_hook*', 'head':'*'}])
                        
            }

            swap_data = [v_enc_1_1_swap] 
            cache_swap = self.run_path_patching(circuit=circuit, patch_data=swap_data)
            """
            [0.90142626 0.0239757 ]
            [0.7816169  0.05932381]
            """
        if False:
            # only keep enc_1_1
            circuit = {'sender_names':[
                # v_enc_1_1_name
                ], 
                    'receiver_names':[k_dec_1_5_name],
                    'knockout_names':                  
                        MLC_utils.get_module_names_by_regex(self.net, [{'module':'*enc*1*self*z_hook*', 'head':'0|2|3|4|5|6|7'}])+\
                        MLC_utils.get_module_names_by_regex(self.net, [{'module':'*enc*0*self*z_hook*', 'head':'*'}])+\
                        MLC_utils.get_module_names_by_regex(self.net, [{'module':'*enc*0*resid_pre_hook*', 'head':'*'}])
                        
            }

            swap_data = [v_enc_1_1_swap] 
            cache_swap = self.run_path_patching(circuit=circuit, patch_data=swap_data)
            """
            [0.6461246  0.05187227]
            [0.6852026  0.10832324]
            """

        if False:
            # swapping v of enc_1_1
            circuit = {'sender_names':
                       v_enc_1_1_name
                       ,
                        'receiver_names':[k_dec_1_5_name],
                        'knockout_names':                  
                        MLC_utils.get_module_names_by_regex(self.net, [{'module':'*enc*1*self*z_hook*', 'head':'0|2|3|4|5|6|7'}])+\
                        MLC_utils.get_module_names_by_regex(self.net, [{'module':'*enc*0*self*z_hook*', 'head':'*'}])+\
                        MLC_utils.get_module_names_by_regex(self.net, [{'module':'*enc*0*resid_pre_hook*', 'head':'*'}])
            }

            swap_data = [v_enc_1_1_swap]  
            cache_swap = self.run_path_patching(circuit=circuit, patch_data=swap_data)

            """
            [0.07434747 0.61297584]
            [0.11778405 0.6788599 ]
            """

        if 1: 
            # only keep enc_1_1 and enc_0_5
            circuit = {'sender_names':[
                # v_enc_1_1_name
                ], 
                    'receiver_names':[v_enc_1_1_name, k_dec_1_5_name],
                    'knockout_names':                  
                        MLC_utils.get_module_names_by_regex(self.net, [{'module':'*enc*1*self*z_hook*', 'head':'0|2|3|4|5|6|7'}])+\
                        MLC_utils.get_module_names_by_regex(self.net, [{'module':'*enc*0*self*z_hook*', 'head':'0|1|2|3|4|6|7'}])
                        # MLC_utils.get_module_names_by_regex(self.net, [{'module':'*enc*0*resid_pre_hook*', 'head':'*'}])
                    }
            swap_data = [v_enc_1_1_swap]
            cache_no_swap = self.run_path_patching(circuit=circuit, patch_data=swap_data)

            """
            [0.6228579  0.10284919]
            [0.69263864 0.09829289]
            """


        if False:
            # swapping pos_embedding for v_enc_0_5
            resid_pre_swap = pos_embedding_swap+token_embedding
            circuit = {'sender_names':resid_pre_name, 
                        'receiver_names':[v_enc_0_5_name, 
                                        v_enc_1_1_name, 
                                        k_dec_1_5_name],
                        'knockout_names':                  
                            MLC_utils.get_module_names_by_regex(self.net, [{'module':'*enc*1*self*z_hook*', 'head':'0|2|3|4|5|6|7'}])+\
                            MLC_utils.get_module_names_by_regex(self.net, [{'module':'*enc*0*self*z_hook*', 'head':'0|1|2|3|4|6|7'}])
                            # analysis.get_module_names_by_regex(self.net, [{'module':'*enc*0*resid_pre_hook*', 'head':'*'}])
                        }
            swap_data = [resid_pre_swap]
            cache_swap = self.run_path_patching(circuit=circuit, patch_data=swap_data)
            """
            [0.10452922 0.50768065]
            [0.15946275 0.51336193]
            """

        if 1:
            # swapping pos_embedding for v_enc_0_45
            resid_pre_swap = pos_embedding_swap+token_embedding
            circuit = {'sender_names':resid_pre_name, 
                        'receiver_names':[
                                        MLC_utils.get_module_names_by_regex(self.net, [{'module':'*enc*0*self*v_hook*', 'head':'4|5'}]), 
                                        v_enc_1_1_name, 
                                        k_dec_1_5_name],
                        'knockout_names':                  
                            MLC_utils.get_module_names_by_regex(self.net, [{'module':'*enc*1*self*z_hook*', 'head':'0|2|3|4|5|6|7'}])+\
                            MLC_utils.get_module_names_by_regex(self.net, [{'module':'*enc*0*self*z_hook*', 'head':'0|1|2|3|6|7'}])
                            # analysis.get_module_names_by_regex(self.net, [{'module':'*enc*0*resid_pre_hook*', 'head':'*'}])
                        }
            
            swap_data = [resid_pre_swap]
            cache_swap = self.run_path_patching(circuit=circuit, patch_data=swap_data)
            """
            [0.09293616 0.581418  ]
            [0.12315495 0.5718108 ]
            """
        # only check the valid batch_ids
        attn_swap = cache_swap.cache[str(attn_dec_1_5_name[0])]
        attn_no_swap = cache_no_swap.cache[str(attn_dec_1_5_name[0])]
        attn_1st_swap = []
        attn_1st_no_swap = []
        for b in batch_ids:
            xq_context = np.array(output['xq_context'][b])
            grammar_str = output['grammar'][b]['aux']['grammar_str']
            grammar_dict = MLC_utils.grammar_to_dict(grammar_str)
            yq = np.array(output['yq'][b])
            yq_predict = np.array(output['yq_predict'][b])
            correct = np.all(yq[arg_pred]==yq_predict[arg_pred+1])

            if (not correct) or (yq[0]==yq[1]):
                continue

            # batch_ids.append(b)

            first_color = yq[0]
            first_color_poses = [i for i, token in enumerate(xq_context) if token==first_color]

            first_color_symbol = grammar_dict[1][first_color]
            first_symbol_poses = [i for i, token in enumerate(xq_context) if token==first_color_symbol]
            second_color = yq[1]
            second_color_poses = [i for i, token in enumerate(xq_context) if token==second_color]
            # second_color_symbol = grammar_dict[1][second_color]
            # second_symbol_poses = [i for i, token in enumerate(xq_context) if token==second_color_symbol]

            attn_1st_swap.append([attn_swap[b,0,0,first_color_poses].sum(0),
                                  attn_swap[b,0,0,second_color_poses].sum(0)])
            attn_1st_no_swap.append([attn_no_swap[b,0,0,first_color_poses].sum(0),
                                     attn_no_swap[b,0,0,second_color_poses].sum(0)]) 
        attn_1st_swap = np.array(attn_1st_swap)
        attn_1st_no_swap = np.array(attn_1st_no_swap)

        fig, ax = plt.subplots(1,2)


        for i in range(2):
            lower_percentile = np.percentile(attn_1st_no_swap[:,i], 5)
            upper_percentile = np.percentile(attn_1st_no_swap[:,i], 95)
            filtered_no_swap = attn_1st_no_swap[:,i][(attn_1st_no_swap[:,i] >= lower_percentile) & (attn_1st_no_swap[:,i] <= upper_percentile)]

            lower_percentile = np.percentile(attn_1st_swap[:,i], 5)
            upper_percentile = np.percentile(attn_1st_swap[:,i], 95)
            filtered_swap = attn_1st_swap[:,i][(attn_1st_swap[:,i] >= lower_percentile) & (attn_1st_swap[:,i] <= upper_percentile)]

            # draw violin plot
            ax[0].violinplot(filtered_no_swap[:,None], [i], showmeans=False, showmedians=True)
            ax[1].violinplot(filtered_swap[:,None], [i], showmeans=False, showmedians=True)

        ax[0].set_title('attn_1st_no_swap')
        ax[1].set_title('attn_1st_swap')
        ax[0].set_ylim([0,1])
        ax[1].set_ylim([0,1])

        plt.show()

        print(attn_1st_swap.mean(0))
        print(attn_1st_no_swap.mean(0))

        a=1






    def perturb_dec_cross_1_5_k_shortcut(self, block, layer, type, head):
        save_dir = self.plot_dir/'dec_cross_1_5'
        save_dir.mkdir(exist_ok=True, parents=True)
        perturbation_plot_dir = save_dir / 'perturbation_plot'
        perturbation_plot_dir.mkdir(exist_ok=True, parents=True)

        attn_dec_1_5_name = MLC_utils.get_module_names_by_regex(self.net, [{'module':'*dec*1*multi*attn_weight*', 'head':'5'}])
        attn_dec_1_5 = self.cache.cache[str(attn_dec_1_5_name[0])]
        k_dec_cross_1_5_name = MLC_utils.get_module_names_by_regex(self.net, [{'module':'*dec*1*multi*k_hook*', 'head':'5'}])
        q_dec_cross_1_5_name = MLC_utils.get_module_names_by_regex(self.net, [{'module':'*dec*1*multi*q_hook*', 'head':'5'}])

        v_enc_1_1_name = MLC_utils.get_module_names_by_regex(self.net, [{'module':'*enc*1*self*v_hook*', 'head':'1'}])
        v_enc_1_1 = self.cache.cache[str(v_enc_1_1_name[0])]
        v_enc_1_1_swap = copy.deepcopy(v_enc_1_1)

        v_enc_1_0_name = MLC_utils.get_module_names_by_regex(self.net, [{'module':'*enc*1*self*v_hook*', 'head':'0'}])
        v_enc_1_0 = self.cache.cache[str(v_enc_1_0_name[0])]
        v_enc_1_0_swap = copy.deepcopy(v_enc_1_0)

        enc_1_knockout_names1 = MLC_utils.get_module_names_by_regex(self.net, [{'module':'*enc*1*self*z_hook*', 'head':'0|2|3|4|5|6|7'}])
        enc_1_knockout_names01 = MLC_utils.get_module_names_by_regex(self.net, [{'module':'*enc*1*self*z_hook*', 'head':'2|3|4|5|6|7'}])
        enc_1_knockout_names_all = MLC_utils.get_module_names_by_regex(self.net, [{'module':'*enc*1*self*z_hook*', 'head':'0|1|2|3|4|5|6|7'}])

        enc_0_knockout_names5 = MLC_utils.get_module_names_by_regex(self.net, [{'module':'*enc*0*self*z_hook*', 'head':'0|1|2|3|4|6|7'}])
        enc_0_knockout_names4 = MLC_utils.get_module_names_by_regex(self.net, [{'module':'*enc*0*self*z_hook*', 'head':'0|1|2|3|5|6|7'}])
        enc_0_knockout_names45 = MLC_utils.get_module_names_by_regex(self.net, [{'module':'*enc*0*self*z_hook*', 'head':'0|1|2|3|6|7'}])

        enc_0_knockout_names_all = MLC_utils.get_module_names_by_regex(self.net, [{'module':'*enc*0*self*z_hook*', 'head':'0|1|2|3|4|5|6|7'}])

        z_enc_1_2_name = MLC_utils.get_module_names_by_regex(self.net, [{'module':'*enc*1*self*z_hook*', 'head':'2'}])
        z_enc_1_6_name = MLC_utils.get_module_names_by_regex(self.net, [{'module':'*enc*1*self*z_hook*', 'head':'6'}])
        seq_q = v_enc_1_0.shape[2]

        v_enc_0_5_name = MLC_utils.get_module_names_by_regex(self.net, [{'module':'*enc*0*self*v_hook*', 'head':'5'}])
        v_enc_0_4_name = MLC_utils.get_module_names_by_regex(self.net, [{'module':'*enc*0*self*v_hook*', 'head':'4'}])
        v_enc_0_45_name = MLC_utils.get_module_names_by_regex(self.net, [{'module':'*enc*0*self*v_hook*', 'head':'4|5'}])

        resid_pre_name = MLC_utils.get_module_names_by_regex(self.net, [{'module':'*enc*0*resid_pre_hook*', 'head':'*'}])
        token_embedding = self.backtrack_analysis.graph['encoder_token'].output_to_resid
        pos_embedding = self.backtrack_analysis.graph['encoder_pos'].output_to_resid
        pos_embedding_swap = copy.deepcopy(pos_embedding)


        # v_enc_
        output = self.output
        n_batch = len(output['yq_predict'])
        arg_pred = np.arange(2)
        # arg_max_k = attention[np.arange(n_batch),0,:len(arg_pred),:].argmax(-1)
        acc_arg_pred = []
        batch_ids = []
        attn_1st = []
        attn_2nd = []
        for b in range(n_batch):
            xq_context = np.array(output['xq_context'][b])
            grammar_str = output['grammar'][b]['aux']['grammar_str']
            grammar_dict = MLC_utils.grammar_to_dict(grammar_str)
            yq = np.array(output['yq'][b])
            yq_predict = np.array(output['yq_predict'][b])
            correct = np.all(yq[arg_pred]==yq_predict[arg_pred+1])

            if (yq[0]==yq[1]) or (not correct) or len(yq)<=2:
                continue

            batch_ids.append(b)

            first_color = yq[0]
            first_color_poses = [i for i, token in enumerate(xq_context) if token==first_color]

            first_color_symbol = grammar_dict[1][first_color]
            first_symbol_poses = [i for i, token in enumerate(xq_context) if token==first_color_symbol]
            second_color = yq[1]
            second_color_poses = [i for i, token in enumerate(xq_context) if token==second_color]
            second_color_symbol = grammar_dict[1][second_color]
            second_symbol_poses = [i for i, token in enumerate(xq_context) if token==second_color_symbol]

            attn_1st.append([attn_dec_1_5[b,0,0,first_color_poses].sum(0),
                             attn_dec_1_5[b,0,0,second_color_poses].sum(0)])
            attn_2nd.append([attn_dec_1_5[b,0,1,second_color_poses].sum(0),
                             attn_dec_1_5[b,0,1,first_color_poses].sum(0)])

            # swap the v between those two symbols
            v_enc_1_1_swap[b,0,first_symbol_poses,:] = v_enc_1_1[b,0,second_symbol_poses,:].mean(0)
            v_enc_1_1_swap[b,0,second_symbol_poses,:] = v_enc_1_1[b,0,first_symbol_poses,:].mean(0)

            v_enc_1_0_swap[b,0,first_color_poses,:] = v_enc_1_0[b,0,second_color_poses,:].mean(0)
            v_enc_1_0_swap[b,0,second_color_poses,:] = v_enc_1_0[b,0,first_color_poses,:].mean(0)

            # swapt the positional embedding between two symbols
            pos_embedding_swap[b, first_symbol_poses[0],:] = pos_embedding[b, second_symbol_poses[0],:]
            pos_embedding_swap[b, second_symbol_poses[0],:] = pos_embedding[b, first_symbol_poses[0],:]


        
        # patch enc_1_1 v_hook and run again
        attn_1st = np.array(attn_1st)
        attn_2nd = np.array(attn_2nd)
        print(attn_1st.mean(0))
        print(attn_2nd.mean(0))
        """
        [0.97484016 0.00637354]
        [0.9359743  0.01036751]
        """
        if False:
            # accuracy with only decoder embedding as q
            circuit = {'sender_names':[
                # v_enc_1_1_name
                ], 
                    'receiver_names':[q_dec_cross_1_5_name+k_dec_cross_1_5_name],
                    'knockout_names':
                        MLC_utils.get_module_names_by_regex(self.net, [{'module':'*dec*1*multi*z_hook*', 'head':'*'}])+\
                        MLC_utils.get_module_names_by_regex(self.net, [{'module':'*dec*1*self*z_hook*', 'head':'*'}])+\
                        MLC_utils.get_module_names_by_regex(self.net, [{'module':'*dec*0*multi*z_hook*', 'head':'*'}])+\
                        MLC_utils.get_module_names_by_regex(self.net, [{'module':'*dec*0*self*z_hook*', 'head':'*'}])
                        # analysis.get_module_names_by_regex(self.net, [{'module':'*dec*0*resid_pre_hook*', 'head':'*'}])
            }
            swap_data = [] 
            cache_swap = self.run_path_patching(circuit=circuit, patch_data=swap_data)
            """
            [0.5545475 0.088108 ]
            [0.21066044 0.11309821]
            """

        if 1:
            # with only decoder embedding as q, only enc_self_1_1 as k
            circuit = {'sender_names':[
                # v_enc_1_1_name
                ], 
                    'receiver_names':[q_dec_cross_1_5_name+k_dec_cross_1_5_name],
                    'knockout_names':
                        MLC_utils.get_module_names_by_regex(self.net, [{'module':'*dec*1*multi*z_hook*', 'head':'*'}])+\
                        MLC_utils.get_module_names_by_regex(self.net, [{'module':'*dec*1*self*z_hook*', 'head':'*'}])+\
                        MLC_utils.get_module_names_by_regex(self.net, [{'module':'*dec*0*multi*z_hook*', 'head':'*'}])+\
                        MLC_utils.get_module_names_by_regex(self.net, [{'module':'*dec*0*self*z_hook*', 'head':'*'}])+\
                        # analysis.get_module_names_by_regex(self.net, [{'module':'*dec*0*resid_pre_hook*', 'head':'*'}])+\

                        
                        # analysis.get_module_names_by_regex(self.net, [{'module':'*enc*1*self*z_hook*', 'head':'*'}])+\
                        MLC_utils.get_module_names_by_regex(self.net, [{'module':'*enc*0*self*z_hook*', 'head':'*'}])+\
                        MLC_utils.get_module_names_by_regex(self.net, [{'module':'*enc*0*resid_pre_hook*', 'head':'*'}])
            }
            swap_data = [] 
            cache_swap = self.run_path_patching(circuit=circuit, patch_data=swap_data)
            """
            [0.30451462 0.11368257]
            [0.20288831 0.0664316 ]
            """       

        if False: #TODO
            # with only decoder embedding as q, only enc_self_1_1_k, enc_self_0_4_k
            circuit = {'sender_names':[
                # v_enc_1_1_name
                ], 
                    'receiver_names':[
                        MLC_utils.get_module_names_by_regex(self.net, [{'module':'*enc*1*self*v_hook*', 'head':'1'}]),
                        q_dec_cross_1_5_name+k_dec_cross_1_5_name
                        ],
                    'knockout_names':
                        MLC_utils.get_module_names_by_regex(self.net, [{'module':'*dec*1*multi*z_hook*', 'head':'*'}])+\
                        MLC_utils.get_module_names_by_regex(self.net, [{'module':'*dec*1*self*z_hook*', 'head':'*'}])+\
                        MLC_utils.get_module_names_by_regex(self.net, [{'module':'*dec*0*multi*z_hook*', 'head':'*'}])+\
                        MLC_utils.get_module_names_by_regex(self.net, [{'module':'*dec*0*self*z_hook*', 'head':'*'}])+\
                        MLC_utils.get_module_names_by_regex(self.net, [{'module':'*enc*1*self*z_hook*', 'head':'0|2|3|4|5|6|7'}])+\
                        
                        MLC_utils.get_module_names_by_regex(self.net, [{'module':'*enc*0*self*z_hook*', 'head':'0|1|2|3|5|6|7'}])+\
                        # analysis.get_module_names_by_regex(self.net, [{'module':'*enc*0*self*k_hook*', 'head':'4'}])+\

                        MLC_utils.get_module_names_by_regex(self.net, [{'module':'*enc*0*resid_pre_hook*', 'head':'*'}])
            }
            swap_data = [] 
            cache_swap = self.run_path_patching(circuit=circuit, patch_data=swap_data)
            """
            [0.40044826 0.2313409 ]
            [0.09515447 0.11936562]
            """       
        # only check the valid batch_ids
        attn_swap = cache_swap.cache[str(attn_dec_1_5_name[0])]
        attn_1st_swap = []
        attn_2nd_swap = []
        z_self_1_1 = []
        z_self_1_1_label = []
        z_self_1_1_cache = MLC_utils.get_activations_by_regex(net=self.net, 
                                                             cache=cache_swap,
                                                             hook_regex=[{'module':'*enc*1*self*z_hook*', 'head':'1'}]
                                                             )[0]
        z_self_0_4 = []
        z_self_0_4_label = []
        z_self_0_4_cache = MLC_utils.get_activations_by_regex(net=self.net, 
                                                             cache=cache_swap,
                                                             hook_regex=[{'module':'*enc*0*self*z_hook*', 'head':'4'}]
                                                             )[0]
        # 0 2 4 5 7 yes
        # 1 3 6 no
        for b in batch_ids:
            xq_context = np.array(output['xq_context'][b])
            grammar_str = output['grammar'][b]['aux']['grammar_str']
            grammar_dict = MLC_utils.grammar_to_dict(grammar_str)
            yq = np.array(output['yq'][b])
            yq_predict = np.array(output['yq_predict'][b])
            correct = np.all(yq[arg_pred]==yq_predict[arg_pred+1])

            if (not correct) or (yq[0]==yq[1]):
                continue

            # batch_ids.append(b)

            first_color = yq[0]
            first_color_poses = np.array([i for i, token in enumerate(xq_context) if token==first_color])

            first_color_symbol = grammar_dict[1][first_color]
            first_symbol_poses = np.array([i for i, token in enumerate(xq_context) if token==first_color_symbol])
            second_color = yq[1]
            second_color_poses = np.array([i for i, token in enumerate(xq_context) if token==second_color])
            second_color_symbol = grammar_dict[1][second_color]
            second_symbol_poses = np.array([i for i, token in enumerate(xq_context) if token==second_color_symbol])

            attn_1st_swap.append([attn_swap[b,0,0,first_color_poses].sum(0),
                             attn_swap[b,0,0,second_color_poses].sum(0)])
            attn_2nd_swap.append([attn_swap[b,0,1,second_color_poses].sum(0),
                             attn_swap[b,0,1,first_color_poses].sum(0)]) 
            
            z_self_1_1.append(z_self_1_1_cache[b,0,first_color_poses,:])
            z_self_1_1.append(z_self_1_1_cache[b,0,second_color_poses,:])
            z_self_1_1_label+=[1]*first_color_poses.shape[0]
            z_self_1_1_label+=[2]*second_color_poses.shape[0]

            z_self_0_4.append(z_self_0_4_cache[b,0,first_symbol_poses,:])
            z_self_0_4.append(z_self_0_4_cache[b,0,second_symbol_poses,:])
            z_self_0_4_label+=[1]*first_symbol_poses.shape[0]
            z_self_0_4_label+=[2]*second_symbol_poses.shape[0]

        attn_1st_swap = np.array(attn_1st_swap)
        attn_2nd_swap = np.array(attn_2nd_swap)
        print(attn_1st_swap.mean(0))
        print(attn_2nd_swap.mean(0))

        if False:
            z_self_1_1 = np.vstack(z_self_1_1)
            z_self_1_1_label = np.array(z_self_1_1_label)
            fig, ax = plt.subplots(1,2,figsize=(8,6))
            backtrack.plot_2D(ax=ax[0], source_data=[z_self_1_1], source_legends=['z'],color_labels=[z_self_1_1_label],
            connect=0, markers=['s'], color_dict={}, projection='PCA')

            plt.show()

        if False:
            z_self_0_4 = np.vstack(z_self_0_4)
            z_self_0_4_label = np.array(z_self_0_4_label)
            fig, ax = plt.subplots(1,2,figsize=(8,6))
            backtrack.plot_2D(ax=ax[0], source_data=[z_self_0_4], source_legends=['z'],color_labels=[z_self_0_4_label],
            connect=0, markers=['s'], color_dict={}, projection='PCA')

            plt.show()        
        a=1
        """ distance between color and it's symbol"""





    def perturb_dec_cross_1_5_q(self, block, layer, type, head):
        save_dir = self.plot_dir/'dec_cross_1_5'
        save_dir.mkdir(exist_ok=True, parents=True)
        perturbation_plot_dir = save_dir / 'perturbation_plot'
        perturbation_plot_dir.mkdir(exist_ok=True, parents=True)

        attn_dec_1_5_name = MLC_utils.get_module_names_by_regex(self.net, [{'module':'*dec*1*multi*attn_weight*', 'head':'5'}])
        attn_dec_1_5 = self.cache.cache[str(attn_dec_1_5_name[0])]
        k_dec_1_5_name = MLC_utils.get_module_names_by_regex(self.net, [{'module':'*dec*1*multi*k_hook*', 'head':'5'}])
        q_dec_cross_1_5_name = MLC_utils.get_module_names_by_regex(self.net, [{'module':'*dec*1*multi*q_hook*', 'head':'5'}])

        v_enc_1_1_name = MLC_utils.get_module_names_by_regex(self.net, [{'module':'*enc*1*self*v_hook*', 'head':'1'}])
        v_enc_1_1 = self.cache.cache[str(v_enc_1_1_name[0])]
        v_enc_1_1_swap = copy.deepcopy(v_enc_1_1)

        v_enc_1_0_name = MLC_utils.get_module_names_by_regex(self.net, [{'module':'*enc*1*self*v_hook*', 'head':'0'}])
        v_enc_1_0 = self.cache.cache[str(v_enc_1_0_name[0])]
        v_enc_1_0_swap = copy.deepcopy(v_enc_1_0)

        token_embedding = self.backtrack_analysis.graph['encoder_token'].output_to_resid
        pos_embedding = self.backtrack_analysis.graph['encoder_pos'].output_to_resid
        pos_embedding_swap = copy.deepcopy(pos_embedding)

        dec_token_embedding = self.backtrack_analysis.graph['decoder_token'].output_to_resid
        
        # v_enc_
        output = self.output
        n_batch = len(output['yq_predict'])
        arg_pred = np.arange(2)
        # arg_max_k = attention[np.arange(n_batch),0,:len(arg_pred),:].argmax(-1)
        acc_arg_pred = []
        batch_ids = []
        attn_1st = []
        attn_2nd = []
        for b in range(n_batch):
            xq_context = np.array(output['xq_context'][b])
            grammar_str = output['grammar'][b]['aux']['grammar_str']
            grammar_dict = MLC_utils.grammar_to_dict(grammar_str)
            yq = np.array(output['yq'][b])
            yq_predict = np.array(output['yq_predict'][b])
            correct = np.all(yq[arg_pred]==yq_predict[arg_pred+1])

            if (not correct) or (yq[0]==yq[1]):
                continue

            batch_ids.append(b)

            first_color = yq[0]
            first_color_poses = [i for i, token in enumerate(xq_context) if token==first_color]

            first_color_symbol = grammar_dict[1][first_color]
            first_symbol_poses = [i for i, token in enumerate(xq_context) if token==first_color_symbol]
            second_color = yq[1]
            second_color_poses = [i for i, token in enumerate(xq_context) if token==second_color]
            second_color_symbol = grammar_dict[1][second_color]
            second_symbol_poses = [i for i, token in enumerate(xq_context) if token==second_color_symbol]

            attn_1st.append([attn_dec_1_5[b,0,0,first_color_poses].sum(0),
                             attn_dec_1_5[b,0,0,second_color_poses].sum(0)])
            attn_2nd.append([attn_dec_1_5[b,0,1,second_color_poses].sum(0),
                             attn_dec_1_5[b,0,1,first_color_poses].sum(0)])

            # swap the v between those two symbols
            v_enc_1_1_swap[b,0,first_symbol_poses,:] = v_enc_1_1[b,0,second_symbol_poses,:].mean(0)
            v_enc_1_1_swap[b,0,second_symbol_poses,:] = v_enc_1_1[b,0,first_symbol_poses,:].mean(0)

            v_enc_1_0_swap[b,0,first_color_poses,:] = v_enc_1_0[b,0,second_color_poses,:].mean(0)
            v_enc_1_0_swap[b,0,second_color_poses,:] = v_enc_1_0[b,0,first_color_poses,:].mean(0)

            # swap the positional embedding between two symbols
            pos_embedding_swap[b, first_symbol_poses[0],:] = pos_embedding[b, second_symbol_poses[0],:]
            pos_embedding_swap[b, second_symbol_poses[0],:] = pos_embedding[b, first_symbol_poses[0],:]
        
        # patch enc_1_1 v_hook and run again
        attn_1st = np.array(attn_1st)
        attn_2nd = np.array(attn_2nd)
        print(attn_1st.mean(0))
        print(attn_2nd.mean(0))
        """
        [0.97484016 0.00637354]
        [0.9359743  0.01036751]
        """

        if False: 
            """
            swap v_enc_1_1 works even when only dec_cross_layer0 exists 
            which means q must contain symbol pos information
            """
            circuit = {'sender_names':
                v_enc_1_1_name
                , 
                    'receiver_names':[v_enc_1_1_name, k_dec_1_5_name+q_dec_cross_1_5_name],
                    'knockout_names':                  
                        MLC_utils.get_module_names_by_regex(self.net, [{'module':'*enc*1*self*z_hook*', 'head':'0|2|3|4|5|6|7'}])+\
                        MLC_utils.get_module_names_by_regex(self.net, [{'module':'*enc*0*self*z_hook*', 'head':'0|1|2|3|4|6|7'}])+\
                        MLC_utils.get_module_names_by_regex(self.net, [{'module':'*enc*0*resid_pre_hook*', 'head':'*'}])+\
                        
                        MLC_utils.get_module_names_by_regex(self.net, [{'module':'*dec*1*self*z_hook*', 'head':'*'}])+\
                        # analysis.get_module_names_by_regex(self.net, [{'module':'*dec*0*multi*z_hook*', 'head':'*'}])+\
                        MLC_utils.get_module_names_by_regex(self.net, [{'module':'*dec*0*self*z_hook*', 'head':'*'}])+\
                        MLC_utils.get_module_names_by_regex(self.net, [{'module':'*dec*0*resid_pre_hook*', 'head':'*'}])
                    }
            perturbation_data = [v_enc_1_1_swap]
            cache_swap = self.run_path_patching(circuit=circuit, patch_data=perturbation_data)

            """
            [0.05384419 0.35988137]
            [0.21433288 0.45827466]
            """

        if 1: 
            """
            swap v_enc_1_1 works even when only dec_cross_layer0 exists 
            which means q must contain symbol pos information
            looks like dec_cross_0_1456 contains pos info
            """
            circuit = {'sender_names':
                v_enc_1_1_name
                , 
                    'receiver_names':[v_enc_1_1_name, k_dec_1_5_name+q_dec_cross_1_5_name],
                    'knockout_names':                  
                        MLC_utils.get_module_names_by_regex(self.net, [{'module':'*enc*1*self*z_hook*', 'head':'0|2|3|4|5|6|7'}])+\
                        MLC_utils.get_module_names_by_regex(self.net, [{'module':'*enc*0*self*z_hook*', 'head':'0|1|2|3|4|6|7'}])+\
                        MLC_utils.get_module_names_by_regex(self.net, [{'module':'*enc*0*resid_pre_hook*', 'head':'*'}])+\
                        
                        MLC_utils.get_module_names_by_regex(self.net, [{'module':'*dec*1*self*z_hook*', 'head':'*'}])+\
                        MLC_utils.get_module_names_by_regex(self.net, [{'module':'*dec*0*multi*z_hook*', 'head':'0|2|3|7'}])+\
                        MLC_utils.get_module_names_by_regex(self.net, [{'module':'*dec*0*self*z_hook*', 'head':'*'}])+\
                        MLC_utils.get_module_names_by_regex(self.net, [{'module':'*dec*0*resid_pre_hook*', 'head':'*'}])
                    }
            perturbation_data = [v_enc_1_1_swap]
            cache_swap = self.run_path_patching(circuit=circuit, patch_data=perturbation_data)

            # save the perturbed activity of dec_cross_1_5_k
            k_dec_1_5_swap = cache_swap.cache[str(k_dec_1_5_name[0])]
            # torch.save(k_dec_1_5_swap, save_dir/'k_dec_1_5_swap.pth')
            # torch.save(k_dec_1_5_swap, save_dir/'k_dec_1_5_noswap.pth')
            """
            [0.11202296 0.32236853]
            [0.18250889 0.44960997]
            """


        """
        Now need to find out who in the encoder gives the decoder the pos information
        """
        
        if False:
            resid_pre_swap = pos_embedding_swap+token_embedding
            k_dec_1_5_swap = torch.load(save_dir/'k_dec_1_5_swap.pth')
            k_dec_1_5_noswap = torch.load(save_dir/'k_dec_1_5_noswap.pth')
            # dec_cross_0_3567 gives a lot of query information
            circuit = {'sender_names':
                       k_dec_1_5_name
                    #    analysis.get_module_names_by_regex(self.net, [{'module':'*enc*0*resid_pre_hook*', 'head':'*'}])
                , 

                    'receiver_names':[
                        # analysis.get_module_names_by_regex(self.net, [{'module':'*dec*0*multi*v_hook*', 'head':'*'}]),
                        # analysis.get_module_names_by_regex(self.net, [{'module':'*dec*1*multi*v_hook*', 'head':'*'}]),
                        # analysis.get_module_names_by_regex(self.net, [{'module':'*dec*0*multi*v_hook*', 'head':'3|5|6|7'}])+\

                        q_dec_cross_1_5_name
                        ],                    
                    'knockout_names':
                        MLC_utils.get_module_names_by_regex(self.net, [{'module':'*dec*1*self*z_hook*', 'head':'*'}])+\
                        MLC_utils.get_module_names_by_regex(self.net, [{'module':'*dec*0*multi*z_hook*', 'head':'0|1|2|4'}])+\
                        MLC_utils.get_module_names_by_regex(self.net, [{'module':'*dec*0*self*z_hook*', 'head':'*'}])+\
                        MLC_utils.get_module_names_by_regex(self.net, [{'module':'*dec*0*resid_pre_hook*', 'head':'*'}])+\
                        
                        MLC_utils.get_module_names_by_regex(self.net, [{'module':'*enc*1*self*z_hook*', 'head':'0|1|2|3|4|5|6|7'}])
            }
            perturbation_data = [k_dec_1_5_swap] 
            cache_swap = self.run_path_patching(circuit=circuit, patch_data=perturbation_data)
            """
            [0.39873192 0.06774113]
            [0.7356824  0.03207196]
            """

        if False:
            # dec_cross_0_57_v gives, with only enc_self_1_012_v, gives query information
            circuit = {'sender_names':[]
                # analysis.get_module_names_by_regex(self.net, [{'module':'*dec*0*resid_pre_hook*', 'head':'*'}])
                , 
                    'receiver_names':[
                        MLC_utils.get_module_names_by_regex(self.net, [{'module':'*dec*0*multi*v_hook*', 'head':'5|7'}]),
                        q_dec_cross_1_5_name
                        ],
                    'knockout_names':
                        MLC_utils.get_module_names_by_regex(self.net, [{'module':'*dec*1*self*z_hook*', 'head':'*'}])+\
                        MLC_utils.get_module_names_by_regex(self.net, [{'module':'*dec*0*multi*z_hook*', 'head':'0|1|2|3|4|6'}])+\
                        MLC_utils.get_module_names_by_regex(self.net, [{'module':'*dec*0*self*z_hook*', 'head':'*'}])+\
                        MLC_utils.get_module_names_by_regex(self.net, [{'module':'*dec*0*resid_pre_hook*', 'head':'*'}])+\
                        
                        MLC_utils.get_module_names_by_regex(self.net, [{'module':'*enc*1*self*z_hook*', 'head':'3|4|5|6|7'}])
            }
            perturbation_data = [dec_token_embedding] 
            cache_swap = self.run_path_patching(circuit=circuit, patch_data=perturbation_data)
            """
            [0.31251022 0.01824983]
            [0.07623822 0.10191116]
            """
        
        if False:
            resid_pre_swap = pos_embedding_swap+token_embedding

            # dec_cross_0_136_v gives, with only enc_self_1_012_v, swap v for enc_self_1_01 
            circuit = {'sender_names':
                        MLC_utils.get_module_names_by_regex(self.net, [{'module':'*enc*0*resid_pre_hook*', 'head':'*'}])

                # analysis.get_module_names_by_regex(self.net, [{'module':'*enc*1*v_hook*', 'head':'0|1'}])
                , 
                    'receiver_names':[
                        MLC_utils.get_module_names_by_regex(self.net, [{'module':'*dec*0*multi*v_hook*', 'head':'*'}]),
                        q_dec_cross_1_5_name
                        ],
                    'knockout_names':
                        # analysis.get_module_names_by_regex(self.net, [{'module':'*dec*1*self*z_hook*', 'head':'*'}])+\
                        # analysis.get_module_names_by_regex(self.net, [{'module':'*dec*0*multi*z_hook*', 'head':'0|1|2|3|4|6'}])+\
                        # analysis.get_module_names_by_regex(self.net, [{'module':'*dec*0*self*z_hook*', 'head':'*'}])+\
                        MLC_utils.get_module_names_by_regex(self.net, [{'module':'*dec*0*resid_pre_hook*', 'head':'*'}])
                        
                        # analysis.get_module_names_by_regex(self.net, [{'module':'*enc*1*self*z_hook*', 'head':'3|4|5|6|7'}])
            }
            perturbation_data = [resid_pre_swap] 
            cache_swap = self.run_path_patching(circuit=circuit, patch_data=perturbation_data)
            """
            [0.207738   0.08662536]
            [0.625617   0.01964375]
            """
        # only check the valid batch_ids
        attn_swap = cache_swap.cache[str(attn_dec_1_5_name[0])]
        attn_1st_swap = []
        attn_2nd_swap = []
        for b in batch_ids:
            xq_context = np.array(output['xq_context'][b])
            grammar_str = output['grammar'][b]['aux']['grammar_str']
            grammar_dict = MLC_utils.grammar_to_dict(grammar_str)
            yq = np.array(output['yq'][b])
            yq_predict = np.array(output['yq_predict'][b])
            correct = np.all(yq[arg_pred]==yq_predict[arg_pred+1])

            if (not correct) or (yq[0]==yq[1]):
                continue

            # batch_ids.append(b)

            first_color = yq[0]
            first_color_poses = [i for i, token in enumerate(xq_context) if token==first_color]

            first_color_symbol = grammar_dict[1][first_color]
            first_symbol_poses = [i for i, token in enumerate(xq_context) if token==first_color_symbol]
            second_color = yq[1]
            second_color_poses = [i for i, token in enumerate(xq_context) if token==second_color]
            second_color_symbol = grammar_dict[1][second_color]
            second_symbol_poses = [i for i, token in enumerate(xq_context) if token==second_color_symbol]

            attn_1st_swap.append([attn_swap[b,0,0,first_color_poses].sum(0),
                             attn_swap[b,0,0,second_color_poses].sum(0)])
            attn_2nd_swap.append([attn_swap[b,0,1,second_color_poses].sum(0),
                             attn_swap[b,0,1,first_color_poses].sum(0)]) 
        attn_1st_swap = np.array(attn_1st_swap)
        attn_2nd_swap = np.array(attn_2nd_swap)
        print(attn_1st_swap.mean(0))
        print(attn_2nd_swap.mean(0))


        a=1
        """ distance between color and it's symbol"""



    def run_path_patching(self, circuit: dict={}, patch_data: list=[], one_back=True, metric=None, rewrite=0):
        

        # assert mode in ['out', 'z','q','k','v'], 'mode must be z, q, k or v'

        net = self.net
        langs = self.langs
        # perma_ablate_names = get_module_names_by_regex(net, 
        #                                                [{'module':'*encoder*layer*0*z_hook*','head':'3'},
        #                                                 {'module':'*encoder*layer*0*z_hook*','head':'4'}
        #                                                 ])


        sender_names = circuit['sender_names']
        receiver_names_chain = circuit['receiver_names']
        freeze_names = MLC_utils.get_module_names_by_regex(self.net, [
            {'module':'*q_hook*', 'head':'*'},
            {'module':'*k_hook*', 'head':'*'},
            {'module':'*v_hook*', 'head':'*'}
            ])
        knockout_names = circuit['knockout_names']
        null_activations = torch.load(self.null_dataset_path)



        # clean input
        val_batch = next(iter(self.dataloader))

        #------first run------
        # forward clean run
        all_hook_names = MLC_utils.get_module_names_by_regex(self.net, [{'module':'*hook*', 'head':'*'}])
        cache_clean, hooked_cache_modules = hook_functions.add_hooks(net, mode='cache', hook_names=all_hook_names)
        output_clean = MLC_utils.eval_model(val_batch, net, langs)
        logits_clean = output_clean['logits_correct']  
        loss_clean = output_clean['loss'] 
        [hooked_module.remove_hooks() for hooked_module in hooked_cache_modules]

        # get corrupted activation at the sender    
        corrupt_sender_activation = patch_data
        if one_back:
            while True:
                print(f'receivers_chain: {receiver_names_chain}, sender: {sender_names}')
                
                _, sender_modules = hook_functions.add_hooks(net, mode='patch', hook_names=sender_names, 
                                                            patch_activation=corrupt_sender_activation)

                # patch clean activations to the freeze hooks
                # need to remove senders and receiver from frozen hooks
                freeze_names_exclusive = [name for name in freeze_names if name not in 
                                          sender_names+receiver_names_chain[0]+knockout_names]
                clean_freeze_activation = [cache_clean.cache[str(name)] for name in freeze_names_exclusive]
                _, freeze_modules = hook_functions.add_hooks(net, mode='patch', hook_names=freeze_names_exclusive,
                                                            patch_activation=clean_freeze_activation)
                
                # knockout patch
                _, knockout_modules = hook_functions.add_hooks(net, mode='patch', hook_names=knockout_names, 
                                            patch_activation=[null_activations[str(name)] for name in knockout_names])
                
                # cache the receiver activations
                cache_patch, receiver_modules = hook_functions.add_hooks(net, mode='cache', hook_names=all_hook_names)

                output_patch = MLC_utils.eval_model(val_batch, net, langs)
                logits_patch = output_patch['logits_correct']

                loss_patch = output_patch['loss']
                [hooked_module.remove_hooks() for hooked_module in sender_modules+freeze_modules+receiver_modules+knockout_modules]

                if len(receiver_names_chain)==1:
                    break
                else:
                    sender_names = receiver_names_chain.pop(0)
                    corrupt_sender_activation = [cache_patch.cache[str(name)] for name in sender_names]
            
        else:
            #------second run------

            loss_diff = []
            pred_tokens_patch = []

            corrupt_sender_activation = [null_activations[str(name)] for name in sender_names]
            _, sender_modules = hook_functions.add_hooks(net, mode='patch', hook_names=sender_names, 
                                                        patch_activation=corrupt_sender_activation)

            # patch clean activations to the freeze hooks
            # need to remove senders and receiver from frozen hooks
            freeze_names_exclusive = [name for name in freeze_names if name not in sender_names+receiver_names_chain]
            clean_freeze_activation = [cache_clean.cache[str(name)] for name in freeze_names_exclusive]
            _, freeze_modules = hook_functions.add_hooks(net, mode='patch', hook_names=freeze_names_exclusive,
                                                        patch_activation=clean_freeze_activation)
            

            
            # cache the receiver activations
            cache_patch, receiver_modules = hook_functions.add_hooks(net, mode='cache', hook_names=receiver_names_chain)

            output_patch = MLC_utils.eval_model(val_batch, net, langs)
            loss_patch = output_patch['loss']
            logits_patch = output_patch['logits_correct']
            [hooked_module.remove_hooks() for hooked_module in sender_modules+freeze_modules+knockout_modules+receiver_modules]
            
            # if no receiver, the second run is the last run
            # no third run

        return cache_patch




    def circuit_name_to_net_name(self, node:str, suffix:str):
        net_names = []
        special_nodes = {
            'decoder_token': '*output_embedding_hook*',
            'decoder_pos': '*pos_embedding_hook*',
            'encoder_token': '*input_embedding_hook*',
            'encoder_pos': '*pos_embedding_hook*',
            'unembedding': 'None',
        }

        head = node.head
        if node.head is not None:
            name = f'{node.module_name}.{node.head}'
        else:
            name = f'{node.module_name}'


        if name in special_nodes.keys():
            regex = [{'module':special_nodes[name], 'head':'*'}]
        else:
            parts = name.split('.')
            if 'cross' in parts:
                parts[1]='multi'
            regex = [{'module':f'*{parts[0]}*{parts[2]}*{parts[1]}{suffix}', 'head':f'{head}'}]
        net_names+=MLC_utils.get_module_names_by_regex(self.net, regex)
        
        return net_names



def sos_around_pos(xq, pos):
    sos_poses = np.where(xq=='SOS')[0]
    sos_before = sos_poses[sos_poses<pos]
    if len(sos_before)==0:
        sos_before = 0
    else:
        sos_before = sos_before[-1]
    sos_after = sos_poses[sos_poses>pos][0]
    return sos_before, sos_after

def get_top_tokens(xq_context, attention, percent):
    n_query = attention.shape[0]

    pred_tokens = []
    for q in range(n_query):
        token_weight={}
        for i, token in enumerate(xq_context):
            if token in token_weight.keys():
                token_weight[token]+=attention[q,i]
            else:
                token_weight[token]=attention[q,i]
        # find token with highest attention
        max_token = max(token_weight, key=token_weight.get)
        pred_tokens.append(max_token)
        # arg_sort = np.argsort(attention[q])[::-1]
        # cumsum = np.cumsum(attention[q][arg_sort])
        # arg_max = arg_sort[cumsum<percent]
        # arg_max = arg_max[arg_max<len(xq_context)]
        # tokens = xq_context[arg_max]
        # token_weight = {}
        # pred_tokens.append(xq_context[arg_max])
    return pred_tokens




