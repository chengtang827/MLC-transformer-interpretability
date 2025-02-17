import MLC_utils as MLC_utils
from MI import hook_functions
import numpy as np
from matplotlib.colors import TwoSlopeNorm
import matplotlib.pyplot as plt
import einops
from tqdm import tqdm
import copy
import torch
import shutil
from MI.directed_graph import DirectedGraph
linear = torch._C._nn.linear
from sklearn.decomposition import PCA
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.feature_selection import mutual_info_regression
from scipy.special import digamma
import umap
from sklearn.manifold import TSNE
from scipy.linalg import null_space
from sklearn.preprocessing import OneHotEncoder

class ComputeNode:
    def __init__(self, net, cache, hooked_modules, output, langs):
        self.net = net
        self.cache = cache
        self.hooked_modules = hooked_modules
        self.output = output
        self.langs = langs

class UmembeddingNode(ComputeNode):
    def __init__(self, input_stream, unembedding_proj, module_name=None, analysis=None):
        self.module_name = module_name
        self.head = None
        self.input_stream = input_stream
        self.logit_problem = None
        self.unembedding_proj = unembedding_proj
        self.analysis = analysis

    def get_problems(self, weight_mask=None, arg_q=None, target_labels=None):
        n_batch = self.input_stream.cached_activation.shape[0]
        problems = []
        unembedding_proj = np.tile(self.unembedding_proj,(n_batch,1,1))

        # target_vectors = unembedding_proj[np.arange(n_batch), arg_source, :]

        self.logit_problem = Problem(source_stream=self.input_stream, target_vectors=unembedding_proj, 
                                     weight_mask=weight_mask, arg_q=arg_q, current_node=self, analysis=self.analysis, 
                                     mode='out', target_labels=target_labels)
        problems.append(self.logit_problem)

        return problems

class EmbeddingNode(ComputeNode):
    def __init__(self, module_name=None, output_to_resid=None):
        self.module_name = module_name
        self.output_to_resid = output_to_resid
        self.head = None

    def get_problems(self,  **kwargs):
        return []

class ActivationStream:
    def __init__(self, source_nodes, in_pipeline:list, cached_activation):
        """
        source_nodes: list of nodes that directly emits output to this activation

        in_pipeline: the emitted output from source_nodes needs to go through the pipeline
        to become this activation
        """
        self.in_pipeline = in_pipeline # a list of functions
        self.source_nodes = source_nodes
        self.cached_activation = cached_activation
    
    def forward(self):
        outputs = []
        nodes = []
        for node in self.source_nodes:
            input = node.output_to_resid
            for func in self.in_pipeline:
                output = func(input)
                input = output

            outputs.append(output)
            nodes.append(node)
        
        return nodes, outputs

class Problem:
    def __init__(self, source_stream:ActivationStream, target_vectors:np.ndarray=None, arg_q=None, weight_mask=None,  
                 analysis=None, current_node=None, mode:str='q', prev_problem=None, target_labels=None,
                 target_args=None):
        self.source_stream = source_stream
        self.arg_q = arg_q
        self.weight_mask = weight_mask
        self.target_vectors = target_vectors
        self.current_node = current_node
        self.mode = mode
        self.analysis = analysis
        self.prev_problem = prev_problem
        self.target_labels = target_labels
        self.target_args = target_args

    def back_track(self, prune_names=None, metric='VAF', sequential=True, eval_func=None, ablate_all=False):
        """
        track_n: max number of top nodes to back track
        threshold: the minimum score to consider a node as a top node

        Backtracking is different for encoder and decoder
        if decoder, only care about the row of the current q
        if encoder, consider all the qs

        ablate_all: if True, only keep one source node and ablate all the others
                    if False, only ablate one source node and keep all the others
        """ 

        assert metric in ['MI', 'VAF', 'inner', 'loss', 'logit', 'customized']
        # list all the previous problems
        prev_nodes = []
        prev_modes = []
        problems_chain = []
        prev_problem = self.prev_problem
        while prev_problem is not None:
            prev_nodes.append(prev_problem.current_node)
            prev_modes.append(prev_problem.mode)
            problems_chain.append(prev_problem)
            prev_problem = prev_problem.prev_problem

        
        source_nodes = self.source_stream.source_nodes

        receiver_nodes = [self.current_node] + prev_nodes
        receiver_modes = [self.mode] + prev_modes
        problems_chain = [self] + problems_chain

        source_scores = []
        n_batch = self.source_stream.cached_activation.shape[0]
        if metric in ['loss','logit', 'customized']:
            for sender_node in source_nodes:
                

                sender_nodes = [sender_node]
                sender_modes = ['z']

                circuit={
                    'sender_nodes':sender_nodes,
                    'sender_modes':sender_modes,
                    'receiver_nodes':receiver_nodes,
                    'receiver_modes':receiver_modes,
                    'problems_chain': problems_chain 
                }

                
                score_clean, score_patch = self.analysis.run_path_patching(circuit=circuit, 
                                                                           sequential=sequential, 
                                                                           metric=metric, 
                                                                           eval_func=eval_func,
                                                                           ablate_all=ablate_all)
                if metric=='loss':
                    score_clean = np.array([score_clean[b, self.arg_q[b]] for b in range(n_batch)])
                    score_patch = np.array([score_patch[b, self.arg_q[b]] for b in range(n_batch)])
                    diff = (score_patch-score_clean).sum()/score_clean.sum()

                elif metric=='logit':
                    score_clean = np.array([score_clean[b][self.arg_q[b]] for b in range(n_batch)])
                    score_patch = np.array([score_patch[b][self.arg_q[b]] for b in range(n_batch)])
                    diff = (score_patch-score_clean).sum()/score_clean.sum()
                elif metric=='customized':
                    if ablate_all:
                        diff = score_patch/score_clean
                    else:
                        diff = score_clean - score_patch


                source_scores.append(diff)
            source_scores = np.array(source_scores)
        elif metric=='VAF':
            source_scores = self.analysis.compute_VAF(problem=self)
        elif metric=='MI':
            source_scores = self.analysis.compute_MI(problem=self)
        elif metric=='inner':
            source_scores = self.analysis.compute_inner(problem=self)


        return source_scores, self.arg_q, self.target_labels, self.target_args
        


class AttentionNode(ComputeNode):
    def __init__(self, q_stream, k_stream, v_stream, attention_score, output_to_resid, module_name, head=None, analysis=None):
        self.module_name = module_name
        self.head = head
        self.analysis = analysis

        self.q_stream = q_stream
        self.k_stream = k_stream
        self.v_stream = v_stream
        self.attention_score = attention_score
        self.output_to_resid = output_to_resid

        self.q_problem = None
        
        self.k_problem = None

        self.v_problem = None


    def get_problems(self, arg_q=None, mode:str='qkv', prev_problem=None, target_labels=None,
                     target_args=None):
        
        assert set(mode).issubset({'q','k','v'})
        n_batch = self.q_stream.cached_activation.shape[0]

        target_q = self.q_stream.cached_activation[np.arange(n_batch), 0, :, :]
        # argmax_k = self.attention_score[np.arange(n_batch),0,arg_q,:].argmax(-1) # batch seq_q
        target_k = self.k_stream.cached_activation[np.arange(n_batch), 0, :, :]
        target_v = self.v_stream.cached_activation[np.arange(n_batch), 0, :, :] # batch seq d_model
        attention = self.attention_score[np.arange(n_batch),0,:,:] # batch seq_q seq_k



        
        problems = []
        if 'q' in mode:
            self.q_problem = Problem(source_stream=self.q_stream, target_vectors=target_k, weight_mask=attention, arg_q=arg_q, 
                                     analysis=self.analysis, current_node=self, mode='q', prev_problem=prev_problem, 
                                     target_labels=target_labels, target_args=target_args)
            problems.append(self.q_problem)
        if 'k' in mode:
            self.k_problem = Problem(source_stream=self.k_stream, target_vectors=target_q, weight_mask=attention, arg_q=arg_q, 
                                     analysis=self.analysis, current_node=self, mode='k', prev_problem=prev_problem, 
                                     target_labels=target_labels, target_args=target_args)
            problems.append(self.k_problem) 
        if 'v' in mode:
            # back propagate q_args to v_args
            percent_attention_explained = 0.5

            if target_args != None and target_args !={}:
                q_ids = target_args['vector_ids']
                q_labels = target_args['vector_labels']

                v_dict = {}
                for b in range(n_batch):
                    for q_id, q_label in zip(q_ids[b], q_labels[b]):
                        attn_k = attention[b, q_id,:]
                        argsort_attn_k = np.argsort(attn_k)[::-1]
                        attn_k_sorted = np.sort(attn_k)[::-1]
                        cumsum_attn_k = np.cumsum(attn_k_sorted)
                        v_ids = argsort_attn_k[:np.where(cumsum_attn_k>percent_attention_explained)[0][0]]
                        v_weights = attn_k_sorted[:np.where(cumsum_attn_k>percent_attention_explained)[0][0]]
                        for v_id, v_weight in zip(v_ids, v_weights):
                            if v_id not in v_dict:
                                v_dict[v_id] = [{q_label: v_weight}]
                            else:
                                v_dict[v_id].append({q_label: v_weight})
            # decide the 
            self.v_problem = Problem(source_stream=self.v_stream, target_vectors=target_v, weight_mask=attention, arg_q=arg_q, 
                                     analysis=self.analysis, current_node=self, mode='v', prev_problem=prev_problem, 
                                     target_labels=target_labels, target_args=target_args)
            problems.append(self.v_problem)
        
        return problems

class CachedLayerNorm:
    def __init__(self, resid_before_ln: np.array, ln: torch.nn.LayerNorm, axis:int=-1, bias=False):
        self.resid_before_ln = resid_before_ln
        self.ln = ln
        self.axis = axis
        self.bias = bias

    def __call__(self, resid):
        mu = np.mean(self.resid_before_ln, axis=self.axis, keepdims=True)
        sigma = np.std(self.resid_before_ln, axis=self.axis, keepdims=True)
        gamma = self.ln.weight.detach().cpu().numpy()
        beta = self.ln.bias.detach().cpu().numpy()
        if self.bias:
            activation_out = (resid-mu)/sigma*gamma+beta
        else:
            activation_out = resid/sigma*gamma
        return activation_out

class ResidToQKV:
    def __init__(self, in_proj, in_bias, mode, head, n_head, d_head, bias=False):
        self.in_proj = in_proj
        self.in_bias = in_bias
        self.mode = mode
        self.head = head
        self.n_head = n_head
        self.d_head = d_head
        self.bias = bias

    def __call__(self, resid):
        if self.bias:
            proj = linear(torch.from_numpy(resid.transpose(1,0,2)), 
                            self.in_proj, self.in_bias)
        else:
            proj = linear(torch.from_numpy(resid.transpose(1,0,2)), 
                            self.in_proj, None)
        d_model = resid.shape[-1]
        proj = proj.unflatten(-1, (3, d_model)).unsqueeze(0).transpose(0, -2).squeeze(-2).contiguous()

        n_batch = resid.shape[0]
        seq = resid.shape[-2]
        q, k, v = proj[0], proj[1], proj[2]
        q = q.view(seq, n_batch * self.n_head, self.d_head).transpose(0, 1)
        q = q.view(n_batch, self.n_head, seq, self.d_head)

        k = k.view(seq, n_batch * self.n_head, self.d_head).transpose(0, 1)
        k = k.view(n_batch, self.n_head, seq, self.d_head)

        v = v.view(seq, n_batch * self.n_head, self.d_head).transpose(0, 1)
        v = v.view(n_batch, self.n_head, seq, self.d_head)

        if self.mode == 'q':
            return q[:,self.head,:,:].detach().cpu().numpy()
        if self.mode == 'k':
            return k[:,self.head,:,:].detach().cpu().numpy()
        if self.mode == 'v':
            return v[:,self.head,:,:].detach().cpu().numpy()
        else:
            raise ValueError('mode must be q, k or v')        

class OZToResid:
    def __init__(self, out_proj, out_bias, n_head, d_head, bias=False):
        self.out_proj = out_proj
        self.out_bias = out_bias
        self.n_head = n_head
        self.d_head = d_head
        self.bias = bias

    def __call__(self, cache_z):
        out_proj = einops.rearrange(self.out_proj, 'd_model (n_head d_head) -> d_model n_head d_head', n_head=self.n_head, d_head=self.d_head)
        attn_out = einops.einsum(cache_z, out_proj, 'batch n_head seq d_head, d_model n_head d_head -> batch n_head seq d_model')
        if self.bias:
            attn_out = attn_out + self.out_bias
        else:
            attn_out = attn_out

        return attn_out    



class Analyzer:
    def __init__(self, dataset, net, plot_dir=None):
        self.dataloader = dataset['dataloader']
        self.net = net
        self.plot_dir = plot_dir
        self.langs = dataset['langs']
        self.null_dataset_path = dataset['null_dataset_path']
        self.cache = None
        self.output = None


    def build_graph(self, bias=False, prune_names=None):
        all_hook_names = MLC_utils.get_module_names_by_regex(self.net, [{'module':'*hook*', 'head':'*'}])
        val_batch = next(iter(self.dataloader))
        cache, hooked_modules = hook_functions.add_hooks(self.net, mode='cache', hook_names=all_hook_names)


        if prune_names is not None:
            null_activations = torch.load(self.null_dataset_path)
            net_prune_names = prune_names['net_name']
            _, ablate_modules = hook_functions.add_hooks(self.net, mode='patch', hook_names=net_prune_names, 
                        patch_activation=[null_activations[str(name)] for name in net_prune_names])
            hooked_modules+=ablate_modules
            
        output = MLC_utils.eval_model(val_batch, self.net, self.langs)
        [hook.remove_hooks() for hook in hooked_modules]


        self.cache = cache
        self.output = output

        nlayers_encoder = self.net.nlayers_encoder
        nlayers_decoder = self.net.nlayers_decoder
        n_head = self.net.nhead
        d_model = self.net.hidden_size
        d_head = d_model // n_head
        graph = {}

        encoder_token_pos = MLC_utils.get_activations_by_regex(self.net, cache, [{'module':'*encoder*layer*0*resid_pre_hook*','head':'*'}])[0]
        encoder_token = MLC_utils.get_activations_by_regex(self.net, cache, [{'module':'*input_embedding_hook*','head':'*'}])[0]
        graph['encoder_token'] = EmbeddingNode(module_name='encoder_token', output_to_resid=encoder_token)
        graph['encoder_pos'] = EmbeddingNode(module_name='encoder_pos', output_to_resid=encoder_token_pos-encoder_token)
        
        decoder_token_pos = MLC_utils.get_activations_by_regex(self.net, cache, [{'module':'*decoder*layer*0*resid_pre_hook*','head':'*'}])[0]
        decoder_token = MLC_utils.get_activations_by_regex(self.net, cache, [{'module':'*output_embedding_hook*','head':'*'}])[0]
        graph['decoder_token'] = EmbeddingNode(module_name='decoder_token', output_to_resid=decoder_token)
        graph['decoder_pos'] = EmbeddingNode(module_name='decoder_pos', output_to_resid=decoder_token_pos-decoder_token)


        # on the encoder side
        for layer in range(nlayers_encoder):
            in_proj_self = self.net.transformer.encoder.layers[layer].self_attn.in_proj_weight
            in_bias_self = self.net.transformer.encoder.layers[layer].self_attn.in_proj_bias
            resid_pre = MLC_utils.get_activations_by_regex(self.net, cache, [{'module':f'*encoder*layer*{layer}*resid_pre_hook*','head':'*'}])[0]

            cached_ln_self = CachedLayerNorm(resid_before_ln=resid_pre,
                                        ln=self.net.transformer.encoder.layers[layer].norm1, bias=bias)
            cache_z_self = MLC_utils.get_activations_by_regex(self.net, cache, [{'module':f'*encoder*layer*{layer}*z_hook*','head':'*'}])
            cache_z_self = np.concatenate(cache_z_self, axis=1)

            out_proj_self = self.net.transformer.encoder.layers[layer].self_attn.out_proj.weight.detach().cpu().numpy() # d_model x d_model
            out_bias_self = self.net.transformer.encoder.layers[layer].self_attn.out_proj.bias.detach().cpu().numpy() # d_model
            
            oz_to_resid_self = OZToResid(out_proj=out_proj_self, out_bias=out_bias_self, n_head=n_head, d_head=d_head, bias=bias)
            attn_out_self = oz_to_resid_self(cache_z_self)

            for head in range(n_head):
                attention_score = MLC_utils.get_activations_by_regex(self.net, cache, [{'module':f'*encoder*layer*{layer}*attn_weight*','head':head}])[0]
                cache_q = MLC_utils.get_activations_by_regex(self.net, cache, [{'module':f'*encoder*layer*{layer}*q_hook*','head':head}])[0]
                cache_k = MLC_utils.get_activations_by_regex(self.net, cache, [{'module':f'*encoder*layer*{layer}*k_hook*','head':head}])[0]
                cache_v = MLC_utils.get_activations_by_regex(self.net, cache, [{'module':f'*encoder*layer*{layer}*v_hook*','head':head}])[0]
                
                resid_to_q = ResidToQKV(in_proj=in_proj_self, in_bias=in_bias_self,mode='q', head=head, n_head=n_head, d_head=d_head, bias=bias) 
                resid_to_k = ResidToQKV(in_proj=in_proj_self, in_bias=in_bias_self,mode='k', head=head, n_head=n_head, d_head=d_head, bias=bias)
                resid_to_v = ResidToQKV(in_proj=in_proj_self, in_bias=in_bias_self,mode='v', head=head, n_head=n_head, d_head=d_head, bias=bias)
                

                source_nodes = [graph['encoder_token'], graph['encoder_pos']]
                for prev_layer in range(layer):
                    [source_nodes.append(graph['enc', prev_layer, 'self', head]) for head in range(n_head)]
                    
                q_stream = ActivationStream(source_nodes=source_nodes, in_pipeline=[cached_ln_self, resid_to_q], cached_activation=cache_q)
                k_stream = ActivationStream(source_nodes=source_nodes, in_pipeline=[cached_ln_self, resid_to_k], cached_activation=cache_k)
                v_stream = ActivationStream(source_nodes=source_nodes, in_pipeline=[cached_ln_self, resid_to_v], cached_activation=cache_v)


                graph['enc', layer, 'self', head] = AttentionNode(module_name=f'enc.self.{layer}', head=head,
                                                                            q_stream=q_stream, k_stream=k_stream, v_stream=v_stream, attention_score=attention_score,
                                                                            output_to_resid=attn_out_self[:,head,:,:], analysis=self)

        # on the decoder side
        for layer in range(nlayers_decoder):
            in_proj_self = self.net.transformer.decoder.layers[layer].self_attn.in_proj_weight
            in_bias_self = self.net.transformer.decoder.layers[layer].self_attn.in_proj_bias

            resid_pre = MLC_utils.get_activations_by_regex(self.net, cache, [{'module':f'*decoder*layer*{layer}*resid_pre_hook*','head':'*'}])[0]
            cached_ln_self = CachedLayerNorm(resid_before_ln=resid_pre,
                                        ln=self.net.transformer.decoder.layers[layer].norm1, bias=bias)
            
            resid_mid = MLC_utils.get_activations_by_regex(self.net, cache, [{'module':f'*decoder*layer*{layer}*resid_mid_hook*','head':'*'}])[0]
            cached_ln_cross = CachedLayerNorm(resid_before_ln=resid_mid,
                                        ln=self.net.transformer.decoder.layers[layer].norm2, bias=bias)

            resid_encoder = MLC_utils.get_activations_by_regex(self.net, cache, [{'module':f'*encoder*layer*{nlayers_encoder-1}*resid_post*','head':'*'}])[0]
            cache_ln_encoder = CachedLayerNorm(resid_before_ln=resid_encoder,
                                        ln=self.net.transformer.encoder.norm, bias=bias)
            
            cache_z_self = MLC_utils.get_activations_by_regex(self.net, cache, [{'module':f'*decoder*layer*{layer}*self*z_hook*','head':'*'}])
            cache_z_self = np.concatenate(cache_z_self, axis=1)

            out_proj_self = self.net.transformer.decoder.layers[layer].self_attn.out_proj.weight.detach().cpu().numpy() # d_model x d_model
            out_bias_self = self.net.transformer.decoder.layers[layer].self_attn.out_proj.bias.detach().cpu().numpy() # d_model

            oz_to_resid_self = OZToResid(out_proj=out_proj_self, out_bias=out_bias_self, n_head=n_head, d_head=d_head, bias=bias)
            attn_out_self = oz_to_resid_self(cache_z_self)

            cache_z_cross = MLC_utils.get_activations_by_regex(self.net, cache, [{'module':f'*decoder*layer*{layer}*multi*z_hook*','head':'*'}])
            cache_z_cross = np.concatenate(cache_z_cross, axis=1)

            in_proj_cross = self.net.transformer.decoder.layers[layer].multihead_attn.in_proj_weight # d_model x d_model
            in_bias_cross = self.net.transformer.decoder.layers[layer].multihead_attn.in_proj_bias # d_model
            out_proj_cross = self.net.transformer.decoder.layers[layer].multihead_attn.out_proj.weight.detach().cpu().numpy() # d_model x d_model
            out_bias_cross = self.net.transformer.decoder.layers[layer].multihead_attn.out_proj.bias.detach().cpu().numpy() # d_model
            
            oz_to_resid_cross = OZToResid(out_proj=out_proj_cross, out_bias=out_bias_cross, n_head=n_head, d_head=d_head)
            attn_out_cross = oz_to_resid_cross(cache_z_cross)

            # handle the self attention
            for head in range(n_head):
                attention_score = MLC_utils.get_activations_by_regex(self.net, cache, [{'module':f'*decoder*layer*{layer}*self_attn*weight*','head':head}])[0]
                cache_q = MLC_utils.get_activations_by_regex(self.net, cache, [{'module':f'*decoder*layer*{layer}*self*q_hook*','head':head}])[0]
                cache_k = MLC_utils.get_activations_by_regex(self.net, cache, [{'module':f'*decoder*layer*{layer}*self*k_hook*','head':head}])[0]
                cache_v = MLC_utils.get_activations_by_regex(self.net, cache, [{'module':f'*decoder*layer*{layer}*self*v_hook*','head':head}])[0]
                
                resid_to_q = ResidToQKV(in_proj=in_proj_self, in_bias=in_bias_self,mode='q', head=head, n_head=n_head, d_head=d_head, bias=bias) 
                resid_to_k = ResidToQKV(in_proj=in_proj_self, in_bias=in_bias_self,mode='k', head=head, n_head=n_head, d_head=d_head, bias=bias)
                resid_to_v = ResidToQKV(in_proj=in_proj_self, in_bias=in_bias_self,mode='v', head=head, n_head=n_head, d_head=d_head, bias=bias)
                
                q_source_nodes = [graph['decoder_token'], graph['decoder_pos']]
                for prev_layer in range(layer):
                    [q_source_nodes.append(graph['dec', prev_layer, 'self', head]) for head in range(n_head)]
                    [q_source_nodes.append(graph['dec', prev_layer, 'cross', head]) for head in range(n_head)]



                q_stream = ActivationStream(source_nodes=q_source_nodes, in_pipeline=[cached_ln_self, resid_to_q], cached_activation=cache_q)
                k_stream = ActivationStream(source_nodes=q_source_nodes, in_pipeline=[cached_ln_self, resid_to_k], cached_activation=cache_k)
                v_stream = ActivationStream(source_nodes=q_source_nodes, in_pipeline=[cached_ln_self, resid_to_v], cached_activation=cache_v)


                graph['dec', layer, 'self', head] = AttentionNode(module_name=f'dec.self.{layer}', head=head,
                                                                            q_stream=q_stream, k_stream=k_stream, v_stream=v_stream, attention_score=attention_score,
                                                                            output_to_resid=attn_out_self[:,head,:,:], analysis=self)

            # handle the cross attention
            for head in range(n_head):
                attention_score = MLC_utils.get_activations_by_regex(self.net, cache, [{'module':f'*decoder*layer*{layer}*multi*weight*','head':head}])[0]
                cache_q = MLC_utils.get_activations_by_regex(self.net, cache, [{'module':f'*decoder*layer*{layer}*multi*q_hook*','head':head}])[0]
                cache_k = MLC_utils.get_activations_by_regex(self.net, cache, [{'module':f'*decoder*layer*{layer}*multi*k_hook*','head':head}])[0]
                cache_v = MLC_utils.get_activations_by_regex(self.net, cache, [{'module':f'*decoder*layer*{layer}*multi*v_hook*','head':head}])[0]
                
                resid_to_q = ResidToQKV(in_proj=in_proj_cross, in_bias=in_bias_cross, mode='q', head=head, n_head=n_head, d_head=d_head, bias=bias) 
                resid_to_k = ResidToQKV(in_proj=in_proj_cross, in_bias=in_bias_cross, mode='k', head=head, n_head=n_head, d_head=d_head, bias=bias)
                resid_to_v = ResidToQKV(in_proj=in_proj_cross, in_bias=in_bias_cross, mode='v', head=head, n_head=n_head, d_head=d_head, bias=bias)
                
                q_source_nodes = [graph['decoder_token'], graph['decoder_pos']]
                for prev_layer in range(layer):
                    [q_source_nodes.append(graph['dec', prev_layer, 'self', head]) for head in range(n_head)]
                    [q_source_nodes.append(graph['dec', prev_layer, 'cross', head]) for head in range(n_head)]
                [q_source_nodes.append(graph['dec', layer, 'self', head]) for head in range(n_head)]


                kv_source_nodes = [graph['encoder_token'], graph['encoder_pos']]
                for enc_layer in range(nlayers_encoder):
                    [kv_source_nodes.append(graph['enc', enc_layer, 'self', head]) for head in range(n_head)]

                q_stream = ActivationStream(source_nodes=q_source_nodes, in_pipeline=[cached_ln_cross, resid_to_q], cached_activation=cache_q)
                k_stream = ActivationStream(source_nodes=kv_source_nodes, in_pipeline=[cache_ln_encoder, resid_to_k], cached_activation=cache_k)
                v_stream = ActivationStream(source_nodes=kv_source_nodes, in_pipeline=[cache_ln_encoder, resid_to_v], cached_activation=cache_v)


                graph['dec', layer, 'cross', head] = AttentionNode(module_name=f'dec.cross.{layer}', head=head,
                                                                            q_stream=q_stream, k_stream=k_stream, v_stream=v_stream, attention_score=attention_score,
                                                                            output_to_resid=attn_out_cross[:,head,:,:], analysis=self)
        
        # unembedding at last
        source_nodes = [graph['decoder_token'], graph['decoder_pos']]
        for layer in range(nlayers_decoder):
            [source_nodes.append(graph['dec', layer, 'self', head]) for head in range(n_head)]
            [source_nodes.append(graph['dec', layer, 'cross', head]) for head in range(n_head)]

        cached_ln_decoder = CachedLayerNorm(resid_before_ln=MLC_utils.get_activations_by_regex(self.net, cache, [{'module':f'*decoder*layer*{nlayers_decoder-1}*resid_post_hook*','head':'*'}])[0],
                            ln=self.net.transformer.decoder.norm, bias=bias)
        cache_output = MLC_utils.get_activations_by_regex(self.net, cache, [{'module':'*decoder_hook*','head':'*'}])[0]
        stream_to_unembedding = ActivationStream(source_nodes=source_nodes, in_pipeline=[cached_ln_decoder], cached_activation=cache_output)
        unembedding_proj = self.net.out.weight.detach().cpu().numpy()
        graph['unembedding'] = UmembeddingNode(module_name='unembedding', input_stream=stream_to_unembedding, unembedding_proj=unembedding_proj, analysis=self)

        self.graph = graph


    def back_track(self, pred_arg, prune_names=None, plot_score=1, metric='VAF', sequential=1, n_track=1, threshold=0.8, rewrite=False):
        back_track_plot_dir = self.plot_dir / f'BT_{metric}_token_{pred_arg}_plot'
        back_track_data_dir = self.plot_dir / f'BT_{metric}_token_{pred_arg}_data'

        if back_track_plot_dir.exists() and plot_score and rewrite:
            shutil.rmtree(back_track_plot_dir)
            shutil.rmtree(back_track_data_dir)
        
        back_track_plot_dir.mkdir(parents=True, exist_ok=True)
        back_track_data_dir.mkdir(parents=True, exist_ok=True)

        problem_mode = 'qkv'

        
        n_batch = len(self.output['yq_predict'])
        seq_q = self.graph['unembedding'].input_stream.cached_activation.shape[1]
        n_vocal = self.net.out.weight.shape[0]

        pred_tokens = []
        pred_tokens_id = []
        pred_arg = np.array(pred_arg)
        for b in range(n_batch):
            output = np.array(self.output['yq_predict'][b])
            pred_token = output[pred_arg+1]
            pred_token_id = [self.langs['output'].symbol2index[t] for t in pred_token]
            pred_tokens.append(pred_token)
            pred_tokens_id.append(pred_token_id)
        pred_tokens_id = np.vstack(pred_tokens_id)

        # first_token_ids = [self.langs['output'].symbol2index[t] for t in pred_tokens]
        token_seq = np.vstack([pred_arg.tolist() for b in range(n_batch)])


        circuit = DirectedGraph(comment='MLC-Transformer', engine='neato') 
        
        output_mask = np.zeros((n_batch, seq_q, n_vocal))
        for b in range(n_batch):
            output_mask[b, token_seq[b], pred_tokens_id[b]] = 1
        
        problem_stack = [self.graph['unembedding'].get_problems(weight_mask=output_mask, arg_q=token_seq, target_labels=pred_tokens_id)[0]]

        
        sort_id = 0
        edges = []
        while problem_stack:
            current_problem = problem_stack.pop(0)
            current_node = current_problem.current_node
            current_problem_str = f'{current_node.module_name}.{current_node.head} {current_problem.mode}'
            source_nodes = current_problem.source_stream.source_nodes

            # check if can load backtracking score from saved data
            if (back_track_data_dir/current_problem_str).exists() and not rewrite:
                source_scores, arg_q, target_args = torch.load(back_track_data_dir/current_problem_str)
                print(f'Loaded {current_problem_str}')
            else:
                source_scores, arg_q, target_labels, target_args = \
                    current_problem.back_track(prune_names=prune_names, metric=metric,
                                               sequential=sequential)
                torch.save([source_scores, arg_q, target_args], back_track_data_dir/current_problem_str)

            arg_top_nodes = get_top_nodes(source_nodes, source_scores, n_track, threshold)


            # ploting the score heatmap
            if plot_score:
                self.plot_node_scores(target_problem = current_problem, 
                    source_nodes = source_nodes, 
                    source_scores = source_scores, 
                    plot_dir=back_track_plot_dir, sort_id=sort_id,
                    cmap='cool',mid_0=False, balance=False)
            
            # only do QKV problem
            next_nodes = [source_nodes[i] for i in arg_top_nodes]
            next_scores = [source_scores[i] for i in arg_top_nodes]
            for node, score in zip(next_nodes, next_scores):
                problems = node.get_problems(arg_q=arg_q, mode=problem_mode, prev_problem=current_problem, 
                                             target_labels=target_labels, target_args=target_args)

                # for problem in problems:
                edge_name = f'{node.module_name}{node.head}->{current_node.module_name}{current_node.head}: {current_problem.mode}'
                if edge_name in edges:
                    # print(edge_name)
                    continue

                problem_stack+=problems
                print(f'({sort_id}){current_problem.current_node.module_name} {current_problem.current_node.head}\
                        {current_problem.mode} track back {node.module_name} {node.head}')
                circuit.add_edge(sender=node, receiver=current_problem, score=str(np.round(score,1)))
                edges.append(edge_name)
                sort_id+=1

        filename = f'MLC-Transformer{pred_arg}'
        save_path = self.plot_dir / f'{filename}.png'
        # if (not save_path.exists()) or rewrite:
        circuit.render(filename=filename, directory=self.plot_dir, view=True)

        return circuit


    def plot_node_scores(self, target_problem, source_nodes, source_scores, cmap='coolwarm', 
                         plot_dir=None, sort_id=None, balance=True, mid_0=True, include_emb=True, display=False):
        
        target_node = target_problem.current_node
        head = f'.{target_node.head}' if target_node.head is not None else ''
        target_name = f'{target_node.module_name}{head}'
        title = f'To {target_name} {target_problem.mode}'   
        n_head = self.net.nhead
        # rearrange the source nodes by attention or embedding pr mlp
        
        rows = {}
        order = 0
        for i, source_node in enumerate(source_nodes):
            name = source_node.module_name
            if name not in rows:
                rows[name] = {'nodes':[i], 'order':order}
                order+=1
            else:
                rows[name]['nodes'].append(i)

        data = []
        for name, v in rows.items():
            if any([s in name for s in ['_token','_pos']]):
                if not include_emb:
                    continue
                data_row = np.zeros((1,n_head))
                data_row[0,0] = source_scores[v['nodes'][0]]
                data.append(data_row)
            else:
                data_row = np.zeros((1,n_head))
                for i,i_node in enumerate(v['nodes']):
                    data_row[0,i] = source_scores[i_node]
                data.append(data_row)

        data = np.concatenate(data, axis=0)
        # data = einops.rearrange(data, ' (n_layer n_head) -> n_layer n_head', n_head=n_head)

        fig, ax = plt.subplots(1,1)

        data = np.flipud(data)
        yticklabels = list(rows.keys())
        yticklabels = yticklabels[::-1]
        yticklabels = yticklabels[:len(data)]
        xticklabels = [f'Head {i}' for i in range(data.shape[1])]

        extreme = np.max(np.abs(data))
        if mid_0==False or data.min()*data.max() >= 0:
            norm = None
        else:
            if balance:
                norm = TwoSlopeNorm(vmin=-extreme, vcenter=0, vmax=extreme)
            else:
                norm = TwoSlopeNorm(vmin=np.min(data), vcenter=0, vmax=np.max(data))
                
        # norm = TwoSlopeNorm(vmin=0, vcenter=0.05, vmax=data.max())

        handle=ax.imshow(data, cmap=cmap, norm=norm)
        fig.colorbar(handle, ax=ax)  # Associate the colorbar with the Axes
        ax.set_title(title)
        ax.set_xticks(np.arange(len(xticklabels)))
        ax.set_xticklabels(xticklabels, fontsize=5, rotation=90)
        ax.set_yticks(np.arange(len(yticklabels)))
        ax.set_yticklabels(yticklabels, fontsize=5, rotation=0)

        if display:
            plt.show()
        else:
            save_path = plot_dir / f'({sort_id}){title}.png'
            fig.savefig(save_path, dpi=400)
            # 
            plt.close(fig)
        # print(f'Saved {save_path}')
        return




  
        # analysis.decoder_to_umembedding(sender_names=[{'module':'*decoder*z_hook*','head':'*'}], rewrite=1)


    def run_minimal_circuit(self, circuit, rewrite=1):
        all_nodes = circuit.all_nodes
        used_nodes = circuit.used_nodes
        unused_nodes = np.setdiff1d(all_nodes, used_nodes)

        null_activations = torch.load(self.null_dataset_path)

        ablate_names = []
        for name in unused_nodes:
            parts = name.split('.')
            block = parts[0]
            attn = parts[1]
            layer = parts[2]
            head = parts[3]

            if block=='dec' and attn=='cross':
                attn='multi'

            ablate_name = MLC_utils.get_module_names_by_regex(self.net, [{'module':f'*{block}*{layer}*{attn}*z_hook*','head':f'{head}'}])
            ablate_names+=ablate_name
        cache, hooked_modules = hook_functions.add_hooks(self.net, mode='patch', hook_names=ablate_names, 
                                patch_activation=[null_activations[str(name)] for name in ablate_names])
        
        val_batch = next(iter(self.dataloader))
        output = MLC_utils.eval_model(val_batch, self.net, self.langs)
        [hook.remove_hooks() for hook in hooked_modules]

        print(output['yq_predict'])
        print(len(used_nodes))
        return output

        

    def prune_circuit(self, seed=0, rewrite=1):
        
        prune_path = self.plot_dir / 'prune_names'
        if prune_path.exists() and not rewrite:
            return torch.load(prune_path)

        null_activations = torch.load(self.null_dataset_path)

        net_prune_names = []

        # start from output
        val_batch = next(iter(self.dataloader))

        decoder_heads = MLC_utils.get_module_names_by_regex(self.net, [{'module':f'*decoder*z_hook*','head':'*'}])
        # shuffle the heads
        # np.random.seed(seed)
        decoder_heads = decoder_heads[::-1]


        for name in decoder_heads:
            _, hooked_modules = hook_functions.add_hooks(self.net, mode='patch', hook_names=net_prune_names+[name], 
                                    patch_activation=[null_activations[str(name)] for name in net_prune_names+[name]])
            output = MLC_utils.eval_model(val_batch, self.net, self.langs)
            [hook.remove_hooks() for hook in hooked_modules]

            if output['v_acc'].mean() == 1:
                net_prune_names+=[name]

        encoder_heads = MLC_utils.get_module_names_by_regex(self.net, [{'module':f'*encoder*z_hook*','head':'*'}])
        # encoder_heads = encoder_heads[::-1]
        for name in encoder_heads:
            _, hooked_modules = hook_functions.add_hooks(self.net, mode='patch', hook_names=net_prune_names+[name], 
                                    patch_activation=[null_activations[str(name)] for name in net_prune_names+[name]])
            output = MLC_utils.eval_model(val_batch, self.net, self.langs)
            [hook.remove_hooks() for hook in hooked_modules]

            if output['v_acc'].mean() == 1:
                net_prune_names+=[name]

        # convert net names to circuit names
        circuit_prune_names = []
        for name in net_prune_names:
            layer = name['module'].split('.')[3]
            head = name['head']
            if 'self_attn' in name['module']:
                attn = 'self'
            elif 'multihead_attn' in name['module']:
                attn = 'cross'

            if 'encoder' in name['module']:
                block = 'enc'
            elif 'decoder' in name['module']:
                block = 'dec'

            circuit_prune_names.append((f'{block}.{attn}.{layer}.{head}'))

        prune_names = {
            'circuit_name': circuit_prune_names,
            'net_name': net_prune_names
        }

        print(len(net_prune_names))
        torch.save(prune_names, prune_path)

        return prune_names



    def run_path_patching(self, circuit: dict={}, sequential=True, metric=None, eval_func=None, ablate_all=0, rewrite=0):
        

        # assert mode in ['out', 'z','q','k','v'], 'mode must be z, q, k or v'
        assert metric in ['loss', 'logit', 'customized']

        null_activations = torch.load(self.null_dataset_path)
        net = self.net
        langs = self.langs
        # perma_ablate_names = get_module_names_by_regex(net, 
        #                                                [{'module':'*encoder*layer*0*z_hook*','head':'3'},
        #                                                 {'module':'*encoder*layer*0*z_hook*','head':'4'}
        #                                                 ])



        sender_nodes = copy.deepcopy(circuit['sender_nodes']) # a list of nodes
        sender_modes = copy.deepcopy(circuit['sender_modes']) # a list of modes
        receiver_nodes = copy.deepcopy(circuit['receiver_nodes']) # a list of nodes
        receiver_modes = copy.deepcopy(circuit['receiver_modes']) # a list of modes
        problems_chain = copy.deepcopy(circuit['problems_chain'])
        # freeze_nodes = copy.deepcopy(circuit.get('knockouts', []))
        knockout_nodes = copy.deepcopy(circuit.get('knockout_nodes', []))
        knockout_modes = copy.deepcopy(circuit.get('knockout_modes', []))
        ablate_all_list = copy.deepcopy(ablate_all)


        sender_names = []
        for i in range(len(sender_nodes)):
            sender_names += self.circuit_name_to_net_name(sender_nodes[i], suffix=f'*{sender_modes[i]}_hook*')

        receiver_names_chain = []
        for i in range(len(receiver_nodes)):
            if receiver_modes[i]=='out':
                continue

            elif receiver_modes[i] in ['q','k','v']:
                receiver_names_chain+=self.circuit_name_to_net_name(receiver_nodes[i], suffix=f'*{receiver_modes[i]}_hook*')

        knockout_names = []
        for i in range(len(knockout_nodes)):
            knockout_names += self.circuit_name_to_net_name(knockout_nodes[i], suffix=f'*{knockout_modes[i]}_hook*')

        freeze_names = MLC_utils.get_module_names_by_regex(self.net, [
            {'module':'*q_hook*', 'head':'*'},
            {'module':'*k_hook*', 'head':'*'},
            {'module':'*v_hook*', 'head':'*'}
            ])
        

        """
        both encoder and decoder share the same pos encoding
        when ablating one side, keep the other side intact
        """
        if any(['encoder_pos' in node.module_name for node in sender_nodes]):
            freeze_names+=MLC_utils.get_module_names_by_regex(net, [{'module':'*decoder*0*resid_pre_hook*', 'head':'*'}])
        if any(['decoder_pos' in node.module_name for node in sender_nodes]):
            freeze_names+=MLC_utils.get_module_names_by_regex(net, [{'module':'*encoder*0*resid_pre_hook*', 'head':'*'}])
        
        do_freeze_mlp = 0
        if do_freeze_mlp:
            freeze_names += MLC_utils.get_module_names_by_regex(net, [{'module':'*mlp*hook*','head':'*'}])


        # clean input
        val_batch = next(iter(self.dataloader))

        #------first run------
        # forward clean run
        all_hook_names = MLC_utils.get_module_names_by_regex(self.net, [{'module':'*hook*', 'head':'*'}])
        cache_clean, hooked_cache_modules = hook_functions.add_hooks(net, mode='cache', hook_names=all_hook_names)
        output_clean = MLC_utils.eval_model(val_batch, net, langs)
        [hooked_module.remove_hooks() for hooked_module in hooked_cache_modules]

        if metric=='loss':
            metric_clean = output_clean['loss']
        elif metric=='logit':
            metric_clean = output_clean['logits_correct']  
        elif metric=='customized':
            metric_clean = eval_func(analyzer=self, cache_patch=cache_clean, output_patch=output_clean)



        # get corrupted activation at the sender    

        cache_patch = null_activations
        first_sender = 1
        if not isinstance(ablate_all_list, list):
            ablate_all_list = [ablate_all_list]*len(receiver_names_chain)

        if sequential:
            for i in range(len(receiver_names_chain)+1):
                receiver_names = [receiver_names_chain[i]] if i<len(receiver_names_chain) else []

                print(f'receivers_chain: {receiver_names}, sender: {sender_names}')
                corrupt_sender_activation = [cache_patch[str(name)] for name in sender_names]
                if first_sender and ablate_all_list[i] :
                    # if the first sender, don't patch the sender activity
                    first_sender = 0
                    sender_modules = [] 
                else:
                    first_sender = 0
                    _, sender_modules = hook_functions.add_hooks(net, mode='patch', hook_names=sender_names, 
                                                                patch_activation=corrupt_sender_activation)

                # patch clean activations to the freeze hooks
                # need to remove senders and receiver from frozen hooks
                freeze_names_exclusive = [name for name in freeze_names if name not in sender_names+receiver_names+knockout_names]
                clean_freeze_activation = [cache_clean.cache[str(name)] for name in freeze_names_exclusive]
                _, freeze_modules = hook_functions.add_hooks(net, mode='patch', hook_names=freeze_names_exclusive,
                                                            patch_activation=clean_freeze_activation)
                
                # knockout patch
                knockout_names = []
                
                problem = problems_chain[i] if i<len(problems_chain) else False

                if problem and ablate_all_list[i]:
                    # knockout all the source nodes other than sender
                    
                    source_nodes = problem.source_stream.source_nodes
                    
                    for source_node in (source_nodes):
                        
                        node_in_sender = 0
                        for sender_node in sender_nodes:
                            module_equal = source_node.module_name==sender_node.module_name
                            head_equal = source_node.head==sender_node.head
                            if module_equal and head_equal:
                                node_in_sender = 1
                                break

                        if not node_in_sender:
                            knockout_name = self.circuit_name_to_net_name(source_node, suffix=f'*z_hook*')
                            knockout_names+=knockout_name


                _, knockout_modules = hook_functions.add_hooks(net, mode='patch', hook_names=knockout_names, 
                                            patch_activation=[null_activations[str(name)] for name in knockout_names])
                
                # cache the receiver activations
                cache_patch, receiver_modules = hook_functions.add_hooks(net, mode='cache', hook_names=all_hook_names)
                output_patch = MLC_utils.eval_model(val_batch, net, langs)
                [hooked_module.remove_hooks() for hooked_module in sender_modules+freeze_modules+knockout_modules+receiver_modules]

                if metric=='loss':
                    metric_patch = output_patch['loss']
                elif metric=='logit':
                    metric_patch = output_patch['logits_correct']
                elif metric=='customized':
                    metric_patch = eval_func(analyzer=self, cache_patch=cache_patch, output_patch=output_patch)



                sender_nodes = [receiver_nodes[i]] if i<len(receiver_nodes) else []
                sender_names = [receiver_names_chain[i]] if i<len(receiver_names_chain) else []
                cache_patch = cache_patch.cache
            
        else:
            #------second run------

            loss_diff = []
            pred_tokens_patch = []

            corrupt_sender_activation = [null_activations[str(name)] for name in sender_names]
            _, sender_modules = hook_functions.add_hooks(net, mode='patch', hook_names=sender_names, 
                                                        patch_activation=corrupt_sender_activation)

            # patch clean activations to the freeze hooks
            # need to remove senders and receiver from frozen hooks
            freeze_names_exclusive = [name for name in freeze_names if name not in sender_names+receiver_names_chain+knockout_names]
            clean_freeze_activation = [cache_clean.cache[str(name)] for name in freeze_names_exclusive]
            _, freeze_modules = hook_functions.add_hooks(net, mode='patch', hook_names=freeze_names_exclusive,
                                                        patch_activation=clean_freeze_activation)
            
            # knockout patch
            _, knockout_modules = hook_functions.add_hooks(net, mode='patch', hook_names=knockout_names, 
                                        patch_activation=[null_activations[str(name)] for name in knockout_names])
            
            # cache the receiver activations
            cache_patch, receiver_modules = hook_functions.add_hooks(net, mode='cache', hook_names=all_hook_names)

            output_patch = MLC_utils.eval_model(val_batch, net, langs)

            [hooked_module.remove_hooks() for hooked_module in sender_modules+freeze_modules+knockout_modules+receiver_modules]

            if metric=='loss':
                metric_patch = output_patch['loss']
            elif metric=='logit':
                metric_patch = output_patch['logits_correct']
            elif metric=='customized':
                metric_patch = eval_func(model=self, cache_patch=cache_patch, output_patch=output_patch)
            # if no receiver, the second run is the last run
            # no third run

        return metric_clean, metric_patch





    def compute_VAF(self, problem):
        
        target_activation = problem.source_stream.cached_activation
        source_nodes, source_activations = problem.source_stream.forward()

        arg_q = problem.arg_q
        n_batch = target_activation.shape[0]
        
        is_encoder = 'enc' in problem.current_node.module_name
        is_kv = problem.mode in ['k','v']
        if (not is_encoder) and (not is_kv):

            target_activation = np.take_along_axis(target_activation.squeeze(), arg_q[:,:,None], axis=1)
            source_activations = [np.take_along_axis(source_activation.squeeze(), arg_q[:,:,None], axis=1) 
                                  for source_activation in source_activations]

        if target_activation.ndim>2:
            d_model = target_activation.shape[-1]
            target_activation = target_activation.reshape(-1, d_model)
            source_activations = [source_activation.reshape(-1, d_model) for source_activation in source_activations]

        def variance_contribution(X, Y):
            """
            Computes the variance contribution of a single vector X to Y.

            Parameters:
            - X: (n_samples, d) A single contributing vector.
            - Y: (n_samples, d) The summed vector.

            Returns:
            - contribution: A scalar representing how much variance in Y is explained by X.
            """
            var_Y = np.var(Y)  # Total variance of Y
            var_X = np.var(X)  # Variance of X
            cov_XY = np.cov(X.flatten(), (Y-X).flatten())[0, 1]  # Covariance of X with the rest of Y

            # Compute variance contribution
            contribution = (var_X + 2 * cov_XY) / var_Y if var_Y > 0 else 0
            return contribution





        # Fit linear regression model
        vaf_scores = []
        Y = target_activation
        for source_activation in source_activations:
            X = source_activation

            if False:
                model = LinearRegression()
                model.fit(X, Y)
                Y_pred = model.predict(X)
                variance_explained = r2_score(Y, Y_pred)
            
            if 1:
                variance_explained = variance_contribution(X, Y)

            vaf_scores.append(variance_explained)
        vaf_scores = np.array(vaf_scores)

        return vaf_scores

    def compute_MI(self, problem):

        target_activations = problem.source_stream.cached_activation
        target_labels = np.array(problem.target_labels)
        vector_ids = problem.target_args['vector_ids']
        vector_labels = problem.target_args['vector_labels']
        source_nodes, source_activations = problem.source_stream.forward()
        

        arg_q = problem.arg_q
        n_batch = target_activations.shape[0]
        mi_scores = []
        for source_activation in source_activations:
            source_activation = source_activation.squeeze()
            X = []
            labels = []
            for b in range(n_batch):
                X.append(source_activation[b,vector_ids[b],:])
                labels+=vector_labels[b]

            X = np.vstack(X)
            labels = np.vstack(labels).squeeze()
            mi = mutual_info_regression(X, labels.ravel()).mean()
            mi_scores.append(mi)

        mi_scores = np.array(mi_scores)

        return mi_scores
    
    def compute_inner(self, problem):

        target_activations = problem.target_vectors
        target_labels = np.array(problem.target_labels)

        source_nodes, source_activations = problem.source_stream.forward()
        

        arg_q = problem.arg_q
        n_batch = target_activations.shape[0]
        inner_scores = []            
        target_activation = np.take_along_axis(target_activations.squeeze(), target_labels[:,:,None], axis=1)
        for source_activation in source_activations:
            source_activation = source_activation.squeeze()
            source_activation = np.take_along_axis(source_activation, target_labels[:,:,None], axis=1)
            inner = einops.einsum(target_activation, source_activation, 
                                  'batch n_arg d , batch n_arg d->batch n_arg').mean()
            inner_scores.append(inner)

        inner_scores = np.array(inner_scores)

        return inner_scores
    def analyze_dec_self_1_3(self):
        output = self.output
        n_batch = len(output['yq_predict'])
        block = 'dec'
        layer = 1
        type = 'self'
        head = 5
        attention = self.graph[block, layer, type, head].attention_score
        arg_max_k = attention[np.arange(n_batch),0,0,:].argmax(-1)
        v = self.graph[block, layer, type, head].v_stream.cached_activation[np.arange(n_batch),0,0,:]
        # k = self.graph[block, layer, type, head].k_stream.cached_activation[np.arange(n_batch),0,arg_max_k,:]

        """
        Hypothesis: 
        the V of dec.self.1.3 should contain relative pos information 1st output's color's symbol to function
        """

        # check the behavior is correct on the first predicted token
        k_argmax_symbol = []
        yq_first = []
        for b in range(n_batch):
            xq_context = np.array(output['xq_context'][b])
            yq = output['yq'][b]
            k_argmax_symbol.append(xq_context[arg_max_k[b]])
            yq_first.append(yq[0])
        k_argmax_symbol = np.array(k_argmax_symbol)
        yq_first = np.array(yq_first)

        acc = (k_argmax_symbol==yq_first).mean()
        print(f'First attended token accuracy: {acc}')

        k_label=[]
        q_label=[]
        exclude = []

        for b in range(n_batch):
            xq_context = np.array(output['xq_context'][b])
            yq_predict = output['yq_predict'][b]
            yq = output['yq'][b]
            v_acc = output['v_acc'][b]
            if v_acc==0:
                exclude.append(b)
                continue
            

            
            grammar_str = output['grammar'][b]['aux']['grammar_str']
            grammar_dict = MLC_utils.grammar_to_dict(grammar_str)

            # exclude trials with more than 1 query function
            pos_1st_sos = np.where(xq_context=='SOS')[0][0]

            # keep only trials with only one function in query
            query_functions = [token for i, token in enumerate(xq_context[:pos_1st_sos]) if token in grammar_dict[2].keys()]
            if len(query_functions)!=1:
                exclude.append(b)
                continue

            query_func = query_functions[0]
            query_func_pos = [i for i, token in enumerate(xq_context[:pos_1st_sos]) if token==query_func][0]

            # find 1st colors after function
            func_poses_all = [i for i, token in enumerate(xq_context) if token==query_func]
            func_poses_single = []
            sos_poses = np.where(xq_context=='SOS')[0]

            for pos in func_poses_all:
                sos_before = sos_poses[sos_poses<pos]
                if len(sos_before)==0:
                    sos_before = 0
                else:
                    sos_before = sos_before[-1]
                sos_after = sos_poses[sos_poses>pos][0]
                n_func = 0
                for s in xq_context[sos_before:sos_after]: # centered around the function
                    if s in grammar_dict[2].keys():
                        n_func+=1
                if n_func==1:
                    func_poses_single.append(pos)

            # exclude trials with no single function demonstrations
            if len(func_poses_single)<=1:
                exclude.append(b)
                continue
            func_pos_single = func_poses_single[-1]


            # find k's color's symbol's pos 
            k_symbol = grammar_dict[1][yq[0]]
            k_symbol_pos = [i for i, token in enumerate(xq_context[:pos_1st_sos]) if token==k_symbol][-1]
            k_label.append(k_symbol_pos-query_func_pos)




        k_label = np.array(k_label)
        # Combine datasets for joint PCA

        # only plot not excluded data
        v = v[~np.isin(np.arange(n_batch), exclude)]

        # print(f'Relative pos equal: {(q_label==k_label).mean()}')


        cmap = plt.cm.Spectral
        unique_pos = np.union1d(q_label,k_label)
        n_colors = unique_pos.shape[0] 
        colors = cmap(np.linspace(0, 1, n_colors))


        combined_data = v

        # Perform PCA
        pca = PCA(n_components=2)  # Reduce to 2D for visualization
        combined_pca = pca.fit_transform(combined_data)

        # Separate the PCA-transformed datasets
        # set1_pca = combined_pca[:q.shape[0]]
        set2_pca = combined_pca

        # Visualization
        fig, ax = plt.subplots(figsize=(8, 6))
        # ax.scatter(set1_pca[:, 0], set1_pca[:, 1], label='Q', alpha=0.7, marker='o', color=[colors[l] for l in q_label])
        ax.scatter(set2_pca[:, 0], set2_pca[:, 1], label='V', alpha=0.7, marker='x', color=[colors[l] for l in k_label])
        # # draw lines connecting paired points
        # for i in range(set1_pca.shape[0]):
        #     ax.plot([set1_pca[i, 0], set2_pca[i, 0]], [set1_pca[i, 1], set2_pca[i, 1]], 'k-', alpha=0.3, linewidth=0.5, linestyle='--')
        
        color_handles = [
            mpatches.Patch(color=colors[label], label=f'{unique_pos[label]}')
            for label in range(n_colors)
        ]
    
        color_legend=ax.legend(handles=color_handles, loc='upper right', title='Legend', fontsize='small')
        ax.add_artist(color_legend)         
        # marker_handles = [
        #     mlines.Line2D([], [], color='gray', marker='o', linestyle='None', markersize=8, label='Q'),
        #     mlines.Line2D([], [], color='gray', marker='x', linestyle='None', markersize=8, label='K')
        # ]
        # ax.legend(handles=marker_handles, loc='upper left', title='Marker Legend', fontsize='small')

        # Formatting
        ax.set_title('Relative pos to function')
        ax.set_xlabel('PC 1')
        ax.set_ylabel('PC 2')
        ax.grid(True)
        plt.show()
        a=1


    def analyze_enc_self_1_0(self, block, layer, type, head):
        output = self.output
        n_batch = len(output['yq_predict'])

        attention = self.graph[block, layer, type, head].attention_score
        arg_max_k = attention[np.arange(n_batch),0,1,:].argmax(-1)
        cached_v = self.graph[block, layer, type, head].v_stream.cached_activation
        cached_q = self.graph[block, layer, type, head].q_stream.cached_activation
        cached_k = self.graph[block, layer, type, head].k_stream.cached_activation[np.arange(n_batch),0,arg_max_k,:]

        """
        Hypothesis: 
        the V symbols contain relative pos info to the nearest function
        """

        # check if the attended token is the first one after IO
        k_argmax_symbol = []
        # yq_first = []
        # for b in range(n_batch):
        #     xq_context = np.array(output['xq_context'][b])
        #     # yq = output['yq'][b]
        #     k_argmax_symbol.append(xq_context[arg_max_k[b]])
        #     # yq_first.append(yq[0])
        # k_argmax_symbol = np.array(k_argmax_symbol)
        # yq_first = np.array(yq_first)

        # acc = (k_argmax_symbol==yq_first).mean()
        # print(f'First attended token accuracy: {acc}')

        k_label=[]
        q_label=[]
        exclude = []

        v = []
        v_label = []
        for b in range(n_batch):
            xq_context = np.array(output['xq_context'][b])
            yq = output['yq'][b]
            v_acc = output['v_acc'][b]
            if v_acc==0:
                exclude.append(b)
                continue
            

            
            grammar_str = output['grammar'][b]['aux']['grammar_str']
            grammar_dict = MLC_utils.grammar_to_dict(grammar_str)

            # exclude trials with more than 1 query function
            pos_1st_sos = np.where(xq_context=='SOS')[0][0]

            # keep only trials with only one function in query
            query_functions = [token for i, token in enumerate(xq_context[:pos_1st_sos]) if token in grammar_dict[2].keys()]
            if len(query_functions)!=1:
                exclude.append(b)
                continue

            query_func = query_functions[0]

            # find 1st colors after function
            func_poses_all = [i for i, token in enumerate(xq_context) if token==query_func]
            func_poses_single = []
            sos_poses = np.where(xq_context=='SOS')[0]

            for pos_symbol in func_poses_all:
                sos_before = sos_poses[sos_poses<pos_symbol]
                if len(sos_before)==0:
                    sos_before = 0
                else:
                    sos_before = sos_before[-1]
                sos_after = sos_poses[sos_poses>pos_symbol][0]
                n_func = 0
                for s in xq_context[sos_before:sos_after]: # centered around the function
                    if s in grammar_dict[2].keys():
                        n_func+=1
                if n_func==1:
                    func_poses_single.append(pos_symbol)

            # exclude trials with no single function demonstrations
            if len(func_poses_single)<=1:
                exclude.append(b)
                continue
            
            # symbols pos
            pos_all_symbols = [i for i, token in enumerate(xq_context) if token in grammar_dict[0].keys()]
            query_func_poses = [i for i, token in enumerate(xq_context[:pos_1st_sos]) if token==query_func]
            # distance to the first function in the SOS chunck
            pos_all_io = np.array([i for i, token in enumerate(xq_context) if token=='IO'])

            for pos_symbol in pos_all_symbols:
                sos_before, sos_after = sos_around_pos(xq_context, pos_symbol)
                # pos_1st_func = [i for i,w in enumerate(xq_context[sos_before:sos_after]) if w in grammar_dict[2].keys()][0]
                # pos_1st_func+=sos_before

                pos_io = pos_all_io[pos_all_io>pos_symbol][0]
                v_label.append(pos_symbol-sos_before)

            v.append(cached_v[b,0,pos_all_symbols,:])

            # IO_poses = np.array([i for i, token in enumerate(xq_context) if token=='IO'])
            # IO_pos_before = IO_poses[IO_poses<[arg_max_k[b]]][-1]
            # k_label.append(arg_max_k[b]-IO_pos_before)






        v = np.vstack(v)
        v_label = np.array(v_label)
        k_label = np.array(k_label)
        print(k_label.mean())
        # Combine datasets for joint PCA

        # only plot not excluded data
        # q = q[~np.isin(np.arange(n_batch), exclude)]
        # k = k[~np.isin(np.arange(n_batch), exclude)]
        # v = v[~np.isin(np.arange(n_batch), exclude)]

        plot_2D(source_data=[v], source_legends=['v'],color_labels=[v_label])

        plot_2D(source_data=[q,k], source_legends=['q','k'],color_labels=[k_label, k_label])

        a=1

    def analyze_enc_self_0_6(self, block, layer, type, head):
        output = self.output
        n_batch = len(output['yq_predict'])

        attention = self.graph[block, layer, type, head].attention_score
        arg_max_k = attention[np.arange(n_batch),0,1,:].argmax(-1)
        cached_v = self.graph[block, layer, type, head].v_stream.cached_activation
        cached_q = self.graph[block, layer, type, head].q_stream.cached_activation
        cached_k = self.graph[block, layer, type, head].k_stream.cached_activation[np.arange(n_batch),0,arg_max_k,:]

        """
        Hypothesis: 
        the V symbols contain relative pos info to the nearest function
        """

        # check if the attended token is the first one after IO
        k_argmax_symbol = []
        # yq_first = []
        # for b in range(n_batch):
        #     xq_context = np.array(output['xq_context'][b])
        #     # yq = output['yq'][b]
        #     k_argmax_symbol.append(xq_context[arg_max_k[b]])
        #     # yq_first.append(yq[0])
        # k_argmax_symbol = np.array(k_argmax_symbol)
        # yq_first = np.array(yq_first)

        # acc = (k_argmax_symbol==yq_first).mean()
        # print(f'First attended token accuracy: {acc}')

        k_label=[]
        q_label=[]    
        exclude = []

        v = []
        v_label = []
        for b in range(n_batch):
            xq_context = np.array(output['xq_context'][b])
            yq = output['yq'][b]
            v_acc = output['v_acc'][b]
            if v_acc==0:
                exclude.append(b)
                continue
            

            
            grammar_str = output['grammar'][b]['aux']['grammar_str']
            grammar_dict = MLC_utils.grammar_to_dict(grammar_str)

            # exclude trials with more than 1 query function
            pos_1st_sos = np.where(xq_context=='SOS')[0][0]

            # keep only trials with only one function in query
            query_functions = [token for i, token in enumerate(xq_context[:pos_1st_sos]) if token in grammar_dict[2].keys()]
            if len(query_functions)!=1:
                exclude.append(b)
                continue

            query_func = query_functions[0]

            # find 1st colors after function
            func_poses_all = [i for i, token in enumerate(xq_context) if token==query_func]
            func_poses_single = []
            sos_poses = np.where(xq_context=='SOS')[0]

            for pos_symbol in func_poses_all:
                sos_before = sos_poses[sos_poses<pos_symbol]
                if len(sos_before)==0:
                    sos_before = 0
                else:
                    sos_before = sos_before[-1]
                sos_after = sos_poses[sos_poses>pos_symbol][0]
                n_func = 0
                for s in xq_context[sos_before:sos_after]: # centered around the function
                    if s in grammar_dict[2].keys():
                        n_func+=1
                if n_func==1:
                    func_poses_single.append(pos_symbol)

            # exclude trials with no single function demonstrations
            if len(func_poses_single)<=1:
                exclude.append(b)
                continue
            
            # symbols pos
            pos_all_symbols = [i for i, token in enumerate(xq_context) if token in grammar_dict[0].keys()]
            query_func_poses = [i for i, token in enumerate(xq_context[:pos_1st_sos]) if token==query_func]
            # distance to the first function in the SOS chunck
            pos_all_io = np.array([i for i, token in enumerate(xq_context) if token=='IO'])

            for pos_symbol in pos_all_symbols:
                sos_before, sos_after = sos_around_pos(xq_context, pos_symbol)
                # pos_1st_func = [i for i,w in enumerate(xq_context[sos_before:sos_after]) if w in grammar_dict[2].keys()][0]
                # pos_1st_func+=sos_before

                pos_io = pos_all_io[pos_all_io>pos_symbol][0]
                v_label.append(pos_symbol-sos_before)

            v.append(cached_v[b,0,pos_all_symbols,:])

            # IO_poses = np.array([i for i, token in enumerate(xq_context) if token=='IO'])
            # IO_pos_before = IO_poses[IO_poses<[arg_max_k[b]]][-1]
            # k_label.append(arg_max_k[b]-IO_pos_before)






        v = np.vstack(v)
        v_label = np.array(v_label)
        k_label = np.array(k_label)
        print(k_label.mean())
        # Combine datasets for joint PCA

        # only plot not excluded data
        # q = q[~np.isin(np.arange(n_batch), exclude)]
        # k = k[~np.isin(np.arange(n_batch), exclude)]
        # v = v[~np.isin(np.arange(n_batch), exclude)]

        plot_2D(source_data=[v], source_legends=['v'],color_labels=[v_label])

        plot_2D(source_data=[q,k], source_legends=['q','k'],color_labels=[k_label, k_label])

        a=1

    def analyze_enc_self_0_5(self, block, layer, type, head):
        output = self.output
        n_batch = len(output['yq_predict'])

        attention = self.graph[block, layer, type, head].attention_score
        arg_max_k = attention[np.arange(n_batch),0,1,:].argmax(-1)
        cached_v = self.graph[block, layer, type, head].v_stream.cached_activation
        cached_z = MLC_utils.get_activations_by_regex(net=self.net, 
                                                    cache=self.cache,
                                                    hook_regex=[{'module':f'*{block}*{layer}*{type}*z_hook*', 'head': f'{head}'}]
                                                    )[0]
        # cached_q = self.graph[block, layer, type, head].q_stream.cached_activation
        # cached_k = self.graph[block, layer, type, head].k_stream.cached_activation[np.arange(n_batch),0,arg_max_k,:]

        """
        Hypothesis: 
        the V symbols contain relative pos info to the nearest function
        """

        # check if the attended token is the first one after IO
        k_argmax_symbol = []
        # yq_first = []
        # for b in range(n_batch):
        #     xq_context = np.array(output['xq_context'][b])
        #     # yq = output['yq'][b]
        #     k_argmax_symbol.append(xq_context[arg_max_k[b]])
        #     # yq_first.append(yq[0])
        # k_argmax_symbol = np.array(k_argmax_symbol)
        # yq_first = np.array(yq_first)

        # acc = (k_argmax_symbol==yq_first).mean()
        # print(f'First attended token accuracy: {acc}')

        k_label=[]
        q_label=[]    
        exclude = []

        z = []
        z_label = []
        for b in range(n_batch):
            xq_context = np.array(output['xq_context'][b])
            yq = output['yq'][b]
            v_acc = output['v_acc'][b]
            if v_acc==0:
                exclude.append(b)
                continue
            

            
            grammar_str = output['grammar'][b]['aux']['grammar_str']
            grammar_dict = MLC_utils.grammar_to_dict(grammar_str)

            # exclude trials with more than 1 query function
            pos_1st_sos = np.where(xq_context=='SOS')[0][0]

            # keep only trials with only one function in query
            query_functions = [token for i, token in enumerate(xq_context[:pos_1st_sos]) if token in grammar_dict[2].keys()]
            if len(query_functions)!=1:
                exclude.append(b)
                continue

            query_func = query_functions[0]

            # find 1st colors after function
            func_poses_all = [i for i, token in enumerate(xq_context) if token==query_func]
            func_poses_single = []
            sos_poses = np.where(xq_context=='SOS')[0]

            for pos_symbol in func_poses_all:
                sos_before = sos_poses[sos_poses<pos_symbol]
                if len(sos_before)==0:
                    sos_before = 0
                else:
                    sos_before = sos_before[-1]
                sos_after = sos_poses[sos_poses>pos_symbol][0]
                n_func = 0
                for s in xq_context[sos_before:sos_after]: # centered around the function
                    if s in grammar_dict[2].keys():
                        n_func+=1
                if n_func==1:
                    func_poses_single.append(pos_symbol)

            # exclude trials with no single function demonstrations
            if len(func_poses_single)<=1:
                exclude.append(b)
                continue
            
            # all query symbols 
            query_tokens = xq_context[:pos_1st_sos]

            for token_pos, token in enumerate(query_tokens):
                all_token_pos = np.array([i for i, t in enumerate(xq_context) if t==token])
                
                z.append(cached_z[b,0,all_token_pos,:])
                z_label+=[token_pos]*all_token_pos.shape[0]

            # pos_all_symbols = [i for i, token in enumerate(xq_context) if token in grammar_dict[0].keys()]
            # query_func_poses = [i for i, token in enumerate(xq_context[:pos_1st_sos]) if token==query_func]
            # # distance to the first function in the SOS chunck
            # pos_all_io = np.array([i for i, token in enumerate(xq_context) if token=='IO'])

            # for pos_symbol in pos_all_symbols:
            #     sos_before, sos_after = sos_around_pos(xq_context, pos_symbol)
            #     # pos_1st_func = [i for i,w in enumerate(xq_context[sos_before:sos_after]) if w in grammar_dict[2].keys()][0]
            #     # pos_1st_func+=sos_before

            #     pos_io = pos_all_io[pos_all_io>pos_symbol][0]
            #     v_label.append(pos_symbol-sos_before)

            # v.append(cached_v[b,0,pos_all_symbols,:])

            # IO_poses = np.array([i for i, token in enumerate(xq_context) if token=='IO'])
            # IO_pos_before = IO_poses[IO_poses<[arg_max_k[b]]][-1]
            # k_label.append(arg_max_k[b]-IO_pos_before)






        z = np.vstack(z)
        z_label = np.array(z_label)
        k_label = np.array(k_label)
        print(k_label.mean())
        # Combine datasets for joint PCA

        # only plot not excluded data
        # q = q[~np.isin(np.arange(n_batch), exclude)]
        # k = k[~np.isin(np.arange(n_batch), exclude)]
        # v = v[~np.isin(np.arange(n_batch), exclude)]
        fig, ax = plt.subplots(1,1,figsize=(8,6))
        plot_2D(ax=ax, source_data=[z], source_legends=['v'],color_labels=[z_label])

        # plot_pca(data=[q,k], source_labels=['q','k'],color_labels=[k_label, k_label])
        plt.show()

        a=1

    def analyze_enc_self_1_1(self, block, layer, type, head):
        output = self.output
        n_batch = len(output['yq_predict'])

        attention = self.graph[block, layer, type, head].attention_score
        arg_max_k = attention[np.arange(n_batch),0,1,:].argmax(-1)
        cached_v = self.graph[block, layer, type, head].v_stream.cached_activation
        cached_z = MLC_utils.get_activations_by_regex(net=self.net, 
                                                    cache=self.cache,
                                                    hook_regex=[{'module':f'*{block}*{layer}*{type}*z_hook*', 'head': f'{head}'}]
                                                    )[0]
        # cached_q = self.graph[block, layer, type, head].q_stream.cached_activation
        # cached_k = self.graph[block, layer, type, head].k_stream.cached_activation[np.arange(n_batch),0,arg_max_k,:]

        """
        Hypothesis: 
        the V symbols contain relative pos info to the nearest function
        """


        k_label=[]
        q_label=[]    
        exclude = []

        z = []
        z_label = []

        v = []
        v_label = []
        for b in range(n_batch):
            xq_context = np.array(output['xq_context'][b])
            yq = output['yq'][b]
            v_acc = output['v_acc'][b]
            if v_acc==0:
                exclude.append(b)
                continue
            

            
            grammar_str = output['grammar'][b]['aux']['grammar_str']
            grammar_dict = MLC_utils.grammar_to_dict(grammar_str)

            # exclude trials with more than 1 query function
            pos_1st_sos = np.where(xq_context=='SOS')[0][0]

            # keep only trials with only one function in query
            query_functions = [token for i, token in enumerate(xq_context[:pos_1st_sos]) if token in grammar_dict[2].keys()]
            if len(query_functions)!=1:
                exclude.append(b)
                continue

            query_func = query_functions[0]

            # find 1st colors after function
            func_poses_all = [i for i, token in enumerate(xq_context) if token==query_func]
            func_poses_single = []
            sos_poses = np.where(xq_context=='SOS')[0]

            for pos_symbol in func_poses_all:
                sos_before = sos_poses[sos_poses<pos_symbol]
                if len(sos_before)==0:
                    sos_before = 0
                else:
                    sos_before = sos_before[-1]
                sos_after = sos_poses[sos_poses>pos_symbol][0]
                n_func = 0
                for s in xq_context[sos_before:sos_after]: # centered around the function
                    if s in grammar_dict[2].keys():
                        n_func+=1
                if n_func==1:
                    func_poses_single.append(pos_symbol)

            # exclude trials with no single function demonstrations
            if len(func_poses_single)<=1:
                exclude.append(b)
                continue
            
            # all query symbols 
            """
            Token: Union[Symbol, Function]
            Symbol: token for color
            """
            query_tokens = xq_context[:pos_1st_sos]
            query_symbols = [token for token in query_tokens if token in grammar_dict[0].keys()]

            for symbol_pos, symbol in enumerate(query_symbols):
                all_symbol_pos = np.array([i for i, t in enumerate(xq_context) if t==symbol])
                
                v.append(cached_v[b,0,all_symbol_pos,:])
                v_label+=[symbol_pos]*all_symbol_pos.shape[0]

            query_colors = [grammar_dict[0][token] for token in query_symbols]
            for color in query_colors:
                all_color_pos = np.array([i for i, t in enumerate(xq_context) if t==color])
                z.append(cached_z[b,0,all_color_pos,:])
                symbol = grammar_dict[1][color]
                symbol_pos = [i for i, token in enumerate(query_tokens) if token==symbol][0]
                z_label+=[symbol_pos]*all_color_pos.shape[0]

        v = np.vstack(v)
        z = np.vstack(z)
        v_label = np.array(v_label)
        z_label = np.array(z_label)
        k_label = np.array(k_label)

        print(k_label.mean())
        # Combine datasets for joint PCA

        # only plot not excluded data
        # q = q[~np.isin(np.arange(n_batch), exclude)]
        # k = k[~np.isin(np.arange(n_batch), exclude)]
        # v = v[~np.isin(np.arange(n_batch), exclude)]
        fig, ax = plt.subplots(1,2,figsize=(8,6))
        plot_2D(ax=ax[0], source_data=[v], source_legends=['v'],color_labels=[v_label])

        plot_2D(ax=ax[1], source_data=[z], source_legends=['z'],color_labels=[z_label])

        # plot_pca(data=[q,k], source_labels=['q','k'],color_labels=[k_label, k_label])
        plt.show()


    def backtrack_dec_cross_1_5(self, block, layer, type, head, rewrite=0):
        save_dir = self.plot_dir/'dec_cross_1_5'
        save_dir.mkdir(exist_ok=True, parents=True)
        backtrack_plot_dir = save_dir / 'BT_k_plot'
        backtrack_data_dir = save_dir / 'BT_k_data'

        if backtrack_plot_dir.exists() and rewrite:
            shutil.rmtree(backtrack_plot_dir)
            shutil.rmtree(backtrack_data_dir)

        backtrack_plot_dir.mkdir(exist_ok=True, parents=True)
        backtrack_data_dir.mkdir(exist_ok=True, parents=True)
        output = self.output
        n_batch = len(output['yq_predict'])
        node = self.graph[block, layer, type, head]
        attention = node.attention_score
        arg_pred = np.arange(3)
        arg_max_k = attention[np.arange(n_batch),0,:len(arg_pred),:].argmax(-1)


        q_labels=[]
        k_labels=[]
        q_ids=[]
        k_ids=[]
        # get the q, k seq_label
        for b in range(n_batch):
            xq_context = np.array(output['xq_context'][b])
            yq = output['yq'][b]
            arg_max_kb = arg_max_k[b]
            arg_max_kb[arg_max_kb>=len(xq_context)]=0
            pred_tokens_enc = xq_context[arg_max_kb]
            pos_1st_sos = np.where(xq_context=='SOS')[0][0]
            query_tokens = xq_context[:pos_1st_sos]
            grammar_str = output['grammar'][b]['aux']['grammar_str']
            grammar_dict = MLC_utils.grammar_to_dict(grammar_str)
            acc_b = []
            q_id = []
            k_id = []
            q_label = []
            k_label = []
            for i in arg_pred:
                token = pred_tokens_enc[i]
                if i>=len(yq):
                    acc_b.append(np.nan)
                    # inner.append(np.nan)
                    # inner_null.append(np.nan)
                    continue
                if token==yq[i]:
                    acc_b.append(1)
                    color_symbol = grammar_dict[1][token]
                    symbol_pos = [i for i, token in enumerate(query_tokens) if token==color_symbol][0]
                    q_id.append(i)
                    k_id.append(arg_max_kb[i])
                    q_label.append(symbol_pos)
                    k_label.append(symbol_pos)

                else:
                    acc_b.append(0)
            q_ids.append(q_id)
            k_ids.append(k_id)
            q_labels.append(q_label)
            k_labels.append(k_label)

        circuit = DirectedGraph(comment='MLC-Transformer', engine='neato') 
        q_args = {'vector_ids':q_ids, 'vector_labels':q_labels}
        k_args = {'vector_ids':k_ids, 'vector_labels':k_labels}
        problem_stack = node.get_problems(mode='k', target_args=k_args)

        problem_stack += node.get_problems(mode='q', target_args=q_args)


        sort_id = 0
        edges = []
        while problem_stack:
            current_problem = problem_stack.pop(0)
            current_node = current_problem.current_node
            current_problem_str = f'{current_node.module_name}.{current_node.head} {current_problem.mode}'
            source_nodes = current_problem.source_stream.source_nodes


            # check if can load backtracking score from saved data
            if (backtrack_data_dir/current_problem_str).exists() and not rewrite:
                source_scores, arg_q, target_labels, target_args = torch.load(backtrack_data_dir/current_problem_str)
                print(f'Loaded {current_problem_str}')
            else:
                source_scores, arg_q, target_labels, target_args = \
                current_problem.back_track(metric='customized', eval_func=eval_dec_cross_1_5_attn, ablate_all=True)

                torch.save([source_scores, arg_q, target_labels, target_args], backtrack_data_dir/current_problem_str)


            arg_top_nodes = get_top_nodes(source_nodes, source_scores, n_track=1, threshold=0.5)


            # ploting the score heatmap
            if True:
                self.plot_node_scores(target_problem = current_problem, 
                    source_nodes = source_nodes, 
                    source_scores = source_scores, 
                    plot_dir=backtrack_plot_dir, sort_id=sort_id,
                    cmap='cool', mid_0=False, include_emb=True, balance=False)
            
            # only do QKV problem
            next_nodes = [source_nodes[i] for i in arg_top_nodes]
            next_scores = [source_scores[i] for i in arg_top_nodes]
            for node, score in zip(next_nodes, next_scores):
                problems = node.get_problems(arg_q=arg_q, mode='v', prev_problem=current_problem, 
                                             target_labels=target_labels, target_args=target_args)

                # for problem in problems:
                edge_name = f'{node.module_name}{node.head}->{current_node.module_name}{current_node.head}: {current_problem.mode}'
                if edge_name in edges:
                    # print(edge_name)
                    continue

                problem_stack+=problems
                print(f'({sort_id}){current_problem.current_node.module_name} {current_problem.current_node.head}\
                        {current_problem.mode} track back {node.module_name} {node.head}')
                circuit.add_edge(sender=node, receiver=current_problem, score=str(np.round(score,1)))
                edges.append(edge_name)
                sort_id+=1


    def analyze_dec_cross_1_5(self, block, layer, type, head, rewrite=0):
        output = self.output
        n_batch = len(output['yq_predict'])
        node = self.graph[block, layer, type, head]

        attention = node.attention_score
        arg_pred = np.arange(3)
        arg_max_k = attention[np.arange(n_batch),0,:len(arg_pred),:].argmax(-1)
        cached_v = self.graph[block, layer, type, head].v_stream.cached_activation
        cached_q = self.graph[block, layer, type, head].q_stream.cached_activation
        cached_k = self.graph[block, layer, type, head].k_stream.cached_activation


        unemb_w = self.net.out.weight.detach().cpu().numpy() # [n_vocab, n_dim]
        unemb_node = self.graph['unembedding']
        prev_nodes, prev_activations = unemb_node.input_stream.forward()

        for i, node in enumerate(prev_nodes):
            if node.module_name==f'{block}.{type}.{layer}' and node.head==head:
                prev_activation = prev_activations[i]
                break

        if ('dec' in block) and ('cross' in type):
            type='multi'


        """
        Hypothesis: 
        the V symbols contain relative pos info to the nearest function
        """ 
        attention_percent = 0.9
        acc = []
        inner = []
        inner_null = []


        q = []
        k = []
        v = []
        z = []
        
        k_shuffle = []
        k_shuffle_label = []

        q_label=[]    
        k_label=[]
        v_label = []
        arg_label = []
        seq_label = []
        z_label = []

        exclude = []



        for b in range(n_batch):
            xq_context = np.array(output['xq_context'][b])
            yq = output['yq'][b]
            v_acc = output['v_acc'][b]
            if v_acc==0:
                exclude.append(b)
                continue
            

            
            grammar_str = output['grammar'][b]['aux']['grammar_str']
            grammar_dict = MLC_utils.grammar_to_dict(grammar_str)

            # exclude trials with more than 1 query function
            pos_1st_sos = np.where(xq_context=='SOS')[0][0]
            query_tokens = xq_context[:pos_1st_sos]
            # keep only trials with only one function in query
            query_functions = [token for i, token in enumerate(query_tokens) if token in grammar_dict[2].keys()]
            if len(query_functions)!=1:
                exclude.append(b)
                continue

            query_func = query_functions[0]

            # find 1st colors after function
            func_poses_all = [i for i, token in enumerate(xq_context) if token==query_func]
            func_poses_single = []
            sos_poses = np.where(xq_context=='SOS')[0]

            for pos_symbol in func_poses_all:
                sos_before = sos_poses[sos_poses<pos_symbol]
                if len(sos_before)==0:
                    sos_before = 0
                else:
                    sos_before = sos_before[-1]
                sos_after = sos_poses[sos_poses>pos_symbol][0]
                n_func = 0
                for s in xq_context[sos_before:sos_after]: # centered around the function
                    if s in grammar_dict[2].keys():
                        n_func+=1
                if n_func==1:
                    func_poses_single.append(pos_symbol)

            # exclude trials with no single function demonstrations
            if len(func_poses_single)<=1:
                exclude.append(b)
                continue
            
            """
            Token: Union[Symbol, Function]
            Symbol: token for color
            """

            # first predicted token
            pred_tokens = get_top_tokens(xq_context=xq_context, attention=attention[b,0,arg_pred,:], percent=attention_percent)
            acc_b = []
            for i, token in enumerate(pred_tokens):
                if i>=len(yq):
                    acc_b.append(np.nan)
                    # inner.append(np.nan)
                    # inner_null.append(np.nan)
                    continue
                if token==yq[i]:
                    acc_b.append(1)
                    color_id=self.langs['output'].symbol2index[yq[i]]
                    color_symbol = grammar_dict[1][yq[i]]
                    symbol_pos = [i for i, token in enumerate(query_tokens) if token==color_symbol][0]

                    inner.append(prev_activation[b,i,:]@unemb_w[color_id,:])
                    rand_id = np.random.choice(np.setdiff1d(np.arange(len(unemb_w)),color_id), 1)[0]
                    inner_null.append(prev_activation[b,i,:]@unemb_w[rand_id,:])

                    q.append(cached_q[b,0,i,:])
                    k.append(cached_k[b,0,arg_max_k[b,i],:])
                    v.append(cached_v[b,0,arg_max_k[b,i],:])
                    arg_label.append(i)
                    seq_label.append(symbol_pos)
                else:
                    acc_b.append(0)



            # all query symbols 
            acc_b += [np.nan]*(len(arg_pred)-len(acc_b))
            acc.append(acc_b)

            query_tokens = xq_context[:pos_1st_sos]
            query_color_symbols = [token for token in query_tokens if token in grammar_dict[0].keys()]


            # color symbols not in query
            all_color_symbols = np.unique([token for token in xq_context if token in grammar_dict[0].keys()])
            color_symbols_not_in_query = np.setdiff1d(all_color_symbols, query_color_symbols)
            for color in color_symbols_not_in_query:
                all_color_pos = np.array([i for i, t in enumerate(xq_context) if t==color])
                k_shuffle.append(cached_k[b,0,all_color_pos,:])
                # symbol = grammar_dict[1][color]
                # symbol_pos = [i for i, token in enumerate(query_tokens) if token==symbol][0]
                k_shuffle_label+=[-1]*all_color_pos.shape[0]


        acc = np.array(acc)
        acc = np.nanmean(acc, axis=0)
        inner = np.array(inner)
        inner_null = np.array(inner_null)

        q = np.vstack(q)
        k = np.vstack(k)
        v = np.vstack(v)
        k_shuffle = np.vstack(k_shuffle)

        # v = np.vstack(v)
        # z = np.vstack(z)
        v_label = np.array(v_label)
        arg_label = np.array(arg_label)
        seq_label = np.array(seq_label)
        # z_label = np.array(z_label)
        k_label = np.array(k_label)
        k_shuffle_label = np.array(k_shuffle_label)
        # print(k_label.mean())
        # Combine datasets for joint PCA


        if 1:
            fig, ax = plt.subplots(1,2,figsize=(8,6))
            axi=ax[0]
            axi.plot(arg_pred, acc, color='k')
            axi.scatter(arg_pred, acc)
            axi.set_xticks(arg_pred)
            axi.set_ylim([0,1.05])
            axi.set_ylabel('Accuracy')
            axi.set_xlabel('N^th token')

            axi=ax[1]
            axi.hist(inner, bins=50, alpha=0.5, weights=np.ones(len(inner)) / len(inner), label='V to unembedding', )
            axi.hist(inner_null, bins=50, alpha=0.5, weights=np.ones(len(inner_null)) / len(inner_null), label='V to shuffle')
            axi.set_xlabel('Inner product')
            axi.set_ylabel('Frequency')
            axi.legend()
            plt.show()

        # how much variance of q k is explained by arg_label
        vaf_scores = []
        for Y in [arg_label, seq_label]:
            for X in [q,k]:
                model = LinearRegression()
                model.fit(X, Y)
                Y_pred = model.predict(X)

                # Compute variance explained (R^2 score)
                variance_explained = r2_score(Y, Y_pred)
                vaf_scores.append(variance_explained)

  


        fig, ax = plt.subplots(1,3,figsize=(8,6))
        
        # plot_2D(ax=ax[0], source_data=[q,k_null], source_labels=['q','k_null'],color_labels=[arg_label,k_null_label],
        #     connect=0, markers=['s','.'], color_dict={-1:'grey'})
        # plot_2D(ax=ax[1], source_data=[k,k_null], source_labels=['k','k_null'],color_labels=[arg_label,k_null_label],
        #          connect=0, markers=['s','.'], color_dict={-1:'grey'})
        label=seq_label
        plot_2D(ax=ax[0], source_data=[q], source_legends=['q'],color_labels=[label],
            connect=0, markers=['s'], color_dict={1:'g'})
        plot_2D(ax=ax[1], source_data=[k], source_legends=['k'],color_labels=[label],
                 connect=0, markers=['x'], color_dict={1:'g'})
        plot_2D(ax=ax[2], source_data=[q,k], source_legends=['q','k'],color_labels=[label,label],
                 connect=1, markers=['s','x'], color_dict={1:'g'})

        # plot_pca(ax=ax[0], data=[q,k,k_null], source_labels=['q','k', 'k_null'],color_labels=[k_label,k_label,k_null_label])


        plt.show()

        a=1

    def analyze_dec_cross_0_3(self, block, layer, type, head):
        output = self.output
        n_batch = len(output['yq_predict'])

        attention = self.graph[block, layer, type, head].attention_score
        arg_max_k = attention[np.arange(n_batch),0,0,:].argmax(-1)
        cached_v = self.graph[block, layer, type, head].v_stream.cached_activation
        cached_q = self.graph[block, layer, type, head].q_stream.cached_activation
        cached_k = self.graph[block, layer, type, head].k_stream.cached_activation
        if ('dec' in block) and ('cross' in type):
            type='multi'
        cached_z = MLC_utils.get_activations_by_regex(net=self.net, 
                                                    cache=self.cache,
                                                    hook_regex=[{'module':f'*{block}*{layer}*{type}*z_hook*', 'head': f'{head}'}]
                                                    )[0]
        # cached_q = self.graph[block, layer, type, head].q_stream.cached_activation
        # cached_k = self.graph[block, layer, type, head].k_stream.cached_activation[np.arange(n_batch),0,arg_max_k,:]

        """
        Hypothesis: 
        the V symbols contain relative pos info to the nearest function
        """

        q = []
        k = []
        v = []
        z = []
        
        k_null = []
        k_null_label = []

        q_label=[]    
        k_label=[]
        v_label = []
        z_label = []

        exclude = []

        for b in range(n_batch):
            xq_context = np.array(output['xq_context'][b])
            yq = output['yq'][b]
            v_acc = output['v_acc'][b]
            if v_acc==0:
                exclude.append(b)
                continue
            

            
            grammar_str = output['grammar'][b]['aux']['grammar_str']
            grammar_dict = MLC_utils.grammar_to_dict(grammar_str)

            # exclude trials with more than 1 query function
            pos_1st_sos = np.where(xq_context=='SOS')[0][0]

            # keep only trials with only one function in query
            query_functions = [token for i, token in enumerate(xq_context[:pos_1st_sos]) if token in grammar_dict[2].keys()]
            if len(query_functions)!=1:
                exclude.append(b)
                continue

            query_func = query_functions[0]

            # find 1st colors after function
            func_poses_all = [i for i, token in enumerate(xq_context) if token==query_func]
            func_poses_single = []
            sos_poses = np.where(xq_context=='SOS')[0]

            for pos_symbol in func_poses_all:
                sos_before = sos_poses[sos_poses<pos_symbol]
                if len(sos_before)==0:
                    sos_before = 0
                else:
                    sos_before = sos_before[-1]
                sos_after = sos_poses[sos_poses>pos_symbol][0]
                n_func = 0
                for s in xq_context[sos_before:sos_after]: # centered around the function
                    if s in grammar_dict[2].keys():
                        n_func+=1
                if n_func==1:
                    func_poses_single.append(pos_symbol)

            # exclude trials with no single function demonstrations
            if len(func_poses_single)<=1:
                exclude.append(b)
                continue
            
            """
            Token: Union[Symbol, Function]
            Symbol: token for color
            """
            # all query symbols 

            z.append(cached_z[b,0,0,:])

            query_tokens = xq_context[:pos_1st_sos]
            # q.append(cached_q[b,0,0,:])
            # k.append(cached_k[b,0,arg_max_k[b],:])


            last_func_pos = np.array([i for i, token in enumerate(xq_context) if token==query_func])[-1]
            all_io_pos = np.array([i for i, token in enumerate(xq_context) if token=='IO'])
            io_pos = all_io_pos[all_io_pos>last_func_pos][0]

            sos_before, sos_after = sos_around_pos(xq_context, last_func_pos)
            correct_color = yq[0]
            symbol = grammar_dict[1][correct_color]
            symbol_pos = [i for i, token in enumerate(query_tokens) if token==symbol][0]
            z_label.append(symbol_pos)
            # k_label.append(symbol_pos)

            # color symbols not in query
            query_color_symbols = [token for token in query_tokens if token in grammar_dict[0].keys()]
            query_colors = [grammar_dict[0][token] for token in query_color_symbols]

            pos_non_query_color = np.array([i for i, token in enumerate(xq_context) if token not in query_colors])
            k_null.append(cached_k[b,0,pos_non_query_color,:])
            k_null_label+=[-1]*pos_non_query_color.shape[0]

            # all_color_symbols = np.unique([token for token in xq_context if token in grammar_dict[0].keys()])
            # color_symbols_not_in_query = np.setdiff1d(all_color_symbols, query_color_symbols)
            # color_not_in_query = np.unique([grammar_dict[0][token] for token in color_symbols_not_in_query])
            # for color in color_not_in_query:
            #     all_color_pos = np.array([i for i, t in enumerate(xq_context) if t==color])
            #     k_null.append(cached_k[b,0,all_color_pos,:])

            #     k_null_label+=[-1]*all_color_pos.shape[0]


        q = np.vstack(q) if len(q)>0 else q
        k = np.vstack(k) if len(k)>0 else k

        k_null = np.vstack(k_null) if len(k_null)>0 else k_null
        z = np.vstack(z) if len(z)>0 else z

        # v = np.vstack(v)
        # z = np.vstack(z)
        # v_label = np.array(v_label)
        z_label = np.array(z_label)
        k_label = np.array(k_label)
        k_null_label = np.array(k_null_label)
        # print(k_label.mean())
        # Combine datasets for joint PCA

        # only plot not excluded data
        # q = q[~np.isin(np.arange(n_batch), exclude)]
        # k = k[~np.isin(np.arange(n_batch), exclude)]
        # v = v[~np.isin(np.arange(n_batch), exclude)]
        fig, ax = plt.subplots(1,2,figsize=(8,6))
        # plot_pca(ax=ax[0], data=[q,k,k_null], source_labels=['q','k', 'k_null'],color_labels=[k_label,k_label,k_null_label])

        plot_2D(ax=ax[1], source_data=[z,k_null], source_legends=['z','k_null'],markers=['s','.'], color_labels=[z_label], color_dict_all={-1:'grey'})

        # plot_pca(data=[q,k], source_labels=['q','k'],color_labels=[k_label, k_label])
        plt.show()

        a=1




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

def get_top_nodes(source_nodes, score, n_track, threshold):
    # in the ranking, exclude embedding nodes
    source_names = [node.module_name for node in source_nodes]

    # source_scores = source_scores/np.linalg.norm(target_vectors,2)**2
    source_scores = score
    arg_embedding = [i for i, name in enumerate(source_names) if 'pos' in name or 'token' in name]
    arg_non_emb = np.setdiff1d(np.arange(len(source_names)), arg_embedding)

    # select all nodes with score more than threshold
    threshold_score = source_scores.min() + (source_scores.max()-source_scores.min())*threshold


    arg_max_nonemb = arg_non_emb[np.argsort(source_scores[arg_non_emb])]
    arg_max_nonemb = arg_max_nonemb[-n_track:]

    arg_larger_than_threshold = np.where(source_scores>threshold_score)[0]
    arg_max_emb = np.intersect1d(arg_larger_than_threshold, arg_embedding)
    arg_max_nodes = np.concatenate([arg_max_nonemb, arg_max_emb])
    return arg_max_nodes

def plot_2D(ax, source_data:list[np.array], source_legends:list[str], color_labels:list[np.array], markers=[], color_dict={}, connect=True, 
            projection='PCA'):
        
        assert projection in ['PCA', 'UMAP', 'TSNE']
        # print(f'Relative pos equal: {(q_label==k_label).mean()}')

        n_sources = len(source_data)

        if len(color_labels)==n_sources:
            cmap = plt.cm.Spectral
            unique_color_labels = np.unique(np.concatenate(color_labels))
            n_colors = unique_color_labels.shape[0] 
            colors = cmap(np.linspace(0, 1, n_colors))
            color_legend = True
            color_dict_all = {}
            for i, label in enumerate(unique_color_labels):
                color_dict_all[label] = colors[i]
        else:
            color_labels = []
            [color_labels.append(np.zeros(source_data[i].shape[0],dtype=int)) for i in range(n_sources)]
            colors = ['b']
            color_legend = False
            color_dict_all = {0:'b'}

        color_dict_all.update(color_dict)


        combined_data = np.vstack(source_data)  # Shape: (500, 16)
        source_id = []
        for i in range(n_sources):
            source_id+=[i]*source_data[i].shape[0]
        source_id = np.array(source_id)

        # Perform PCA
        if projection=='PCA':
            reducer = PCA(n_components=2)  # Reduce to 2D for visualization
        elif projection=='UMAP':
            reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1)
        elif projection=='TSNE':
            reducer = TSNE(n_components=2, perplexity=30)

        combined_proj = reducer.fit_transform(combined_data)


        if markers==[]: 
            # assign default markers
            markers = ['s','x','.','s', 'p', 'P', '*', '+', 'X', 'D', 'd', '|', '_', '<', '>', '^', 'v', 'h', 'H']
        # Visualization
        # fig, ax = plt.subplots(figsize=(8, 6))
        r2_values = []
        for i in range(n_sources):

            # Compute R^2 for each feature
            Y = source_data[i]
            Z = color_labels[i]


            # Average R^2
            r2 = average_r_squared(Y, Z)
            r2_values.append(r2)

            source_pca = combined_proj[source_id==i]
            ax.scatter(source_pca[:, 0], source_pca[:, 1], label=source_legends[i], alpha=0.3, marker=markers[i], color=[color_dict_all[l] for l in color_labels[i]])

        # id_positive = np.where(source_pca[:, 0]>0)[0]
        # id_negative = np.where(source_pca[:, 0]<0)[0]
        # torch.save((id_positive, id_negative), 'id_positive_negative.pt')
        # draw lines connecting paired points
        if connect and n_sources>=2:
            set1_pca = combined_proj[source_id==0]
            set2_pca = combined_proj[source_id==1]
            for i in range(set1_pca.shape[0]):
                ax.plot([set1_pca[i, 0], set2_pca[i, 0]], [set1_pca[i, 1], set2_pca[i, 1]], 'k-', alpha=0.3, linewidth=0.5, linestyle='--')
        
        if color_legend:
            color_handles = [
                mpatches.Patch(color=colors[label], label=f'{unique_color_labels[label]}')
                for label in range(n_colors)
            ]
            color_legend=ax.legend(handles=color_handles, loc='upper right', title='Legend', fontsize='small')
            ax.add_artist(color_legend)   

        marker_handles = [
            mlines.Line2D([], [], color='gray', marker=markers[i], linestyle='None', markersize=8, label=source_legends[i]) for i in range(n_sources)
        ]
        ax.legend(handles=marker_handles, loc='upper left', title='Marker Legend', fontsize='small')

        # Formatting
        ax.set_title(f'r^2={r2_values[0]:.2f}')
        ax.set_xlabel('PC 1')
        ax.set_ylabel('PC 2')

def average_r_squared(Y, Z):
    """
    Compute the average R^2 for Y explained by categorical Z using one-hot encoding.
    
    Args:
        Y (np.ndarray): High-dimensional data (n_samples, n_features).
        Z (np.ndarray): Categorical variable (n_samples,).
    
    Returns:
        float: Average R^2 across all features.
    """
    # Step 1: One-hot encode Z
    encoder = OneHotEncoder()
    Z_encoded = encoder.fit_transform(Z.reshape(-1, 1))
    
    # Step 2: Fit linear models for each feature in Y
    r_squared_values = []
    for i in range(Y.shape[1]):  # Loop through each feature in Y
        model = LinearRegression()
        model.fit(Z_encoded, Y[:, i])  # Fit model for the i-th feature
        r_squared = model.score(Z_encoded, Y[:, i])  # Compute R^2
        r_squared_values.append(r_squared)
    
    # Step 3: Average R^2 across all features
    return np.mean(r_squared_values)

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
    """
    return the token identity with the most summed attention
    """
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

    return pred_tokens

def eval_dec_cross_1_5_attn(analyzer, cache_patch, output_patch):
    output_org = analyzer.output
    n_batch = len(output_org['yq_predict'])
    arg_pred = np.arange(5)
    batch_ids = []
    attn_correct = []
    # attn_2nd = []

    """
    pickup the trials where the original model worked
    and evaluate attention diff between 1st and 2nd tokens on those trials
    """

    attn_dec_1_5 = MLC_utils.get_activations_by_regex(net=analyzer.net,
                                                     cache=cache_patch,
                                                     hook_regex=[{'module':'*dec*1*multi*attn_weight*', 'head':'5'}])[0]
    for b in range(n_batch):
        xq_context = np.array(output_org['xq_context'][b])
        grammar_str = output_org['grammar'][b]['aux']['grammar_str']
        grammar_dict = MLC_utils.grammar_to_dict(grammar_str)
        yq = np.array(output_org['yq'][b])
        yq_predict = np.array(output_org['yq_predict'][b])

        try:
            correct = np.all(yq[arg_pred]==yq_predict[arg_pred+1])
        except:
            correct = False

        if not correct:
            # the intact model performed 
            continue

        batch_ids.append(b)
        color_poses = []
        color_symbols = []
        symbol_poses = []
        for i,color in enumerate(yq[arg_pred]):

            color_poses.append([j for j, token in enumerate(xq_context) if token==color])
            color_symbols.append(grammar_dict[1][color])
            symbol_poses.append([j for j, token in enumerate(xq_context) if token==grammar_dict[1][color]])


            attn_correct.append(attn_dec_1_5[b,0,i,color_poses[-1]].sum(0))

    attn_correct = np.array(attn_correct).mean()

    return attn_correct





