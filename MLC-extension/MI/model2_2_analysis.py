import MI.analysis as analysis
from MI import hook_functions
import numpy as np
from matplotlib.colors import TwoSlopeNorm
import matplotlib.pyplot as plt
import einops
from tqdm import tqdm
import re
import torch
import shutil
from MI.directed_graph import DirectedGraph
linear = torch._C._nn.linear
from sklearn.decomposition import PCA
import matplotlib.patches as mpatches
import matplotlib.lines as mlines




class ComputeNode:
    def __init__(self, net, cache, hooked_modules, output, langs):
        self.net = net
        self.cache = cache
        self.hooked_modules = hooked_modules
        self.output = output
        self.langs = langs

class UmembeddingNode(ComputeNode):
    def __init__(self, input_stream, unembedding_proj, module_name=None):
        self.module_name = module_name
        self.head = None
        self.input_stream = input_stream
        self.logit_problem = None
        self.unembedding_proj = unembedding_proj

    def get_problems(self, weight_mask=None, arg_q=None):
        n_batch = self.input_stream.cached_activation.shape[0]
        problems = []
        unembedding_proj = np.tile(self.unembedding_proj,(n_batch,1,1))

        # target_vectors = unembedding_proj[np.arange(n_batch), arg_source, :]

        self.logit_problem = Problem(source_stream=self.input_stream, target_vectors=unembedding_proj, 
                                     weight_mask=weight_mask, arg_q=arg_q, current_node=self, mode='out')
        problems.append(self.logit_problem)

        return problems

class EmbeddingNode(ComputeNode):
    def __init__(self, module_name=None, output_to_resid=None):
        self.module_name = module_name
        self.output_to_resid = output_to_resid
        self.head = None

    def get_problems(self, arg_q=None, mode=None):
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
    def __init__(self, source_stream:ActivationStream, target_vectors:np.ndarray=None, arg_q=None, weight_mask=None,  current_node=None, mode:str='q'):
        self.source_stream = source_stream
        self.arg_q = arg_q
        self.weight_mask = weight_mask
        self.target_vectors = target_vectors
        self.current_node = current_node
        self.mode = mode
    
    def back_track(self, n_track=2, threshold=0.5, prune_names=None):
        """
        track_n: max number of top nodes to back track
        threshold: the minimum score to consider a node as a top node

        Backtracking is different for encoder and decoder
        if decoder, only care about the row of the current q
        if encoder, consider all the qs

        """

        source_nodes, source_activations = self.source_stream.forward()

        # perform pruning here
        if prune_names is not None:
            id_prune = []
            for i, node in enumerate(source_nodes):
                name = node.module_name
                head = node.head
                if head is None:
                    continue
                elif f'{name}.{head}' in prune_names['circuit_name']:
                    id_prune.append(i)
            
            source_nodes = [source_nodes[i] for i in range(len(source_nodes)) if i not in id_prune]
            source_activations = [source_activations[i] for i in range(len(source_activations)) if i not in id_prune]

        # target is sum of source (without any biases along the way)
        # target_vectors = np.stack(source_activations).sum(0)
        
        target_vectors = self.target_vectors
        # target_vectors = self.target_vectors
        source_inner = []
        for source_activation in source_activations:
            n_batch = source_activation.shape[0]

            if 'dec' in self.current_node.module_name:
                # for decoder, only consider the row of the current q
                mask_zero = np.zeros_like(self.weight_mask)
                mask_zero[np.arange(n_batch),[self.arg_q], :] = 1
                weight_mask = self.weight_mask*mask_zero
            else:
                weight_mask = self.weight_mask

            if self.mode=='k':
                inner = einops.einsum(target_vectors, source_activation, 'b seq_q dim, b seq_k dim -> b seq_q seq_k')
            elif self.mode=='v':
                inner = einops.einsum(target_vectors, source_activation,'b seq_k dim, b seq_k dim -> b seq_k')
                inner = np.tile(inner[:,np.newaxis,:], (1, weight_mask.shape[1], 1))
            else: # 'q' or 'out'
                inner = einops.einsum(source_activation, target_vectors,'b seq_q dim, b seq_k dim -> b seq_q seq_k')
            inner_masked = inner*weight_mask

            score = inner_masked.mean()
            source_inner.append(score)


 
        # in the ranking, exclude embedding nodes
        source_names = [node.module_name for node in source_nodes]
        source_scores = np.array(source_inner)

        # source_scores = source_scores/np.linalg.norm(target_vectors,2)**2
        source_scores = source_scores/source_scores.sum()
        arg_embedding = [i for i, name in enumerate(source_names) if 'pos' in name or 'token' in name]

        # select all nodes with score more than threshold
        threshold_score = source_scores.min() + (source_scores.max()-source_scores.min())*threshold
        arg_larger_than_threshold = np.where(source_scores>threshold_score)[0]
        arg_top_nodes_nonemb = np.setdiff1d(arg_larger_than_threshold, arg_embedding)
        arg_tops_nodes_emb = np.intersect1d(arg_larger_than_threshold, arg_embedding)
        # rank arg_expand by score
        arg_top_nodes_nonemb = arg_top_nodes_nonemb[np.argsort(source_scores[arg_top_nodes_nonemb])]
        arg_top_nodes_nonemb = arg_top_nodes_nonemb[-n_track:]

        arg_top_nodes = np.concatenate([arg_top_nodes_nonemb, arg_tops_nodes_emb])

        # need to normalize score by the original inner product
        return source_nodes, source_scores, arg_top_nodes, self.arg_q
        


class AttentionNode(ComputeNode):
    def __init__(self, q_stream, k_stream, v_stream, attention_score, output_to_resid, module_name, head=None):
        self.module_name = module_name
        self.head = head

        self.q_stream = q_stream
        self.k_stream = k_stream
        self.v_stream = v_stream
        self.attention_score = attention_score
        self.output_to_resid = output_to_resid

        self.q_problem = None
        
        self.k_problem = None

        self.v_problem = None


    def get_problems(self, arg_q=None, mode:str='qkv'):
        
        assert set(mode).issubset({'q','k','v'})
        n_batch = self.q_stream.cached_activation.shape[0]



        target_q = self.q_stream.cached_activation[np.arange(n_batch), 0, :, :]
        # argmax_k = self.attention_score[np.arange(n_batch),0,arg_q,:].argmax(-1) # batch seq_q
        target_k = self.k_stream.cached_activation[np.arange(n_batch), 0, :, :]
        target_v = self.v_stream.cached_activation[np.arange(n_batch), 0, :, :] # batch seq d_model
        attention = self.attention_score[np.arange(n_batch),0,:,:] # batch seq_q seq_k



        
        problems = []
        if 'q' in mode:
            self.q_problem = Problem(source_stream=self.q_stream, target_vectors=target_k, weight_mask=attention, arg_q=arg_q, current_node=self, mode='q')
            problems.append(self.q_problem)
        if 'k' in mode:
            self.k_problem = Problem(source_stream=self.k_stream, target_vectors=target_q, weight_mask=attention, arg_q=arg_q, current_node=self, mode='k')
            problems.append(self.k_problem)
        if 'v' in mode:
            self.v_problem = Problem(source_stream=self.v_stream, target_vectors=target_v, weight_mask=attention, arg_q=arg_q, current_node=self, mode='v')
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



class Analysis:
    def __init__(self, dataloader, net, langs, plot_dir,null_dataset_path):
        self.dataloader = dataloader
        self.net = net
        self.plot_dir = plot_dir
        self.langs = langs
        self.null_dataset_path = null_dataset_path
        self.cache = None
        self.output = None


    def build_graph(self, bias=False, prune_names=None):
        all_hook_names = analysis.get_module_names_by_regex(self.net, [{'module':'*hook*', 'head':'*'}])
        val_batch = next(iter(self.dataloader))
        cache, hooked_modules = hook_functions.add_hooks(self.net, mode='cache', hook_names=all_hook_names)


        if prune_names is not None:
            null_activations = torch.load(self.null_dataset_path)
            net_prune_names = prune_names['net_name']
            _, ablate_modules = hook_functions.add_hooks(self.net, mode='patch', hook_names=net_prune_names, 
                        patch_activation=[null_activations[str(name)] for name in net_prune_names])
            hooked_modules+=ablate_modules
            
        output = analysis.eval_model(val_batch, self.net, self.langs)
        [hook.remove_hooks() for hook in hooked_modules]


        self.cache = cache
        self.output = output

        nlayers_encoder = self.net.nlayers_encoder
        nlayers_decoder = self.net.nlayers_decoder
        n_head = self.net.nhead
        d_model = self.net.hidden_size
        d_head = d_model // n_head
        graph = {}

        encoder_token_pos = analysis.get_activations_by_regex(self.net, cache, [{'module':'*encoder*layer*0*resid_pre_hook*','head':'*'}])[0]
        encoder_token = analysis.get_activations_by_regex(self.net, cache, [{'module':'*input_embedding_hook*','head':'*'}])[0]
        graph['encoder_token'] = EmbeddingNode(module_name='encoder_token', output_to_resid=encoder_token)
        graph['encoder_pos'] = EmbeddingNode(module_name='encoder_pos', output_to_resid=encoder_token_pos-encoder_token)
        
        decoder_token_pos = analysis.get_activations_by_regex(self.net, cache, [{'module':'*decoder*layer*0*resid_pre_hook*','head':'*'}])[0]
        decoder_token = analysis.get_activations_by_regex(self.net, cache, [{'module':'*output_embedding_hook*','head':'*'}])[0]
        graph['decoder_token'] = EmbeddingNode(module_name='decoder_token', output_to_resid=decoder_token)
        graph['decoder_pos'] = EmbeddingNode(module_name='decoder_pos', output_to_resid=decoder_token_pos-decoder_token)


        # on the encoder side
        for layer in range(nlayers_encoder):
            in_proj_self = self.net.transformer.encoder.layers[layer].self_attn.in_proj_weight
            in_bias_self = self.net.transformer.encoder.layers[layer].self_attn.in_proj_bias
            resid_pre = analysis.get_activations_by_regex(self.net, cache, [{'module':f'*encoder*layer*{layer}*resid_pre_hook*','head':'*'}])[0]

            cached_ln_self = CachedLayerNorm(resid_before_ln=resid_pre,
                                        ln=self.net.transformer.encoder.layers[layer].norm1, bias=bias)
            cache_z_self = analysis.get_activations_by_regex(self.net, cache, [{'module':f'*encoder*layer*{layer}*z_hook*','head':'*'}])
            cache_z_self = np.concatenate(cache_z_self, axis=1)

            out_proj_self = self.net.transformer.encoder.layers[layer].self_attn.out_proj.weight.detach().cpu().numpy() # d_model x d_model
            out_bias_self = self.net.transformer.encoder.layers[layer].self_attn.out_proj.bias.detach().cpu().numpy() # d_model
            
            oz_to_resid_self = OZToResid(out_proj=out_proj_self, out_bias=out_bias_self, n_head=n_head, d_head=d_head, bias=bias)
            attn_out_self = oz_to_resid_self(cache_z_self)

            for head in range(n_head):
                attention_score = analysis.get_activations_by_regex(self.net, cache, [{'module':f'*encoder*layer*{layer}*attn_weight*','head':head}])[0]
                cache_q = analysis.get_activations_by_regex(self.net, cache, [{'module':f'*encoder*layer*{layer}*q_hook*','head':head}])[0]
                cache_k = analysis.get_activations_by_regex(self.net, cache, [{'module':f'*encoder*layer*{layer}*k_hook*','head':head}])[0]
                cache_v = analysis.get_activations_by_regex(self.net, cache, [{'module':f'*encoder*layer*{layer}*v_hook*','head':head}])[0]
                
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
                                                                            output_to_resid=attn_out_self[:,head,:,:])

        # on the decoder side
        for layer in range(nlayers_decoder):
            in_proj_self = self.net.transformer.decoder.layers[layer].self_attn.in_proj_weight
            in_bias_self = self.net.transformer.decoder.layers[layer].self_attn.in_proj_bias

            resid_pre = analysis.get_activations_by_regex(self.net, cache, [{'module':f'*decoder*layer*{layer}*resid_pre_hook*','head':'*'}])[0]
            cached_ln_self = CachedLayerNorm(resid_before_ln=resid_pre,
                                        ln=self.net.transformer.decoder.layers[layer].norm1, bias=bias)
            
            resid_mid = analysis.get_activations_by_regex(self.net, cache, [{'module':f'*decoder*layer*{layer}*resid_mid_hook*','head':'*'}])[0]
            cached_ln_cross = CachedLayerNorm(resid_before_ln=resid_mid,
                                        ln=self.net.transformer.decoder.layers[layer].norm2, bias=bias)

            resid_encoder = analysis.get_activations_by_regex(self.net, cache, [{'module':f'*encoder*layer*{nlayers_encoder-1}*resid_post*','head':'*'}])[0]
            cache_ln_encoder = CachedLayerNorm(resid_before_ln=resid_encoder,
                                        ln=self.net.transformer.encoder.norm, bias=bias)
            
            cache_z_self = analysis.get_activations_by_regex(self.net, cache, [{'module':f'*decoder*layer*{layer}*self*z_hook*','head':'*'}])
            cache_z_self = np.concatenate(cache_z_self, axis=1)

            out_proj_self = self.net.transformer.decoder.layers[layer].self_attn.out_proj.weight.detach().cpu().numpy() # d_model x d_model
            out_bias_self = self.net.transformer.decoder.layers[layer].self_attn.out_proj.bias.detach().cpu().numpy() # d_model

            oz_to_resid_self = OZToResid(out_proj=out_proj_self, out_bias=out_bias_self, n_head=n_head, d_head=d_head, bias=bias)
            attn_out_self = oz_to_resid_self(cache_z_self)

            cache_z_cross = analysis.get_activations_by_regex(self.net, cache, [{'module':f'*decoder*layer*{layer}*multi*z_hook*','head':'*'}])
            cache_z_cross = np.concatenate(cache_z_cross, axis=1)

            in_proj_cross = self.net.transformer.decoder.layers[layer].multihead_attn.in_proj_weight # d_model x d_model
            in_bias_cross = self.net.transformer.decoder.layers[layer].multihead_attn.in_proj_bias # d_model
            out_proj_cross = self.net.transformer.decoder.layers[layer].multihead_attn.out_proj.weight.detach().cpu().numpy() # d_model x d_model
            out_bias_cross = self.net.transformer.decoder.layers[layer].multihead_attn.out_proj.bias.detach().cpu().numpy() # d_model
            
            oz_to_resid_cross = OZToResid(out_proj=out_proj_cross, out_bias=out_bias_cross, n_head=n_head, d_head=d_head)
            attn_out_cross = oz_to_resid_cross(cache_z_cross)

            # handle the self attention
            for head in range(n_head):
                attention_score = analysis.get_activations_by_regex(self.net, cache, [{'module':f'*decoder*layer*{layer}*self_attn*weight*','head':head}])[0]
                cache_q = analysis.get_activations_by_regex(self.net, cache, [{'module':f'*decoder*layer*{layer}*self*q_hook*','head':head}])[0]
                cache_k = analysis.get_activations_by_regex(self.net, cache, [{'module':f'*decoder*layer*{layer}*self*k_hook*','head':head}])[0]
                cache_v = analysis.get_activations_by_regex(self.net, cache, [{'module':f'*decoder*layer*{layer}*self*v_hook*','head':head}])[0]
                
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
                                                                            output_to_resid=attn_out_self[:,head,:,:])

            # handle the cross attention
            for head in range(n_head):
                attention_score = analysis.get_activations_by_regex(self.net, cache, [{'module':f'*decoder*layer*{layer}*multi*weight*','head':head}])[0]
                cache_q = analysis.get_activations_by_regex(self.net, cache, [{'module':f'*decoder*layer*{layer}*multi*q_hook*','head':head}])[0]
                cache_k = analysis.get_activations_by_regex(self.net, cache, [{'module':f'*decoder*layer*{layer}*multi*k_hook*','head':head}])[0]
                cache_v = analysis.get_activations_by_regex(self.net, cache, [{'module':f'*decoder*layer*{layer}*multi*v_hook*','head':head}])[0]
                
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
                                                                            output_to_resid=attn_out_cross[:,head,:,:])
        
        # unembedding at last
        source_nodes = [graph['decoder_token'], graph['decoder_pos']]
        for layer in range(nlayers_decoder):
            [source_nodes.append(graph['dec', layer, 'self', head]) for head in range(n_head)]
            [source_nodes.append(graph['dec', layer, 'cross', head]) for head in range(n_head)]

        cached_ln_decoder = CachedLayerNorm(resid_before_ln=analysis.get_activations_by_regex(self.net, cache, [{'module':f'*decoder*layer*{nlayers_decoder-1}*resid_post_hook*','head':'*'}])[0],
                            ln=self.net.transformer.decoder.norm, bias=bias)
        cache_output = analysis.get_activations_by_regex(self.net, cache, [{'module':'*decoder_hook*','head':'*'}])[0]
        stream_to_unembedding = ActivationStream(source_nodes=source_nodes, in_pipeline=[cached_ln_decoder], cached_activation=cache_output)
        unembedding_proj = self.net.out.weight.detach().cpu().numpy()
        graph['unembedding'] = UmembeddingNode(module_name='unembedding', input_stream=stream_to_unembedding, unembedding_proj=unembedding_proj)

        self.graph = graph


    def back_track(self, pred_arg, prune_names=None, plot_score=1, rewrite=False):
        back_track_dir = self.plot_dir / f'back_track_token_{pred_arg}'

        if back_track_dir.exists() and plot_score and rewrite:
            shutil.rmtree(back_track_dir)

        back_track_dir.mkdir(parents=True, exist_ok=True)

        problem_mode = 'qkv'
        n_track = 2
        threshold = 0.7
        
        n_batch = len(self.output['yq_predict'])
        seq_q = self.graph['unembedding'].input_stream.cached_activation.shape[1]
        n_vocal = self.net.out.weight.shape[0]

        pred_tokens = [self.output['yq_predict'][b][pred_arg+1] for b in range(n_batch)]
        first_token_ids = [self.langs['output'].symbol2index[t] for t in pred_tokens]

        circuit = DirectedGraph(comment='MLC-Transformer', engine='neato') 
        
        output_mask = np.zeros((n_batch, seq_q, n_vocal))
        output_mask[np.arange(n_batch), 0, first_token_ids] = 1 # mask for the first token
        problem_stack = [self.graph['unembedding'].get_problems(weight_mask=output_mask, arg_q=first_token_ids)[0]]

        sort_id = 0
        edges = []
        while problem_stack:
            current_problem = problem_stack.pop(0)
            current_node = current_problem.current_node

            source_nodes, source_scores, arg_top_nodes, arg_q = current_problem.back_track(n_track=n_track, threshold=threshold,
                                                                                           prune_names=prune_names)

            if plot_score:
                self.plot_node_scores(target_problem = current_problem, 
                    source_nodes = source_nodes, 
                    source_scores = source_scores, 
                    plot_dir=back_track_dir, sort_id=sort_id,
                    balance=False)
            

            next_nodes = [source_nodes[i] for i in arg_top_nodes]
            next_scores = [source_scores[i] for i in arg_top_nodes]
            for node, score in zip(next_nodes, next_scores):
                problems = node.get_problems(arg_q=arg_q, mode=problem_mode)

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
        if (not save_path.exists()) or rewrite:
            circuit.render(filename=filename, directory=self.plot_dir, view=True)

        return circuit


    def plot_node_scores(self, target_problem, source_nodes, source_scores, cmap='coolwarm', 
                         plot_dir=None,sort_id=None, balance=True):
        target_node = target_problem.current_node
        head = f'.{target_node.head}' if target_node.head is not None else ''
        target_name = f'{target_node.module_name}{head}'
        title = f'To {target_name} {target_problem.mode}'   
        save_path = plot_dir / f'({sort_id}){title}.png'
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
            if 'embed' in name:
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
        xticklabels = [f'Head {i}' for i in range(data.shape[1])]

        extreme = np.max(np.abs(data))
        norm = TwoSlopeNorm(vmin=-extreme, vcenter=0, vmax=extreme)
        if balance:
            handle=ax.imshow(data, cmap=cmap, norm=norm)
        else:
            handle=ax.imshow(data, cmap=cmap)
        fig.colorbar(handle, ax=ax)  # Associate the colorbar with the Axes
        ax.set_title(title)
        ax.set_xticks(np.arange(len(xticklabels)))
        ax.set_xticklabels(xticklabels, fontsize=5, rotation=90)
        ax.set_yticks(np.arange(len(yticklabels)))
        ax.set_yticklabels(yticklabels, fontsize=5, rotation=0)

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

            ablate_name = analysis.get_module_names_by_regex(self.net, [{'module':f'*{block}*{layer}*{attn}*z_hook*','head':f'{head}'}])
            ablate_names+=ablate_name
        cache, hooked_modules = hook_functions.add_hooks(self.net, mode='patch', hook_names=ablate_names, 
                                patch_activation=[null_activations[str(name)] for name in ablate_names])
        
        val_batch = next(iter(self.dataloader))
        output = analysis.eval_model(val_batch, self.net, self.langs)
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

        decoder_heads = analysis.get_module_names_by_regex(self.net, [{'module':f'*decoder*z_hook*','head':'*'}])
        # shuffle the heads
        # np.random.seed(seed)
        decoder_heads = decoder_heads[::-1]


        for name in decoder_heads:
            _, hooked_modules = hook_functions.add_hooks(self.net, mode='patch', hook_names=net_prune_names+[name], 
                                    patch_activation=[null_activations[str(name)] for name in net_prune_names+[name]])
            output = analysis.eval_model(val_batch, self.net, self.langs)
            [hook.remove_hooks() for hook in hooked_modules]

            if output['v_acc'].mean() == 1:
                net_prune_names+=[name]

        encoder_heads = analysis.get_module_names_by_regex(self.net, [{'module':f'*encoder*z_hook*','head':'*'}])
        # encoder_heads = encoder_heads[::-1]
        for name in encoder_heads:
            _, hooked_modules = hook_functions.add_hooks(self.net, mode='patch', hook_names=net_prune_names+[name], 
                                    patch_activation=[null_activations[str(name)] for name in net_prune_names+[name]])
            output = analysis.eval_model(val_batch, self.net, self.langs)
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


    def analyze_dec_cross_1_5(self):
        output = self.output
        n_batch = len(output['yq_predict'])
        layer = 1
        head = 5
        attention = self.graph['dec', layer, 'cross', head].attention_score
        arg_max_k = attention[np.arange(n_batch),0,0,:].argmax(-1)
        q = self.graph['dec', layer, 'cross', head].q_stream.cached_activation[np.arange(n_batch),0,0,:]
        k = self.graph['dec', layer, 'cross', head].k_stream.cached_activation[np.arange(n_batch),0,arg_max_k,:]

        """
        Hypothesis: 
        K encode relative distance between color's symbol's relative distance to function
        Q encode the first ouput's symbol's relative distance to function
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
            grammar_dict = analysis.grammar_to_dict(grammar_str)

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



            # the first color after function
            pos_color, first_color_after_func = [(i, token) for i, token in enumerate(xq_context[func_pos_single:func_pos_single+10]) if token in grammar_dict[1].keys()][0]
            pos_color+=func_pos_single

            q_symbol = grammar_dict[1][first_color_after_func]
            q_symbol_pos = np.array([i for i, token in enumerate(xq_context) if token==q_symbol])
            q_symbol_pos = q_symbol_pos[q_symbol_pos<pos_color][-1]

            q_label.append(q_symbol_pos-func_pos_single)



        q_label = np.array(q_label)
        k_label = np.array(k_label)
        # Combine datasets for joint PCA

        # only plot not excluded data
        q = q[~np.isin(np.arange(n_batch), exclude)]
        k = k[~np.isin(np.arange(n_batch), exclude)]

        print(f'Relative pos equal: {(q_label==k_label).mean()}')


        cmap = plt.cm.Spectral
        unique_pos = np.union1d(q_label,k_label)
        n_colors = unique_pos.shape[0] 
        colors = cmap(np.linspace(0, 1, n_colors))


        combined_data = np.vstack((q, k))  # Shape: (500, 16)

        # Perform PCA
        pca = PCA(n_components=2)  # Reduce to 2D for visualization
        combined_pca = pca.fit_transform(combined_data)

        # Separate the PCA-transformed datasets
        set1_pca = combined_pca[:q.shape[0]]
        set2_pca = combined_pca[k.shape[0]:]

        # Visualization
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(set1_pca[:, 0], set1_pca[:, 1], label='Q', alpha=0.7, marker='o', color=[colors[l] for l in q_label])
        ax.scatter(set2_pca[:, 0], set2_pca[:, 1], label='K', alpha=0.7, marker='x', color=[colors[l] for l in k_label])
        # draw lines connecting paired points
        for i in range(set1_pca.shape[0]):
            ax.plot([set1_pca[i, 0], set2_pca[i, 0]], [set1_pca[i, 1], set2_pca[i, 1]], 'k-', alpha=0.3, linewidth=0.5, linestyle='--')
        
        color_handles = [
            mpatches.Patch(color=colors[label], label=f'{unique_pos[label]}')
            for label in range(n_colors)
        ]
    
        color_legend=ax.legend(handles=color_handles, loc='upper right', title='Legend', fontsize='small')
        ax.add_artist(color_legend)         
        marker_handles = [
            mlines.Line2D([], [], color='gray', marker='o', linestyle='None', markersize=8, label='Q'),
            mlines.Line2D([], [], color='gray', marker='x', linestyle='None', markersize=8, label='K')
        ]
        ax.legend(handles=marker_handles, loc='upper left', title='Marker Legend', fontsize='small')

        # Formatting
        ax.set_title('Relative pos to function')
        ax.set_xlabel('PC 1')
        ax.set_ylabel('PC 2')
        ax.grid(True)
        plt.show()
        a=1

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
            grammar_dict = analysis.grammar_to_dict(grammar_str)

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

    def analyze_dec_cross_0_4(self, block, layer, type, head):
        output = self.output
        n_batch = len(output['yq_predict'])

        attention = self.graph[block, layer, type, head].attention_score
        arg_max_k = attention[np.arange(n_batch),0,1,:].argmax(-1)
        v = self.graph[block, layer, type, head].v_stream.cached_activation[np.arange(n_batch),0,0,:]
        q = self.graph[block, layer, type, head].q_stream.cached_activation[np.arange(n_batch),0,0,:]
        k = self.graph[block, layer, type, head].k_stream.cached_activation[np.arange(n_batch),0,arg_max_k,:]

        """
        Hypothesis: 
        the V of dec.self.1.3 should contain relative pos information 1st output's color's symbol to function
        """

        # check if the attended token is the first one after IO
        k_argmax_symbol = []
        # yq_first = []
        for b in range(n_batch):
            xq_context = np.array(output['xq_context'][b])
            # yq = output['yq'][b]
            k_argmax_symbol.append(xq_context[arg_max_k[b]])
            # yq_first.append(yq[0])
        k_argmax_symbol = np.array(k_argmax_symbol)
        # yq_first = np.array(yq_first)

        # acc = (k_argmax_symbol==yq_first).mean()
        # print(f'First attended token accuracy: {acc}')

        k_label=[]
        q_label=[]
        exclude = []

        for b in range(n_batch):
            xq_context = np.array(output['xq_context'][b])
            yq = output['yq'][b]
            v_acc = output['v_acc'][b]
            if v_acc==0:
                exclude.append(b)
                continue
            

            
            grammar_str = output['grammar'][b]['aux']['grammar_str']
            grammar_dict = analysis.grammar_to_dict(grammar_str)

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

            IO_poses = np.array([i for i, token in enumerate(xq_context) if token=='IO'])
            IO_pos_before = IO_poses[IO_poses<[arg_max_k[b]]][-1]
            k_label.append(arg_max_k[b]-IO_pos_before)








        k_label = np.array(k_label)
        print(k_label.mean())
        # Combine datasets for joint PCA

        # only plot not excluded data
        q = q[~np.isin(np.arange(n_batch), exclude)]
        k = k[~np.isin(np.arange(n_batch), exclude)]
        v = v[~np.isin(np.arange(n_batch), exclude)]

        plot_pca(data=[v], source_labels=['v'],color_labels=[k_label])

        plot_pca(data=[q,k], source_labels=['q','k'],color_labels=[k_label, k_label])

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
            grammar_dict = analysis.grammar_to_dict(grammar_str)

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

        plot_pca(data=[v], source_labels=['v'],color_labels=[v_label])

        plot_pca(data=[q,k], source_labels=['q','k'],color_labels=[k_label, k_label])

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
            grammar_dict = analysis.grammar_to_dict(grammar_str)

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

        plot_pca(data=[v], source_labels=['v'],color_labels=[v_label])

        plot_pca(data=[q,k], source_labels=['q','k'],color_labels=[k_label, k_label])

        a=1







def plot_pca(data:list[np.array], source_labels:list[str], color_labels:list[np.array]):
        # print(f'Relative pos equal: {(q_label==k_label).mean()}')

        n_sources = len(data)

        if len(color_labels)==n_sources:
            cmap = plt.cm.Spectral
            unique_color_labels = np.unique(np.concatenate(color_labels))
            n_colors = unique_color_labels.shape[0] 
            colors = cmap(np.linspace(0, 1, n_colors))
            color_legend = True
        else:
            color_labels = []
            [color_labels.append(np.zeros(data[i].shape[0],dtype=int)) for i in range(n_sources)]
            colors = ['b']
            color_legend = False



        combined_data = np.vstack(data)  # Shape: (500, 16)
        source_id = []
        for i in range(n_sources):
            source_id+=[i]*data[i].shape[0]
        source_id = np.array(source_id)

        # Perform PCA
        pca = PCA(n_components=2)  # Reduce to 2D for visualization
        combined_pca = pca.fit_transform(combined_data)


        markers = ['o','x']
        # Visualization
        fig, ax = plt.subplots(figsize=(8, 6))
        for i in range(n_sources):
            source_pca = combined_pca[source_id==i]
            ax.scatter(source_pca[:, 0], source_pca[:, 1], label=source_labels[i], alpha=0.7, marker=markers[i], color=[colors[l] for l in color_labels[i]])

        # draw lines connecting paired points
        if n_sources==2:
            set1_pca = combined_pca[source_id==0]
            set2_pca = combined_pca[source_id==1]
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
            mlines.Line2D([], [], color='gray', marker=markers[i], linestyle='None', markersize=8, label=source_labels[i]) for i in range(n_sources)
        ]
        ax.legend(handles=marker_handles, loc='upper left', title='Marker Legend', fontsize='small')

        # Formatting
        ax.set_title('Relative pos to function')
        ax.set_xlabel('PC 1')
        ax.set_ylabel('PC 2')
        ax.grid(True)
        plt.show()

def sos_around_pos(xq, pos):
    sos_poses = np.where(xq=='SOS')[0]
    sos_before = sos_poses[sos_poses<pos]
    if len(sos_before)==0:
        sos_before = 0
    else:
        sos_before = sos_before[-1]
    sos_after = sos_poses[sos_poses>pos][0]
    return sos_before, sos_after