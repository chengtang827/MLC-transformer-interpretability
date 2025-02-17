from graphviz import Digraph

class DirectedGraph:
    def __init__(self, comment, engine='dot'):
        dot = Digraph(comment=comment, engine=engine)
        n_layers = 2
        n_heads = 8
        head_offset = 1.5
        layer_offset = 3
        decoder_offset = 3
        jitter_y = 0.2
        jitter_x = 0.5
        all_nodes = []

        # encoder embedding
        dot.node(f'encoder_token', f'encoder_token', pos=f'{(n_heads-4.2)*head_offset/2},{-layer_offset}!', 
                    shape='box', style='rounded', fillcolor='lightblue')
        dot.node(f'encoder_pos', f'encoder_pos', pos=f'{n_heads*head_offset/2},{-layer_offset}!', 
                    shape='box', style='rounded', fillcolor='lightblue')
        
        # decoder embedding
        dot.node(f'decoder_token', f'decoder_token', pos=f'{(n_heads-1)*1.5*head_offset+decoder_offset},{-layer_offset}!', 
                    shape='box', style='rounded', fillcolor='lightblue')
        dot.node(f'decoder_pos', f'decoder_pos', pos=f'{n_heads*1.5*head_offset+decoder_offset},{-layer_offset}!', 
                    shape='box', style='rounded', fillcolor='lightblue')
        
        # unembedding
        dot.node(f'unembedding', f'unembedding', pos=f'{n_heads*1.5*head_offset+decoder_offset},{2*layer_offset+layer_offset*3/2}!', 
                    shape='box', style='rounded', fillcolor='lightblue')
        
        all_nodes+=['encoder_token', 'encoder_pos', 'decoder_token', 'decoder_pos', 'unembedding']

        # Encoder side
        for l in range(n_layers):
            for h in range(n_heads):
                dot.node(f'enc.self.{l}.{h}', f'Self {l}.{h}', pos=f'{h*head_offset},{l*layer_offset}!', 
                        shape='box', style='rounded', fillcolor='lightblue')
                all_nodes+= [f'enc.self.{l}.{h}']

        # Decoder side
        for l in range(n_layers):
            for h in range(n_heads):
                dot.node(f'dec.cross.{l}.{h}', f'Cross {l}.{h}', 
                        pos=f'{(h+n_heads)*head_offset+decoder_offset},{l*layer_offset+layer_offset*(3/2+jitter_y)}!', 
                        shape='box', style='rounded', fillcolor='lightgreen')
                dot.node(f'dec.self.{l}.{h}', f'Self {l}.{h}', 
                            pos=f'{(h+n_heads-jitter_x)*head_offset+decoder_offset},{l*layer_offset+layer_offset*(3/2-1/2+jitter_y)}!', 
                            shape='box', style='rounded', fillcolor='lightgreen')
                all_nodes+= [f'dec.cross.{l}.{h}', f'dec.self.{l}.{h}']
        # Add edges
        # dot.edge('enc.self.1.0', 'dec.cross.1.1', label='A->B', penwidth='2')
        dot.attr(nodesep='1.0', ranksep='4.5')
        # dot.attr(overlap='false', splines='true')

        self.dot = dot
        self.edges = []
        self.all_nodes = all_nodes
        self.used_nodes = []


    def add_node(self, node_name, node_label, pos, shape='circle', style='filled', fillcolor='lightblue'):
        self.dot.node(node_name, node_label, pos=pos, shape=shape, style=style, fillcolor=fillcolor)
        
    def add_edge(self, sender, receiver, score):
        """
        sender: node
        receiver: problem
        """
        sender_module_name = sender.module_name
        sender_head = sender.head
        if sender_head is not None:
            sender_name = f'{sender_module_name}.{sender_head}'
        else:
            sender_name = sender_module_name

        receiver_module_name = receiver.current_node.module_name
        receiver_head = receiver.current_node.head
        if receiver_head is not None:
            receiver_name = f'{receiver_module_name}.{receiver_head}'
        else:
            receiver_name = receiver_module_name

        problem_mode = receiver.mode
        edge_name = f'{sender_name}->{receiver_name}: {problem_mode}'

        color_dict = {
            'q': '#051094',
            'k': '#52a447',
            'v': '#FF2400',
            'out': '#FF2400',}

        if edge_name not in self.edges:
            self.edges.append(edge_name)
            self.dot.edge(sender_name, receiver_name, color=color_dict[problem_mode], headlabel=score,penwidth='2'),
        
        if sender_name not in self.used_nodes:
            self.used_nodes.append(sender_name)
        if receiver_name not in self.used_nodes:
            self.used_nodes.append(receiver_name)
        
        return
        
    def render(self, filename, format='png', cleanup=True, directory='MI', view=True):
        self.dot.render(filename, format=format, cleanup=cleanup, directory=directory, view=view)
        
    def print_source(self):
        print(self.dot.source)
# Create a directed graph using neato layout
# dot = Digraph(comment='Directed Graph with Specified Node Positions', engine='neato')

# # Add nodes with specified positions (x, y coordinates)
# # dot.node('A', 'Node A', pos='0,0!', shape='circle', style='filled', fillcolor='lightblue')
# # dot.node('B', 'Node B', pos='2,2!', shape='circle', style='filled', fillcolor='lightgreen')
# # dot.node('C', 'Node C', pos='4,0!', shape='circle', style='filled', fillcolor='lightpink')
# # dot.node('D', 'Node D', pos='6,-2!', shape='circle', style='filled', fillcolor='lightyellow')


# # dot.edge('B', 'C', label='B->C')
# # dot.edge('C', 'D', label='C->D')
# # dot.edge('A', 'D', label='A->D')

# # Render and view the graph
# dot.render('directed_graph_with_positions', format='png', cleanup=True, directory='MI', view=True)
# print(dot.source)  # Print the Graphviz source code
