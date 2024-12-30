import numpy as np

# *SUPPORT*
# IN: a OUT: 1
# IN: b d a OUT: 2 2 1
# IN: b OUT: 2
# IN: c OUT: 3

# *QUERY*
# IN: a d c OUT: 1 1 3

input = "adc a:1 bda:221 b:2 c:3"

# split query and support
query, support = input.split(' ')[0], input.split(' ')[1:]
print(f'Query: {query}')
print(f'Support: {support}')

# detect primitives
primitives = {}
for i, s in enumerate(support):
    if len(s) == 3:
        primitives[s[0]] = s[2]
print(f'Primitives: {primitives}')

# detect function
functions = {}
for i, s in enumerate(support):
    if len(s) > 3:
        func, out = s.split(':')
        functions[func] = out
print(f'Functions: {functions}')

# to abstract a function, we need to convert the RHS into the id of args
func_mapping = {}
for func, out in functions.items():
    func_symbol = func[1]
    args = [primitives[func[0]], primitives[func[-1]]]
    
    pos=[args.index(o) for o in out]
    func_mapping[func_symbol] = pos
print(f'Function mapping: {func_mapping}')

# Generate the output symbols according to the rule
query_func = query[1]
query_args = [query[0], query[-1]]
output_symbols = [query_args[i] for i in func_mapping[query[1]]]
print(f'Query output symbol: {output_symbols}')

# translate the output symbols into tokens
output_tokens = [primitives[s] for s in output_symbols]
print(f'Qeury output token: {output_tokens}')
a=1
