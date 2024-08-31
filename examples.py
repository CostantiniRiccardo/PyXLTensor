import PyXLTensor as xlt

"""
the tensor t1 has two indices, alpha (up) and beta (down), the value of the tensor is known and all the metrics are the identity.
the tensor t2 has two indices, beta (up) and gamma (down), the value of the tensor is unknown and the metric associated with gamma is not the identity.
"""
t1 = xlt.Tensor('t1', '^alpha_beta', tensor=[[1, 2], [0, -1]])
t2 = xlt.Tensor('t2', '^beta_gamma', shape=(2, 2), metrics=[[[1, 0], [0, 1]], [[-1, 0], [0, 1]]])
print('Definition of the tensors t1 and t2')
print(t1)
print(t2)

"""
The operations +, -, * (scalar), / (scalar), @ (contraction) can be carried with the symbols.
The indices must match like in standard tensor operations, they must have the same name, size, positioning an metrics.
"""
t1 = xlt.Tensor('t1', '^alpha_beta', tensor=[[1, 2], [0, -1]])
t2 = xlt.Tensor('t2', '_beta^alpha', shape=(2, 2))
t3 = -t2 + t1
print('Sum and subtraction')
print(t3)

t1 = xlt.Tensor('t1', '^alpha_beta', tensor=[[1, 2], [0, -1]])
t2 = xlt.Tensor('t2', '^beta_gamma', shape=(2, 2), metrics=[[[1, 0], [0, 1]], [[-1, 0], [0, 1]]])
t3 = 2 * t1 @ t2
print('Multiplication and contraction')
print(t3)

"""
Sum of tensors
"""
t1 = xlt.Tensor('t1', '^alpha_beta', tensor=[[1, 2], [0, -1]])
t2 = xlt.Tensor('t2', '^alpha_beta', tensor=[[3, -1], [1, 1]])
t3 = xlt.Tensor('t3', '^alpha_beta', tensor=[[2, 1], [-4, 3]])
t4 = xlt.Tensor.sum_all([t1, t2, t3], 't4')
print('Sum of a list of tensors')
print(t4)

"""
Contraction of tensors
"""
t1 = xlt.Tensor('t1', '^alpha_beta', tensor=[[1, 2], [0, -1]])
t2 = xlt.Tensor('t2', '^beta_gamma^i', tensor=[[[3, -1], [1, 1]], [[0, -2], [3, 2]]])
t3 = xlt.Tensor('t3', '^gamma_alpha', tensor=[[2, 1], [-4, 3]])
t4 = xlt.Tensor.contract_all([t1, t2, t3], 't4')
print('Contraction of a list of tensors')
print(t4)

"""
Block tensor
In order to specify which indices will be combined and how we will give a list, grouping_of_indices.
The elements of this list will be another list, with two elements,
the first one is the name for the new index in the block tensor,
the other is the list of the indices name that will be grouped together.
"""
a = xlt.Tensor('a', '^i1^K_j1_L', shape=(2, 8, 1, 8))
b = xlt.Tensor('b', '^i1_L^K_j2', shape=(2, 8, 8, 3))
c = xlt.Tensor('c', '^K_L_j3^i1', shape=(8, 8, 4, 2))
d = xlt.Tensor('d', '^i2_j1^K_L', shape=(6, 1, 8, 8))
e = xlt.Tensor('e', '^K^i2_L_j2', shape=(8, 6, 8, 3))
f = xlt.Tensor('f', '^K^i2_L_j3', shape=(8, 6, 8, 4))
grouping_of_indices = [['I', ['i1', 'i2']],
                       ['J', ['j1', 'j2', 'j3']]]
T = xlt.Tensor.block_tensor([a, b, c, d, e, f], grouping_of_indices)
print('Definition of a block tensor')
print(T)

"""
(Anti-)Symmetrization
"""
t = xlt.Tensor('t', '^a^b', tensor=[[1, 2], [0, -1]])
t_sym = t.symmetrize('a', 'b')
t_asym = t.anti_symmetrize('a', 'b')
print('(Anti-)Symmetrization')
print(t_sym)
print(t_asym)

"""
Duality
"""
t = xlt.Tensor('t', '^c', tensor=[1, -1, 0])
star_t = t.dual('a', 'b', 'c')
print('Dual tensor')
print(star_t)

"""
Delta and epsilon tensors
"""
d8 = xlt.Tensor.delta('^i_j', 8)
epsilon = xlt.Tensor.epsilon('_a_b^c')
print('Delta and epsilon')
print(d8)
print(epsilon)

"""
Raising and lowering indices
"""
t = xlt.Tensor('t', '^i_j_k^l', shape=(2, 2, 2, 2), metrics=[[[-1, 0], [0, 1]], ] * 4)
t_all_up = t.to_raise('j', 'k')
t_all_down = t.to_lower('i', 'l')
print('Rising and lowering indices')
print(t_all_up)
print(t_all_down)

"""
Changing the indexing
"""
print('Indexing of the tensor')
t = xlt.Tensor('t', '^a^j^c^l', shape=(2, 2, 2, 2))
print(t.indices)  # ['a', 'j', 'c', 'l']
t1 = t.change_indices_name(['j', 'l'], ['b', 'd'])
print(t1.indices)  # ['a', 'b', 'c', 'd']
t2 = t['a', 'b', 'c', 'd']
print(t2.indices)  # ['a', 'b', 'c', 'd']
t2.set_order_indices('d', 'c', 'b', 'a')
print(t2.indices)  # ['d', 'c', 'b', 'a']

"""
Getting the elements of the tensor
"""
print('Getting elements of a tensor an slicing')
t = xlt.Tensor('t', '^a^b', tensor=[[1, 2], [0, -1]])
print(t[0, 1])  # 2 (Poly_Expression)
print(t['a', 0])  # t^{a0} = (1, 0)^{a0} (Tensor)
print(t[1, 'c'])  # t^{1c} = (0, -1)^{1c} (Tensor)
print(t.get_numeric_tensor())  # [[1, 2], [0, -1]] (numpy.array)

"""
Write the equations and initialize the System.
"""
v = xlt.Tensor('v', '_a', shape=(3, ))
t = xlt.Tensor('t', '_a', shape=(3, ))
M = xlt.Tensor('M', '^a^b', tensor=[[0, 1, 0], [1, 0, 0], [0, 0, -1]])
u = xlt.Tensor('u', '_b', tensor=[0, 1, 1])
one = xlt.Tensor('', '', tensor=1)

Cond1 = M @ v['b'] + t.to_raise('a')
Cond2 = v @ M @ u
Cond3 = v @ t.to_raise('a') - one
Cond4 = v.to_raise('a') @ v - one

unknown_tensors = [v, t]
list_equations = []
list_equations += Cond1.get_equations(unknown_tensors)
list_equations += Cond2.get_equations(unknown_tensors)
list_equations += Cond3.get_equations(unknown_tensors)
list_equations += Cond4.get_equations(unknown_tensors)

system = xlt.Solver(list_equations, 15)

xlt.zero_tolerance = 1e-8  # default value 1e-12

Complete_Solutions, Undetermined_Solutions = system.get_Solutions()
C_Tensor_Solutions = [xlt.Tensor.get_tensors_from_Sol(unknown_tensors, Sol) for Sol in Complete_Solutions]
U_Tensor_Solutions = [xlt.Tensor.get_tensors_from_Sol(unknown_tensors, Sol) for Sol in Undetermined_Solutions]

print('Solutions to tensor equations')
for i, Tensor_Sol in enumerate(C_Tensor_Solutions + U_Tensor_Solutions):
    print(f'Solution {i + 1}:\n')
    for tensor in Tensor_Sol:
        print(tensor)
