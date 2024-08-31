import numpy as np
import math
from copy import deepcopy
from functools import cmp_to_key


zero_tolerance = 1e-12


class Poly_Expression:
    def __init__(self, constant=1, list_tensor_names=[], list_indices=[]):
        if constant != 0:
            self.sum_variables = [[constant, sorted(list(zip(list_tensor_names, list_indices)))]]
        else:
            self.sum_variables = []

    def __str__(self):
        # return str(self.sum_variables)
        if len(self.sum_variables) == 0:
            return '0'
        str_PE = Poly_Expression.__str_mono(self.sum_variables[0])
        if str_PE[0] == '+':
            str_PE = str_PE[1:]
        for monomial in self.sum_variables[1:]:
            str_PE += Poly_Expression.__str_mono(monomial)
        return str_PE

    def __repr__(self):
        return str(self)

    def __add__(self, other):
        new_pe = deepcopy(self)
        list_self_variables = [variables for _, variables in self.sum_variables]
        for other_constant, other_variables in other.sum_variables:
            if other_variables in list_self_variables:
                new_pe.sum_variables[list_self_variables.index(other_variables)][0] += other_constant
            else:
                new_pe.sum_variables.append([other_constant, other_variables])
        for monomial in list(new_pe.sum_variables):
            if -zero_tolerance < monomial[0] < zero_tolerance:
                new_pe.sum_variables.remove(monomial)
        return new_pe

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        if not isinstance(other, Poly_Expression):
            new_pe = deepcopy(self)
            if -zero_tolerance < other < zero_tolerance:
                for index in range(len(new_pe.sum_variables)):
                    new_pe.sum_variables[index][0] *= 0
            else:
                for index in range(len(new_pe.sum_variables)):
                    new_pe.sum_variables[index][0] *= other
            return new_pe
        new_pe = Poly_Expression(0)
        new_variables_used = []
        for constant1, variables1 in self.sum_variables:
            for constant2, variables2 in other.sum_variables:
                new_variables = sorted(variables1 + variables2)
                if new_variables in new_variables_used:
                    new_pe.sum_variables[new_variables_used.index(new_variables)][0] += constant1 * constant2
                else:
                    new_pe.sum_variables.append([constant1 * constant2, new_variables])
                    new_variables_used.append(new_variables)
        return new_pe

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, scalar):
        return 1 / scalar * self

    def __pos__(self):
        return deepcopy(self)

    def __neg__(self):
        return -1 * self

    def get_numeric_value(self):
        if len(self.sum_variables) == 0:
            return 0
        if len(self.sum_variables) == 1 and len(self.sum_variables[0][1]) == 0:
            return self.sum_variables[0][0]

        raise Exception('The expression has undefined parameters')

    @staticmethod
    def from_num_to_PE(tensor):
        if isinstance(tensor, np.ndarray):
            tensor = tensor.tolist()
        if isinstance(tensor, list) or isinstance(tensor, tuple):
            return [Poly_Expression.from_num_to_PE(sub_tensor) for sub_tensor in tensor]
        return Poly_Expression(tensor)

    @staticmethod
    def __str_mono(monomial):
        num_fig = 4
        if monomial[0] < 0:
            str_mon = '-'
        else:
            str_mon = '+'
        if not -zero_tolerance < math.fabs(monomial[0]) - 1 < zero_tolerance or len(monomial[1]) == 0:
            str1 = str(math.fabs(monomial[0]))
            str2 = f'{math.fabs(monomial[0]):.{num_fig}f}'
            if len(str1) < len(str2):
                str_mon += str1
            else:
                str_mon += str2
        for variable_name, indices in monomial[1]:
            str_mon += variable_name + str(tuple(indices))
        return str_mon


class Tensor:
    def __init__(self, tensor_name, indices_string, tensor=None, shape=None, metrics=None):
        self.tensor_name = tensor_name

        self.indices = []
        self.indices_size = []
        self.metrics = []
        self.inverse_metrics = []

        self.upper_indices = []
        self.lower_indices = []
        self.sym_indices = []
        self.asym_indices = []

        self.tensor = None

        if tensor is None and shape is None:
            raise Exception(f'Undefined tensor ({tensor_name}), you must give the tensor components or the shape')

        if tensor is not None:
            self.tensor = np.array(Poly_Expression.from_num_to_PE(tensor))
            shape = self.tensor.shape
        else:
            self.tensor = np.full(shape, None)
            for indices_val in Tensor.__get_iterable(shape):
                self.tensor[indices_val] = Poly_Expression(1, [tensor_name], [list(indices_val)])

        indices_string = indices_string.replace(' ', '')
        rank = sum(char_ == '^' or char_ == '_' for char_ in indices_string)

        if metrics is None:
            metrics = [np.identity(dim) for dim in shape]
        else:
            metrics = [np.array(metric) for metric in metrics]

        if rank != len(shape):
            raise Exception(f'(tensor {self.tensor_name}): Mismatch for shape {shape} and indexing {indices_string}')

        if rank != len(metrics):
            raise Exception(f'(tensor {self.tensor_name}): Mismatch for shape {shape} and number of metrics given')
        else:
            for i, metric in enumerate(metrics):
                if np.linalg.det(metric) == 0:
                    raise Exception(f'(tensor {self.tensor_name}): Non invertible metric found {metric}')
                if len(metric) != shape[i]:
                    raise Exception(f'(tensor {self.tensor_name}): Incorrect metric shape, expected {shape[i]} for the metric {metric}')

        up_index = None
        current_name = ''
        current_position = 0
        for char in indices_string + '_':
            if char != '^' and char != '_':
                current_name += char
            else:
                if up_index is not None:
                    if current_name == '':
                        raise Exception(f'Invalid index naming for: {indices_string}')
                    elif up_index is not None:
                        if up_index is None:
                            up_index = char == '^'
                        else:
                            self.indices.append(current_name)
                            self.indices_size.append(shape[current_position])
                            self.metrics.append(metrics[current_position])
                            self.inverse_metrics.append(np.linalg.inv(metrics[current_position]))
                            if up_index:
                                self.upper_indices.append(current_name)
                            else:
                                self.lower_indices.append(current_name)
                        current_name = ''
                        current_position += 1
                up_index = char == '^'

        if len(self.upper_indices) + len(self.lower_indices) != len(set(self.upper_indices+self.lower_indices)):
            raise Exception(f'(tensor {self.tensor_name}): Same name used for different indices')

    def __getitem__(self, *indices):
        if isinstance(indices[0], tuple) or isinstance(indices[0], list):
            indices = indices[0]
        if len(self.indices) != len(indices):
            raise Exception(f'The number of indices given do not match the number of indices of the tensor')
        for index in indices:
            if isinstance(index, int) and index < 0:
                raise Exception(f'The indices must have non negative entries')

        if all(isinstance(index, int) for index in indices):
            return self.tensor[tuple(indices)]

        if all(isinstance(index, str) for index in indices):
            original_names = []
            new_names = []
            for i, index in enumerate(self.indices):
                if index != indices[i]:
                    original_names.append(index)
                    new_names.append(indices[i])
            return self.change_indices_name(original_names, new_names)

        new_indices = []
        indices_not_to_get = []
        indices_to_get = []
        indices_val = []
        for i, new_index in enumerate(indices):
            if isinstance(new_index, int):
                indices_val.append(new_index)
                indices_to_get.append(self.indices[i])
                new_indices.append(self.indices[i])
            else:
                indices_not_to_get.append(new_index)
                new_indices.append(new_index)

        new_Tensor = self[new_indices]
        new_Tensor.set_order_indices(*(indices_to_get + indices_not_to_get))

        new_Tensor.indices = new_Tensor.indices[len(indices_val):]
        new_Tensor.indices_size = new_Tensor.indices_size[len(indices_val):]
        new_Tensor.metrics = new_Tensor.metrics[len(indices_val):]
        new_Tensor.inverse_metrics = new_Tensor.inverse_metrics[len(indices_val):]

        for index in indices_to_get:
            if index in new_Tensor.upper_indices:
                new_Tensor.upper_indices.remove(index)
            else:
                new_Tensor.lower_indices.remove(index)
            for i, s_indices in enumerate(new_Tensor.sym_indices):
                if index in s_indices:
                    new_Tensor.sym_indices[i].remove(index)
            for i, a_indices in enumerate(new_Tensor.asym_indices):
                if index in a_indices:
                    new_Tensor.asym_indices[i].remove(index)

        for s_indices in new_Tensor.sym_indices:
            if len(s_indices) < 2:
                new_Tensor.sym_indices.remove(s_indices)

        for a_indices in new_Tensor.asym_indices:
            if len(a_indices) < 2:
                new_Tensor.asym_indices.remove(a_indices)

        new_Tensor.tensor = new_Tensor.tensor[tuple(indices_val)]

        return new_Tensor

    def __str__(self):
        indices_string = ''
        for index in self.indices:
            if index in self.upper_indices:
                indices_string += '^'
            else:
                indices_string += '_'
            indices_string += index
        str_Tensor = '-=-=-=-=-=-=-=-=-=-'
        str_Tensor += '\ntensor: ' + self.tensor_name
        str_Tensor += '\nindices: ' + indices_string
        str_Tensor += '\nshape: ' + str(self.tensor.shape)
        str_Tensor += '\n' + str(self.tensor)
        str_Tensor += '\n-=-=-=-=-=-=-=-=-=-'
        return str_Tensor

    def __add__(self, other_Tensor):
        if not isinstance(other_Tensor, Tensor):
            raise Exception(f'{other_Tensor} is not a Tensor')
        if len(self.upper_indices) != len(other_Tensor.upper_indices):
            raise Exception(f'{self.tensor_name} and {other_Tensor.tensor_name} have a different number of upper indices')
        if len(self.lower_indices) != len(other_Tensor.lower_indices):
            raise Exception(f'{self.tensor_name} and {other_Tensor.tensor_name} have a different number of lower indices')
        for index in self.upper_indices:
            if index not in other_Tensor.upper_indices:
                raise Exception(f'{self.tensor_name} and {other_Tensor.tensor_name} have different upper indices')
        for index in self.lower_indices:
            if index not in other_Tensor.lower_indices:
                raise Exception(f'{self.tensor_name} and {other_Tensor.tensor_name} have different lower indices')
        for i, index in enumerate(self.indices):
            j = other_Tensor.indices.index(index)
            if self.indices_size[i] != other_Tensor.indices_size[j]:
                raise Exception(f'{self.tensor_name} and {other_Tensor.tensor_name} have different sizes for the index {index}')
            if not np.allclose(self.metrics[i], other_Tensor.metrics[j]):
                raise Exception(f'{self.tensor_name} and {other_Tensor.tensor_name} have different metrics for the index {index}')

        new_Tensor = deepcopy(self)

        new_Tensor.sym_indices = []
        new_Tensor.asym_indices = []
        for s_indices_1 in self.sym_indices:
            for s_indices_2 in other_Tensor.sym_indices:
                s_indices = list(set(s_indices_1).intersection(set(s_indices_2)))
                if len(s_indices) > 1:
                    new_Tensor.sym_indices.append(s_indices)
        for a_indices_1 in self.asym_indices:
            for a_indices_2 in other_Tensor.asym_indices:
                a_indices = list(set(a_indices_1).intersection(set(a_indices_2)))
                if len(a_indices) > 1:
                    new_Tensor.asym_indices.append(a_indices)

        new_Tensor.tensor_name += other_Tensor.tensor_name

        other_Tensor_ = deepcopy(other_Tensor)
        other_Tensor_.set_order_indices(*new_Tensor.indices)
        new_Tensor.tensor += other_Tensor_.tensor

        return new_Tensor

    def __sub__(self, other_Tensor):
        return self + (-other_Tensor)

    def __mul__(self, scalar):
        new_Tensor = deepcopy(self)
        new_Tensor.tensor *= scalar
        return new_Tensor

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, scalar):
        return 1 / scalar * self

    def __pos__(self):
        return deepcopy(self)

    def __neg__(self):
        return -1 * self

    def __matmul__(self, other_Tensor):
        if not isinstance(other_Tensor, Tensor):
            raise Exception(f'{other_Tensor} is not a Tensor')
        if len(set(self.upper_indices).intersection(other_Tensor.upper_indices)) != 0:
            raise Exception(f'{self.tensor_name} and {other_Tensor.tensor_name} have common upper indices')
        if len(set(self.lower_indices).intersection(other_Tensor.lower_indices)) != 0:
            raise Exception(f'{self.tensor_name} and {other_Tensor.tensor_name} have common lower indices')
        for index in self.upper_indices + self.lower_indices:
            if index in other_Tensor.upper_indices + other_Tensor.lower_indices:
                i = self.indices.index(index)
                j = other_Tensor.indices.index(index)
                if self.indices_size[i] != other_Tensor.indices_size[j]:
                    raise Exception(f'The tensors {self.tensor_name} and {other_Tensor.tensor_name} have different sizes for the index {index}')
                if not np.allclose(self.metrics[i], other_Tensor.metrics[j]):
                    raise Exception(f'The tensors {self.tensor_name} and {other_Tensor.tensor_name} have different metrics for the index {index}')

        new_Tensor = deepcopy(self)
        new_other_Tensor = deepcopy(other_Tensor)

        new_Tensor.tensor_name += other_Tensor.tensor_name

        new_Tensor.upper_indices += other_Tensor.upper_indices
        new_Tensor.lower_indices += other_Tensor.lower_indices
        contracted_indices = set(new_Tensor.upper_indices).intersection(set(new_Tensor.lower_indices))
        ref_indices_1 = new_Tensor.indices.copy()
        ref_indices_2 = new_other_Tensor.indices.copy()
        for contracted_index in contracted_indices:
            i = new_Tensor.indices.index(contracted_index)
            j = new_other_Tensor.indices.index(contracted_index)

            new_Tensor.indices.pop(i)
            new_Tensor.indices_size.pop(i)
            new_Tensor.metrics.pop(i)
            new_Tensor.inverse_metrics.pop(i)

            new_other_Tensor.indices.pop(j)
            new_other_Tensor.indices_size.pop(j)
            new_other_Tensor.metrics.pop(j)
            new_other_Tensor.inverse_metrics.pop(j)

            new_Tensor.upper_indices.remove(contracted_index)
            new_Tensor.lower_indices.remove(contracted_index)

        for num_contractions, contracted_index in enumerate(contracted_indices):
            i = ref_indices_1.index(contracted_index)
            j = ref_indices_2.index(contracted_index)

            new_Tensor.tensor = np.rollaxis(new_Tensor.tensor, i, len(ref_indices_1))
            ref_indices_1.append(ref_indices_1.pop(i))
            new_other_Tensor.tensor = np.rollaxis(new_other_Tensor.tensor, j, num_contractions)
            ref_indices_2.insert(num_contractions, ref_indices_2.pop(j))

        new_Tensor.tensor = np.tensordot(new_Tensor.tensor, new_other_Tensor.tensor, axes=len(contracted_indices))

        new_Tensor.indices += new_other_Tensor.indices
        new_Tensor.indices_size += new_other_Tensor.indices_size
        new_Tensor.metrics += new_other_Tensor.metrics
        new_Tensor.inverse_metrics += new_other_Tensor.inverse_metrics

        new_Tensor.sym_indices = []
        new_Tensor.asym_indices = []
        for s_indices in self.sym_indices:
            s_indices = list(set(s_indices) - contracted_indices)
            if len(s_indices) > 1:
                new_Tensor.sym_indices.append(s_indices)
        for s_indices in other_Tensor.sym_indices:
            s_indices = list(set(s_indices) - contracted_indices)
            if len(s_indices) > 1:
                new_Tensor.sym_indices.append(s_indices)
        for a_indices in self.asym_indices:
            a_indices = list(set(a_indices) - contracted_indices)
            if len(a_indices) > 1:
                new_Tensor.asym_indices.append(a_indices)
        for a_indices in other_Tensor.asym_indices:
            a_indices = list(set(a_indices) - contracted_indices)
            if len(a_indices) > 1:
                new_Tensor.asym_indices.append(a_indices)

        return new_Tensor

    @staticmethod
    def contract_all(list_tensors, new_tensor_name=None):
        total_upper_indices = []
        total_lower_indices = []
        for tensor in list_tensors:
            total_upper_indices += tensor.upper_indices
            total_lower_indices += tensor.lower_indices
        if len(total_upper_indices) != len(set(total_upper_indices)):
            raise Exception('The list of tensors has common upper indices')
        if len(total_lower_indices) != len(set(total_lower_indices)):
            raise Exception('The list of tensors has common lower indices')

        new_Tensor = deepcopy(list_tensors[0])
        for other_Tensor in list_tensors[1:]:
            new_Tensor = new_Tensor @ other_Tensor

        if new_tensor_name is not None:
            new_Tensor.tensor_name = new_tensor_name

        return new_Tensor

    @staticmethod
    def sum_all(list_tensors, new_tensor_name=None):
        new_Tensor = deepcopy(list_tensors[0])
        for other_Tensor in list_tensors[1:]:
            new_Tensor = new_Tensor + other_Tensor

        if new_tensor_name is not None:
            new_Tensor.tensor_name = new_tensor_name

        return new_Tensor

    @staticmethod
    def block_tensor(list_tensors, grouping_of_indices, new_tensor_name=None):
        list_tensors = deepcopy(list_tensors)

        if len(list_tensors) == 1:
            return list_tensors[0]

        new_Tensor = Tensor(list_tensors[0].tensor_name, '', shape=())

        num_indices = len(list_tensors[0].indices)
        indices_non_block = deepcopy(list_tensors[0].indices)
        sizes_non_block = deepcopy(list_tensors[0].indices_size)
        uppers_non_block = [index in list_tensors[0].upper_indices for index in indices_non_block]
        metrics_non_block = deepcopy(list_tensors[0].metrics)
        inv_metrics_non_block = deepcopy(list_tensors[0].inverse_metrics)
        for index in list(indices_non_block):
            for _, associated_indices in grouping_of_indices:
                if index in associated_indices:
                    i = indices_non_block.index(index)
                    indices_non_block.pop(i)
                    sizes_non_block.pop(i)
                    uppers_non_block.pop(i)
                    metrics_non_block.pop(i)
                    inv_metrics_non_block.pop(i)

        for tensor in list_tensors[1:]:
            new_Tensor.tensor_name += tensor.tensor_name
            if len(tensor.indices) != num_indices:
                raise Exception('There are tensors with different number of indices')
            for i_non_block, index in enumerate(indices_non_block):
                if index not in tensor.indices:
                    raise Exception('There are tensors with different set of non block indices')
                else:
                    i = tensor.indices.index(index)
                    if tensor.indices_size[i] != sizes_non_block[i_non_block]:
                        raise Exception(f'The index {index} has not an unique size')
                    if (index in tensor.upper_indices) != uppers_non_block[i_non_block]:
                        raise Exception(f'The index {index} is used as both up and down')
                    if not np.allclose(tensor.metrics[i], metrics_non_block[i_non_block]):
                        raise Exception(f'The index {index} has not an unique metric')

        shape_of_blocks = [len(associated_indices) for _, associated_indices in grouping_of_indices]
        list_blocks = np.full(shape_of_blocks, None)
        indices_block = [index_block for index_block, _ in grouping_of_indices]
        sizes_block = [[None for _ in range(len(associated_indices))] for _, associated_indices in grouping_of_indices]
        uppers_block = [None for _ in grouping_of_indices]
        metrics_block = [[None for _ in range(len(associated_indices))] for _, associated_indices in grouping_of_indices]
        inv_metrics_block = [[None for _ in range(len(associated_indices))] for _, associated_indices in grouping_of_indices]

        for tensor in list_tensors:
            position = []
            for map_indices in grouping_of_indices:
                for i, index in enumerate(map_indices[1]):
                    if index in tensor.indices:
                        position.append(i)
                        break
            if list_blocks[tuple(position)] is None:
                list_blocks[tuple(position)] = tensor
            else:
                raise Exception(f'The indices {[grouping_of_indices[i_block][1][i] for i_block, i in enumerate(position)]} are used for more than one tensor')
            for i_block, i in enumerate(position):
                if sizes_block[i_block][i] is None:
                    sizes_block[i_block][i] = tensor.indices_size[tensor.indices.index(grouping_of_indices[i_block][1][i])]
                elif sizes_block[i_block][i] != tensor.indices_size[tensor.indices.index(grouping_of_indices[i_block][1][i])]:
                    raise Exception(f'The index {grouping_of_indices[i_block][1][i]} has different sizes across the matrices')
                if uppers_block[i_block] is None:
                    uppers_block[i_block] = grouping_of_indices[i_block][1][i] in tensor.upper_indices
                elif uppers_block[i_block] != (grouping_of_indices[i_block][1][i] in tensor.upper_indices):
                    raise Exception(f'The index {grouping_of_indices[i_block][1][i]} is used as both up and down')
                if metrics_block[i_block][i] is None:
                    metrics_block[i_block][i] = tensor.metrics[tensor.indices.index(grouping_of_indices[i_block][1][i])]
                    inv_metrics_block[i_block][i] = tensor.inverse_metrics[tensor.indices.index(grouping_of_indices[i_block][1][i])]
                elif not np.allclose(metrics_block[i_block][i], tensor.metrics[tensor.indices.index(grouping_of_indices[i_block][1][i])]):
                    raise Exception(f'The index {grouping_of_indices[i_block][1][i]} has different metrics across the matrices')

        list_blocks_flat = list_blocks.flatten()
        for i in range(len(list_blocks_flat)):
            if list_blocks_flat[i] is None:
                raise Exception('Not enough tensors')

        for i_block, sizes in enumerate(sizes_block):
            grouping_of_indices[i_block].append(sizes)

        sizes_block = [sum(sizes) for sizes in sizes_block]
        metrics_block = [Tensor.__block_diagonal_matrix(metrics) for metrics in metrics_block]
        inv_metrics_block = [Tensor.__block_diagonal_matrix(metrics) for metrics in inv_metrics_block]

        if new_tensor_name is not None:
            new_Tensor.tensor_name = new_tensor_name

        new_Tensor.indices = indices_non_block + indices_block
        new_Tensor.indices_size = sizes_non_block + sizes_block
        new_Tensor.metrics = metrics_non_block + metrics_block
        new_Tensor.inverse_metrics = inv_metrics_non_block + inv_metrics_block

        for i, index in enumerate(indices_block):
            if uppers_block[i]:
                new_Tensor.upper_indices.append(index)
            else:
                new_Tensor.lower_indices.append(index)
        for i, index in enumerate(indices_non_block):
            if uppers_non_block[i]:
                new_Tensor.upper_indices.append(index)
            else:
                new_Tensor.lower_indices.append(index)

        new_Tensor.sym_indices = []
        new_Tensor.asym_indices = []

        new_Tensor.tensor = np.block(Tensor.__list_tensor_to_list_np(list_blocks.tolist(), new_Tensor.indices, grouping_of_indices))

        return new_Tensor

    def symmetrize(self, *sym_indices):
        for a_indices in list(self.asym_indices):
            n_intersection = len(set(a_indices).intersection(set(sym_indices)))
            if n_intersection > 1:
                return 0 * self

            if n_intersection > 0:
                self.asym_indices.remove(a_indices)

        for s_indices in list(self.sym_indices):
            if len(set(s_indices).intersection(set(sym_indices))) > 0:
                self.sym_indices.remove(s_indices)

        self.sym_indices.append(list(sym_indices))

        return self.__generalized_symmetrize(False, list(sym_indices))

    def anti_symmetrize(self, *asym_indices):
        for s_indices in list(self.sym_indices):
            n_intersection = len(set(s_indices).intersection(set(asym_indices)))
            if n_intersection > 1:
                return 0 * self

            if n_intersection > 0:
                self.sym_indices.remove(list(s_indices))

        for a_indices in list(self.asym_indices):
            if len(set(a_indices).intersection(set(asym_indices))) > 0:
                self.asym_indices.remove(a_indices)

        self.asym_indices.append(list(asym_indices))

        return self.__generalized_symmetrize(True, list(asym_indices))

    def dual(self, *epsilon_indices):
        if len(epsilon_indices) == 0:
            return deepcopy(self)
        dual_indices = []
        dual_contracted_indices = []
        dual_dim = len(epsilon_indices)
        dual_indices_metric = None
        dual_indices_inverse_metric = None
        upper_dual_indices = None
        untouched_indices = deepcopy(self.indices)
        for index in epsilon_indices:
            if index not in self.indices:
                dual_indices.append(index)
            else:
                untouched_indices.remove(index)
                dual_contracted_indices.append(index)
                i = self.indices.index(index)
                if self.indices_size[i] != dual_dim:
                    return Exception('Incorrect dimension for the dual operation / indices sizes')
                if dual_indices_metric is None:
                    dual_indices_metric = self.metrics[i]
                    dual_indices_inverse_metric = self.inverse_metrics[i]
                    upper_dual_indices = index in self.lower_indices
                else:
                    if not np.allclose(self.metrics[i], dual_indices_metric):
                        raise Exception(f'The contracted dual indices have different metrics')
                    if (index in self.lower_indices) != upper_dual_indices:
                        raise Exception(f'The contracted dual indices are both up and down')

        if dual_indices_metric is None:
            raise Exception(f'There are no contracted dual indices')

        ref_Tensor = deepcopy(self).anti_symmetrize(*dual_contracted_indices)
        ref_Tensor.set_order_indices(*(dual_contracted_indices + untouched_indices))

        try:
            new_Tensor = ref_Tensor[*([0, ] * len(dual_contracted_indices) + untouched_indices)]
            new_Tensor.tensor_name = '*' + new_Tensor.tensor_name
        except:
            new_Tensor = Tensor('*' + self.tensor_name, '', shape=())

        new_Tensor.indices = dual_indices + new_Tensor.indices
        new_Tensor.indices_size = [dual_dim, ] * len(dual_indices) + new_Tensor.indices_size
        new_Tensor.metrics = [dual_indices_metric, ] * len(dual_indices) + new_Tensor.metrics
        new_Tensor.inverse_metrics = [dual_indices_inverse_metric, ] * len(dual_indices) + new_Tensor.inverse_metrics

        dual_tensor = np.full(new_Tensor.indices_size, Poly_Expression(0))

        if upper_dual_indices:
            new_Tensor.upper_indices = dual_indices + new_Tensor.upper_indices
        else:
            new_Tensor.lower_indices = dual_indices + new_Tensor.lower_indices
        if len(dual_indices) > 1:
            new_Tensor.asym_indices.append(dual_indices)

        i_cut = dual_dim - len(dual_contracted_indices)
        for indices in Tensor.__get_iterable((dual_dim, ) * dual_dim, [list(range(i_cut)), list(range(i_cut, dual_dim))]):
            if len(set(indices)) == dual_dim:
                dual_indices_val = indices[:i_cut]
                dual_contracted_indices_val = indices[i_cut:]

                sub_tensor = ref_Tensor[*(list(dual_contracted_indices_val) + untouched_indices)]
                if isinstance(sub_tensor, Tensor):
                    sub_tensor = sub_tensor.tensor
                _sub_tensor = -sub_tensor

                base_parity = sum(sum(index_d > index_dc for index_d in dual_indices_val) for index_dc in dual_contracted_indices_val)
                base_parity = 1 - 2 * (base_parity % 2)

                permutations, parities = Tensor.__permutations(list(dual_indices_val))
                for permutation, sigma in zip(permutations, parities):
                    if base_parity * sigma == 1:
                        dual_tensor[tuple(permutation)] = sub_tensor
                    else:
                        dual_tensor[tuple(permutation)] = _sub_tensor

        new_Tensor.tensor = dual_tensor

        return new_Tensor

    def change_indices_name(self, original_names, new_names):
        new_Tensor = deepcopy(self)

        # Changing the indices on new_Tensor
        for i_sub, index in enumerate(original_names):
            if index in self.indices:
                new_Tensor.indices[self.indices.index(index)] = new_names[i_sub]

                if index in self.upper_indices:
                    new_Tensor.upper_indices[self.upper_indices.index(index)] = new_names[i_sub]
                else:
                    new_Tensor.lower_indices[self.lower_indices.index(index)] = new_names[i_sub]

        for i_sym, s_indices in enumerate(self.sym_indices):
            for i, index in enumerate(s_indices):
                if index in original_names:
                    new_Tensor.sym_indices[i_sym][i] = new_names[original_names.index(index)]
        for i_asym, a_indices in enumerate(self.asym_indices):
            for i, index in enumerate(a_indices):
                if index in original_names:
                    new_Tensor.asym_indices[i_asym][i] = new_names[original_names.index(index)]

        # Verifying validity of change of indices
        if len(new_Tensor.upper_indices) != len(set(new_Tensor.upper_indices)):
            raise Exception(f'(tensor {self.tensor_name}): Found upper indices with same name')
        if len(new_Tensor.lower_indices) != len(set(new_Tensor.lower_indices)):
            raise Exception(f'(tensor {self.tensor_name}): Found lower indices with same name')

        contracted_indices = set(new_Tensor.upper_indices).intersection(set(new_Tensor.lower_indices))
        for contracted_index in contracted_indices:
            i = new_Tensor.indices.index(contracted_index)
            j = new_Tensor.indices.index(contracted_index, i + 1)

            if new_Tensor.indices_size[i] != new_Tensor.indices_size[j]:
                raise Exception(f'The contracted index {contracted_index} in the tensors {self.tensor_name} has different sizes')
            if not np.allclose(new_Tensor.metrics[i], new_Tensor.metrics[j]):
                raise Exception(f'The contracted index {contracted_index} in the tensors {self.tensor_name} has different metrics')

            new_Tensor.indices.pop(j)
            new_Tensor.indices_size.pop(j)
            new_Tensor.metrics.pop(j)
            new_Tensor.inverse_metrics.pop(j)
            new_Tensor.indices.pop(i)
            new_Tensor.indices_size.pop(i)
            new_Tensor.metrics.pop(i)
            new_Tensor.inverse_metrics.pop(i)

            new_Tensor.upper_indices.remove(contracted_index)
            new_Tensor.lower_indices.remove(contracted_index)
            if contracted_index in new_Tensor.sym_indices:
                new_Tensor.sym_indices.remove(contracted_index)
            if contracted_index in new_Tensor.asym_indices:
                new_Tensor.asym_indices.remove(contracted_index)

            new_Tensor.tensor = np.trace(new_Tensor.tensor, axis1=i, axis2=j)

        if isinstance(new_Tensor.tensor, Poly_Expression):
            new_Tensor.tensor = np.array(new_Tensor.tensor)

        return new_Tensor

    def set_order_indices(self, *new_indices):
        if len(new_indices) != len(self.indices) or set(new_indices) != set(self.indices):
            raise Exception(f'The indices given are different from the indices {self.indices}')
        for i, index in enumerate(new_indices[:-1]):
            j = self.indices.index(index)
            if i != j:
                self.indices[i], self.indices[j] = self.indices[j], self.indices[i]
                self.indices_size[i], self.indices_size[j] = self.indices_size[j], self.indices_size[i]
                self.metrics[i], self.metrics[j] = self.metrics[j], self.metrics[i]
                self.inverse_metrics[i], self.inverse_metrics[j] = self.inverse_metrics[j], self.inverse_metrics[i]
                self.tensor = np.swapaxes(self.tensor, i, j)

    def to_raise(self, *indices_to_raise):
        if not set(self.indices) >= set(indices_to_raise):
            raise Exception('The given indices are not all contained in the indices of the tensor')
        indices_to_raise_ = []
        intermediate_indices = []
        metrics_to_contract = []
        for index_to_raise in indices_to_raise:
            if index_to_raise in self.upper_indices:
                print(f'Index {index_to_raise} is already an upper index')
            else:
                metric = self.metrics[self.indices.index(index_to_raise)]
                inv_metric = self.inverse_metrics[self.indices.index(index_to_raise)]
                indices_to_raise_.append(index_to_raise)
                while index_to_raise in self.indices + intermediate_indices:
                    index_to_raise += '\''
                intermediate_indices.append(index_to_raise)
                g = Tensor('', '^'+indices_to_raise_[-1]+'^'+index_to_raise, tensor=inv_metric, metrics=[metric, ] * 2)
                metrics_to_contract.append(g)

        return Tensor.contract_all([self.change_indices_name(indices_to_raise_, intermediate_indices)] + metrics_to_contract)

    def to_lower(self, *indices_to_lower):
        if not set(self.indices) >= set(indices_to_lower):
            raise Exception('The given indices are not all contained in the indices of the tensor')
        indices_to_lower_ = []
        intermediate_indices = []
        metrics_to_contract = []
        for index_to_lower in indices_to_lower:
            if index_to_lower in self.lower_indices:
                print(f'Index {index_to_lower} is already an upper index')
            else:
                metric = self.metrics[self.indices.index(index_to_lower)]
                indices_to_lower_.append(index_to_lower)
                while index_to_lower in self.indices + intermediate_indices:
                    index_to_lower += '\''
                intermediate_indices.append(index_to_lower)
                g = Tensor('', '_'+indices_to_lower_[-1]+'_'+ index_to_lower, tensor=metric, metrics=[metric, ] * 2)
                metrics_to_contract.append(g)

        return Tensor.contract_all([self.change_indices_name(indices_to_lower_, intermediate_indices)] + metrics_to_contract)

    def get_numeric_tensor(self):
        new_tensor = np.zeros(self.tensor.shape)
        for indices_val in Tensor.__get_iterable(self.tensor.shape):
            try:
                new_tensor[indices_val] = self.tensor[indices_val].get_numeric_value()
            except:
                raise Exception('The tensor has undefined parameters')
        return new_tensor

    def get_equations(self, unknown_tensors):
        unknown_tensor_names = [tensor.tensor_name for tensor in unknown_tensors]
        if len(unknown_tensor_names) != len(set(unknown_tensor_names)):
            raise Exception('Some unknown tensors share the same name')
        offsets = {}
        vector_for_conversion = {}
        num_variables = 0
        for unknown_tensor in unknown_tensors:
            offsets.update({unknown_tensor.tensor_name: num_variables})
            vector = []
            tensor_size = 1
            for dim in reversed(unknown_tensor.indices_size):
                vector.append(tensor_size)
                tensor_size *= dim
            vector_for_conversion.update({unknown_tensor.tensor_name: np.array(list(reversed(vector)))})
            num_variables += tensor_size

        list_equations = []
        asym_indices_position = [[self.indices.index(index) for index in a_indices] for a_indices in self.asym_indices]
        sym_indices_position = [[self.indices.index(index) for index in s_indices] for s_indices in self.sym_indices]
        for indices_value in self.__get_iterable(self.indices_size, asym_indices_position, sym_indices_position):
            expression = self[indices_value]
            eq = {}
            for constant, variables in expression.sum_variables:
                var = [0 for _ in range(num_variables)]
                for variable in variables:
                    if len(variable[1]) == 0:
                        base_pos = 0
                    else:
                        base_pos = vector_for_conversion[variable[0]] @ np.array(variable[1])
                    var_index = offsets[variable[0]] + base_pos
                    var[var_index] += 1
                eq.update({tuple(var): constant})
            list_equations.append(eq)

        return list_equations

    @staticmethod
    def get_tensors_from_Sol(unknown_tensors, Sol):
        unknown_tensors = deepcopy(unknown_tensors)
        Sol = deepcopy(Sol)
        for index_tensor, unknown_tensor in enumerate(unknown_tensors):
            unknown_tensor.tensor = unknown_tensor.tensor.flatten()
            for index_variable, variable in enumerate(Sol[:len(unknown_tensor.tensor)]):
                if variable is not None:
                    unknown_tensor.tensor[index_variable] = Poly_Expression(variable)
            Sol = Sol[len(unknown_tensor.tensor):]
            unknown_tensor.tensor = np.reshape(unknown_tensor.tensor, unknown_tensor.indices_size)
            unknown_tensors[index_tensor] = unknown_tensor
        return unknown_tensors

    @staticmethod
    def epsilon(indices_string):
        dim = len(indices_string.replace('_', '^').split('^')) - 1
        base = np.zeros((dim,) * dim)
        permutations, parities = Tensor.__permutations(list(range(dim)))
        for permutation, sigma in zip(permutations, parities):
            base[tuple(permutation)] = sigma
        epsilon_Tensor = Tensor('', indices_string, tensor=base)
        if len(epsilon_Tensor.upper_indices) > 1:
            epsilon_Tensor.asym_indices.append(deepcopy(epsilon_Tensor.upper_indices))
        if len(epsilon_Tensor.lower_indices) > 1:
            epsilon_Tensor.asym_indices.append(deepcopy(epsilon_Tensor.lower_indices))
        return epsilon_Tensor

    @staticmethod
    def delta(indices_string, dim):
        if len(indices_string.replace('_', '^').split('^')) != 3:
            raise Exception(f'Incorrect number of indices')
        delta_Tensor = Tensor('', indices_string, tensor=np.eye(dim))
        if len(delta_Tensor.upper_indices) == 0 or len(delta_Tensor.lower_indices) == 0:
            delta_Tensor.sym_indices = [deepcopy(delta_Tensor.indices)]
        return delta_Tensor

    def __generalized_symmetrize(self, a_sym, sym_indices):
        if len(sym_indices) == 0:
            return deepcopy(self)
        if len(sym_indices) != len(set(sym_indices)):
            raise Exception('The given indices have repetitions')
        unused_indices = list(set(sym_indices) - set(self.indices))
        if len(unused_indices) != 0:
            raise Exception(f'The indices {unused_indices} are not present in {self.tensor_name}')
        ups = [sym_index in self.upper_indices for sym_index in sym_indices]
        if not all(ups) and any(ups):
            raise Exception(f'The given indices are both up and down in {self.tensor_name}')
        used_size = self.indices_size[self.indices.index(sym_indices[0])]
        used_metric = self.metrics[self.indices.index(sym_indices[0])]
        for sym_index in sym_indices[1:]:
            if self.indices_size[self.indices.index(sym_index)] != used_size:
                raise Exception(f'The given indices have different sizes')
            if not np.allclose(self.metrics[self.indices.index(sym_index)], used_metric):
                raise Exception(f'The given indices have different metrics')

        tensors_to_sum = []
        permutations, parities = self.__permutations(sym_indices)
        for permutation, sigma in zip(permutations, parities):
            new_Tensor = self.change_indices_name(sym_indices, permutation)
            if a_sym:
                if sigma == -1:
                    new_Tensor = -new_Tensor
            tensors_to_sum.append(new_Tensor)

        return Tensor.sum_all(tensors_to_sum, self.tensor_name) / math.factorial(len(sym_indices))

    @staticmethod
    def __get_iterable(max_value, asym_indices_position=[], sym_indices_position=[]):
        indices_value = [0, ] * len(max_value)
        if len(max_value) != 0:
            while indices_value[0] != max_value[0]:
                do_yield = True
                for a_indices_position in asym_indices_position:
                    a_indices_value = [indices_value[index] for index in a_indices_position]
                    if a_indices_value != sorted(a_indices_value):
                        do_yield = False
                    if len(a_indices_value) != len(set(a_indices_value)):
                        do_yield = False
                    if not do_yield:
                        break
                for s_indices_position in sym_indices_position:
                    s_indices_value = [indices_value[index] for index in s_indices_position]
                    if s_indices_value != sorted(s_indices_value):
                        do_yield = False
                    if not do_yield:
                        break
                if do_yield:
                    yield tuple(indices_value)
                indices_value[-1] += 1
                for i in range(len(max_value) - 1, 0, -1):
                    if max_value[i] == indices_value[i]:
                        indices_value[i] = 0
                        indices_value[i - 1] += 1
        else:
            yield ()

    @staticmethod
    def __permutations(elements):
        if len(elements) == 1:
            return [elements], [1]

        prev_permutations, prev_parities = Tensor.__permutations(elements[:-1])
        base_permutation = [perm + [elements[-1]] for perm in prev_permutations]
        inv_parities = [-sigma for sigma in prev_parities]
        parities = prev_parities
        permutations = deepcopy(base_permutation)
        for i in range(len(elements) - 1):
            new_permutation = deepcopy(base_permutation)
            for perm in new_permutation:
                perm[i], perm[-1] = perm[-1], perm[i]
            permutations += new_permutation
            parities += inv_parities

        return permutations, parities

    @staticmethod
    def __block_diagonal_matrix(matrices):
        if len(matrices) > 2:
            matrices = [Tensor.__block_diagonal_matrix(matrices[:-1]), matrices[-1]]
        block_diagonal = np.zeros((len(matrices[0]) + len(matrices[1]),) * 2)
        for i in range(len(matrices[0])):
            for j in range(len(matrices[0])):
                block_diagonal[i, j] = matrices[0][i, j]
        for i in range(len(matrices[1])):
            for j in range(len(matrices[1])):
                block_diagonal[len(matrices[0]) + i, len(matrices[0]) + j] = matrices[1][i, j]
        return block_diagonal

    @staticmethod
    def __list_tensor_to_list_np(list_blocks, indices, grouping_of_indices):
        if isinstance(list_blocks, list):
            return [Tensor.__list_tensor_to_list_np(elem, indices, grouping_of_indices) for elem in list_blocks]
        tensor_indices = list_blocks.indices.copy()
        for index, associated_indices, __ in grouping_of_indices:
            for associated_index in associated_indices:
                if associated_index in tensor_indices:
                    tensor_indices[tensor_indices.index(associated_index)] = index
                    break
        for i, index in enumerate(indices[:-1]):
            j = tensor_indices.index(index)
            if i != j:
                tensor_indices[i], tensor_indices[j] = tensor_indices[j], tensor_indices[i]
                list_blocks.tensor = np.swapaxes(list_blocks.tensor, i, j)
        return list_blocks.tensor


class System:
    # Initialize the system, if an initial solution is given substitute it in
    def __init__(self, list_eq, fixed_scaling, Sol=None):
        if Sol is None:
            Sol = [None, ] * len(list(list_eq[0])[0])
        else:
            # Substitute in the known values
            for index_eq, eq in enumerate(list_eq):
                new_eq = {}
                for var in eq:
                    new_var = list(var)
                    new_val = eq[var]
                    for index, elem in enumerate(var):
                        if Sol[index] is not None:
                            new_val *= Sol[index] ** elem
                            new_var[index] = 0
                    new_var = tuple(new_var)
                    if any(var_ == new_var for var_ in new_eq):
                        new_eq[new_var] += new_val
                    else:
                        new_eq.update({new_var: new_val})
                list_eq[index_eq] = new_eq

        self.Sol = Sol
        if all(elem is None or elem == 0 for elem in Sol):
            self.fixed_scaling = fixed_scaling
        else:
            self.fixed_scaling = True
        self.list_eq = []
        self.used_var = []
        self.num_used_var = []
        self.leading_var = []
        self.solvable = True
        self.determined = True
        self.order = 0
        self.need_to_step_up = False
        self.reduced_list_eq = []
        self.reduced_used_var = []
        self.reduced_num_used_var = []
        self.reduced_leading_var = []

        for eq in list_eq:
            # Delete the zero values
            val_to_del = []
            for val in eq:
                if -zero_tolerance < eq[val] < zero_tolerance:
                    val_to_del.append(val)
            for val in val_to_del:
                del eq[val]
            if len(eq) != 0:
                if len(eq) == 1 and all(elem == 0 for elem in list(eq)[0]):
                    self.solvable = False
                    return

                # Evaluate the used variables
                used_var_eq = [False, ] * len(Sol)
                for var in eq:
                    for index, elem in enumerate(var):
                        if elem != 0:
                            used_var_eq[index] = True

                self.list_eq.append(eq)
                self.used_var.append(used_var_eq)
                self.num_used_var.append(sum(used_var_eq))
                self.leading_var.append(self.__get_leading_var(eq))
                self.order = max(self.order, sum(self.leading_var[-1]))

        if not fixed_scaling:
            self.__standardize()

        self.__sort()

    # Search new solutions for the system of equation
    def find_new_Sol(self):
        # Create the (empty) list for the new solutions
        New_Solutions = []
        to_reduce = True
        # Check all the equations and find which can be solved
        for index_eq, eq in enumerate(self.list_eq):

            if self.num_used_var[index_eq] == 2:
                unk_vars = [index for index in range(len(self.Sol)) if self.used_var[index_eq][index]]
                # Get the maximum and minimum exponent for the powers
                max_px = 0
                max_py = 0
                min_px = tuple(eq)[0][unk_vars[0]]
                min_py = tuple(eq)[0][unk_vars[1]]
                for var in eq:
                    if var[unk_vars[0]] > max_px:
                        max_px = var[unk_vars[0]]
                    if var[unk_vars[1]] > max_py:
                        max_py = var[unk_vars[1]]
                    if var[unk_vars[0]] < min_px:
                        min_px = var[unk_vars[0]]
                    if var[unk_vars[1]] < min_py:
                        min_py = var[unk_vars[1]]
                # Checks whether a variable can be fully factored
                x_same_pow = True
                y_same_pow = True
                for var in eq:
                    if var[unk_vars[0]] != min_px:
                        x_same_pow = False
                    if var[unk_vars[1]] != min_py:
                        y_same_pow = False
                    if not x_same_pow and not y_same_pow:
                        break

                # If the scale is set and no variable can be fully factored no information can be extracted
                if not x_same_pow and not y_same_pow and self.fixed_scaling:
                    # TODO: check for possible factorization
                    continue

                to_reduce = False
                # If the scale is not set and no variable can be fully factored
                # set a variable to 1 and solve for the other
                if not x_same_pow and not y_same_pow and not self.fixed_scaling:
                    poly_y = [0, ] * (max_py - min_py + 1)
                    for var in eq:
                        poly_y[var[unk_vars[1]] - min_py] += eq[var]
                    New_Sol = self.Sol.copy()
                    New_Sol[unk_vars[0]] = 1
                    for sol_var in self.__polynomial_solver(poly_y):
                        New_Sol_ = New_Sol.copy()
                        New_Sol_[unk_vars[1]] = sol_var
                        New_Solutions.append(New_Sol_)
                    if min_px == 0 and min_py == 0:
                        New_Sol = self.Sol.copy()
                        New_Sol[unk_vars[0]] = 0
                        New_Sol[unk_vars[1]] = 0
                        New_Solutions.append(New_Sol)

                # If x can be fully factored solve for y
                elif x_same_pow and not y_same_pow:
                    poly_y = [0, ] * (max_py - min_py + 1)
                    for var in eq:
                        poly_y[var[unk_vars[1]] - min_py] += eq[var]
                    for sol_var in self.__polynomial_solver(poly_y):
                        New_Sol = self.Sol.copy()
                        New_Sol[unk_vars[1]] = sol_var
                        New_Solutions.append(New_Sol)

                # If y can be fully factored solve for x
                elif not x_same_pow and y_same_pow:
                    poly_x = [0, ] * (max_px - min_px + 1)
                    for var in eq:
                        poly_x[var[unk_vars[0]] - min_px] += eq[var]
                    for sol_var in self.__polynomial_solver(poly_x):
                        New_Sol = self.Sol.copy()
                        New_Sol[unk_vars[0]] = sol_var
                        New_Solutions.append(New_Sol)

                # If the equation can give information add the zero solutions, if present
                if min_px != 0:
                    New_Sol = self.Sol.copy()
                    New_Sol[unk_vars[0]] = 0
                    New_Solutions.append(New_Sol)
                if min_py != 0:
                    New_Sol = self.Sol.copy()
                    New_Sol[unk_vars[1]] = 0
                    New_Solutions.append(New_Sol)
                break

            elif self.num_used_var[index_eq] == 1:
                to_reduce = False
                unk_var = [index for index in range(len(self.Sol)) if self.used_var[index_eq][index]][0]
                poly = [0, ] * (max(var[unk_var] for var in eq) + 1)
                for var in eq:
                    poly[var[unk_var]] += eq[var]
                for sol_var in self.__polynomial_solver(poly):
                    New_Sol = self.Sol.copy()
                    New_Sol[unk_var] = sol_var
                    New_Solutions.append(New_Sol)
                break

        return New_Solutions, to_reduce

    # Simplify the system
    def reduce(self):
        self.need_to_step_up = True
        new_list_eq = self.list_eq
        new_used_var = self.used_var
        new_leading_var = self.leading_var
        old_list_eq = []
        old_used_var = []
        old_leading_var = []
        while len(new_list_eq) != 0:
            new_list_eq, new_used_var, new_leading_var, old_list_eq, old_used_var, old_leading_var = self.__reduce_step(new_list_eq, new_used_var, new_leading_var, old_list_eq, old_used_var, old_leading_var)
        self.list_eq = old_list_eq
        self.used_var = old_used_var
        self.num_used_var = [sum(used_var_eq) for used_var_eq in self.used_var]
        self.leading_var = old_leading_var
        self.__sort()

    def __reduce_step(self, new_list_eq, new_used_var, new_leading_var, old_list_eq, old_used_var, old_leading_var):
        next_list_eq = []
        next_used_var = []
        next_leading_var = []
        indices_new_to_remove = []
        for index_new_1, new_eq_1 in enumerate(new_list_eq):
            for index_new_2, new_eq_2 in enumerate(new_list_eq):
                if index_new_2 > index_new_1 and index_new_1 not in indices_new_to_remove and index_new_2 not in indices_new_to_remove:
                    type_child, child_eq, child_used_var, child_leading_var = self.__get_child(new_eq_1, new_used_var[index_new_1], new_leading_var[index_new_1], new_eq_2, new_used_var[index_new_2], new_leading_var[index_new_2])
                    if type_child != 0:
                        if type_child == -2:
                            self.solvable = False
                            return [], [], [], [], [], []
                        if type_child == -1:
                            indices_new_to_remove.append(index_new_1)
                        else:
                            self.need_to_step_up = False
                            next_list_eq.append(child_eq)
                            next_used_var.append(child_used_var)
                            next_leading_var.append(child_leading_var)
                            if type_child == 1:
                                indices_new_to_remove.append(index_new_1)
                            if type_child == 2:
                                indices_new_to_remove.append(index_new_2)
        for index_new in sorted(indices_new_to_remove, reverse=True):
            new_list_eq.pop(index_new)
            new_used_var.pop(index_new)
            new_leading_var.pop(index_new)
        indices_new_to_remove = []
        indices_old_to_remove = []
        for index_old, old_eq in enumerate(old_list_eq):
            for index_new, new_eq in enumerate(new_list_eq):
                if index_new not in indices_new_to_remove and index_old not in indices_new_to_remove:
                    type_child, child_eq, child_used_var, child_leading_var = System.__get_child(new_eq, new_used_var[index_new], new_leading_var[index_new], old_eq, old_used_var[index_old], old_leading_var[index_old])
                    if type_child != 0:
                        if type_child == -2:
                            self.solvable = False
                            return [], [], [], [], [], []
                        if type_child == -1:
                            indices_old_to_remove.append(index_old)
                        else:
                            next_list_eq.append(child_eq)
                            next_used_var.append(child_used_var)
                            next_leading_var.append(child_leading_var)
                            if type_child == 1:
                                indices_new_to_remove.append(index_new)
                            if type_child == 2:
                                indices_old_to_remove.append(index_old)
        for index_new in sorted(indices_new_to_remove, reverse=True):
            new_list_eq.pop(index_new)
            new_used_var.pop(index_new)
            new_leading_var.pop(index_new)
        for index_old in sorted(indices_old_to_remove, reverse=True):
            old_list_eq.pop(index_old)
            old_used_var.pop(index_old)
            old_leading_var.pop(index_old)

        return next_list_eq, next_used_var, next_leading_var, old_list_eq + new_list_eq, old_used_var + new_used_var, old_leading_var + new_leading_var

    @staticmethod
    def __get_child(eq_1, used_var_1, leading_var_1, eq_2, used_var_2, leading_var_2):
        if sum((elem != used_var_1[index] for index, elem in enumerate(used_var_2))) > max(sum(used_var_1), sum(used_var_2)):
            return 0, None, None, None
        if sum(used_var_1) < sum(used_var_2) or (sum(used_var_1) == sum(used_var_2) and (
                len(eq_1) < len(eq_2) or (len(eq_1) == len(eq_2) and System.__has_priority(leading_var_1, leading_var_2)))):
            ref_eq = eq_2
            ref_used_var = used_var_2
            ref_leading_var = leading_var_2
            type_child = 2
        else:
            ref_eq = eq_1
            ref_used_var = used_var_1
            ref_leading_var = leading_var_1
            type_child = 1
        total_vars = set(eq_1).union(set(eq_2))
        found_new = False
        for common_vars in set(eq_1).intersection(set(eq_2)):
            # Generate a candidate
            equation_candidate = {}
            for var in total_vars:
                num = 0
                if var in eq_1:
                    num += eq_1[var] * eq_2[common_vars]
                if var in eq_2:
                    num -= eq_2[var] * eq_1[common_vars]
                if num < -zero_tolerance or num > zero_tolerance:
                    equation_candidate.update({var: num})
            if len(equation_candidate) == 0:
                return -1, None, None, None
            if len(equation_candidate) == 1 and all(elem == 0 for elem in list(equation_candidate)[0]):
                return -2, None, None, None
            used_var_eq_cand = [False, ] * len(used_var_1)
            for var in equation_candidate:
                for index, elem in enumerate(var):
                    if elem != 0:
                        used_var_eq_cand[index] = True
            leading_var_eq_cand = System.__get_leading_var(equation_candidate)

            if sum(used_var_eq_cand) < sum(ref_used_var) or (sum(used_var_eq_cand) == sum(ref_used_var) and (
                    len(equation_candidate) < len(ref_eq) or (
                    len(equation_candidate) == len(ref_eq) and System.__has_priority(leading_var_eq_cand, ref_leading_var)))):
                found_new = True
                ref_eq = equation_candidate
                ref_used_var = used_var_eq_cand
                ref_leading_var = leading_var_eq_cand

        if found_new:
            norm_factor = 0
            for var in ref_eq:
                norm_factor += ref_eq[var] ** 2
            for var in ref_eq:
                ref_eq[var] /= math.sqrt(norm_factor)
            return type_child, ref_eq, ref_used_var, ref_leading_var
        else:
            return 0, [], [], []

    # Increase the order of the system by 1 as for the XL-algorithm
    def step_up(self):
        self.need_to_step_up = False
        self.order += 1
        new_list_eq = []
        new_used_var = []
        new_num_used_var = []
        new_leading_var = []
        for index, elem in enumerate(self.Sol):
            if elem is None:
                for index_eq, eq in enumerate(self.list_eq):
                    new_eq = {}
                    used_var_eq = self.used_var[index_eq].copy()
                    used_var_eq[index] = True
                    for var in eq:
                        new_var = list(var)
                        new_var[index] += 1
                        new_eq.update({tuple(new_var): eq[var]})
                    new_list_eq.append(new_eq)
                    new_used_var.append(used_var_eq)
                    new_num_used_var.append(sum(used_var_eq))
                    new_leading_var.append(self.__get_leading_var(new_eq))
        self.list_eq += new_list_eq
        self.used_var += new_used_var
        self.num_used_var += new_num_used_var
        self.leading_var += new_leading_var

    # If possible, set the firs non-zero known variable to 1 and scale everything accordingly
    def __standardize(self):
        norm = None
        for elem in self.Sol:
            if elem is not None and elem != 0:
                norm = elem
                break
        if norm is not None:
            for index, elem in enumerate(self.Sol):
                if elem is not None:
                    self.Sol[index] /= norm
            for index_eq, eq in enumerate(self.list_eq):
                for var in eq:
                    self.list_eq[index_eq][var] *= norm ** sum(var)

    # Generate the comparator needed for the sorter
    def __comparator(self):
        def compare(index1, index2):
            if self.num_used_var[index1] < self.num_used_var[index2]:
                return -1
            elif self.num_used_var[index1] > self.num_used_var[index2]:
                return 1
            else:
                if self.leading_var[index1] != self.leading_var[index2]:
                    if self.__has_priority(self.leading_var[index1], self.leading_var[index2]):
                        return -1
                    else:
                        return 1
                else:
                    if len(self.list_eq[index1]) < len(self.list_eq[index2]):
                        return -1
                    elif len(self.list_eq[index1]) > len(self.list_eq[index2]):
                        return 1
                    else:
                        return 0

        return compare

    # Sort the equation by three criterion, by relevance:
    # the least number of variables, the greatest priority, the least number of var
    def __sort(self):
        sorted_list_eq = []
        sorted_used_var = []
        sorted_num_used_var = []
        sorted_leading_var = []
        sorted_indices = sorted(list(range(len(self.list_eq))), key=cmp_to_key(self.__comparator()))
        for index_eq in sorted_indices:
            sorted_list_eq.append(self.list_eq[index_eq])
            sorted_used_var.append(self.used_var[index_eq])
            sorted_num_used_var.append(self.num_used_var[index_eq])
            sorted_leading_var.append(self.leading_var[index_eq])
        self.list_eq = sorted_list_eq
        self.used_var = sorted_used_var
        self.num_used_var = sorted_num_used_var
        self.leading_var = sorted_leading_var

    # Print the equations
    def print_equations(self):
        for eq in self.list_eq:
            str_eq = ''
            for index, var in enumerate(eq):
                if index != 0 and eq[var] > 0:
                    str_eq += '+'
                if eq[var] != 1:
                    str_eq += str(eq[var])
                if eq[var] == -1:
                    str_eq += '-'
                for var_index, elem in enumerate(var):
                    if elem != 0:
                        str_eq += 'x_' + str(var_index)
                    if elem > 1:
                        str_eq += '^' + str(elem)
            print(str_eq + '=0')
        print('\n')

    @staticmethod
    def __polynomial_solver(poly):
        if len(poly) == 0:
            return None
        if len(poly) == 1:
            return []
        if len(poly) == 2:
            return [-poly[0] / poly[1]]

        D_poly = [(p + 1) * coefficient for p, coefficient in enumerate(poly[1:])]
        intervals = System.__polynomial_solver(D_poly)
        if len(intervals) == 0:
            intervals = [0]
        values = [System.__poly_eval(poly, interval) for interval in intervals]
        coeff = -poly[-1] * (-1) ** len(poly)
        if values[0] * coeff < 0:
            step = 1
            x_inf = intervals[0] - 1
            val_x_inf = System.__poly_eval(poly, x_inf)
            while val_x_inf * coeff < 0:
                step *= 2
                x_inf -= step
                val_x_inf = System.__poly_eval(poly, x_inf)
            intervals.insert(0, x_inf)
            values.insert(0, val_x_inf)
        if values[-1] * poly[-1] < 0:
            step = 1
            x_sup = intervals[-1] + 1
            val_x_sup = System.__poly_eval(poly, x_sup)
            while val_x_sup * poly[-1] < 0:
                step *= 2
                x_sup += step
                val_x_sup = System.__poly_eval(poly, x_sup)
            intervals.append(x_sup)
            values.append(val_x_sup)

        roots = []
        if len(intervals) == 1 and values[0] == 0:
            return [intervals[0]]
        for index, value in enumerate(values[:-1]):
            if zero_tolerance < math.fabs(value) and zero_tolerance < math.fabs(values[index + 1]) and value * values[index + 1] < 0:
                roots.append(System.__poly_sol_between(poly, intervals[index], intervals[index + 1]))
            if -zero_tolerance < value < zero_tolerance:
                if len(roots) == 0 or intervals[index] - roots[-1] > zero_tolerance:
                    roots.append(intervals[index])
                else:
                    roots[-1] = (roots[-1] + intervals[index]) / 2
        if -zero_tolerance < values[-1] < zero_tolerance:
            if len(roots) == 0 or intervals[-1] - roots[-1] > zero_tolerance:
                roots.append(intervals[-1])
            else:
                roots[-1] = (roots[-1] + intervals[-1]) / 2
        return roots

    @staticmethod
    def __poly_sol_between(poly, interval1, interval2):
        x1 = interval1
        val1 = System.__poly_eval(poly, x1)
        if val1 == 0:
            return x1
        x2 = interval2
        val2 = System.__poly_eval(poly, x2)
        if val2 == 0:
            return x2
        while val2 < -zero_tolerance or zero_tolerance < val2 or (x2 - x1) > zero_tolerance / 1000:
            x_mean = (x1 + x2) / 2
            val_mean = System.__poly_eval(poly, x_mean)
            if val_mean == 0:
                return x_mean
            if val1 * val_mean < 0:
                x2 = x_mean
                val2 = val_mean
            else:
                x1 = x_mean
                val1 = val_mean
        return x2

    @staticmethod
    def __poly_eval(poly, x):
        value = 0
        for p, coefficient in enumerate(poly):
            value += coefficient * x ** p
        return value

    # The higher order var have priority
    @staticmethod
    def __has_priority(var1, var2):
        sum1 = sum(var1)
        sum2 = sum(var2)
        if sum1 > sum2:
            return True
        if sum1 < sum2:
            return False
        for index, elem in enumerate(var1):
            if elem > var2[index]:
                return True
            if elem < var2[index]:
                return False
        return False

    # Get the variable with the greatest priority
    @staticmethod
    def __get_leading_var(vars_):
        vars_ = tuple(vars_)
        max_var = vars_[0]
        for var in vars_[1:]:
            if System.__has_priority(var, max_var):
                max_var = var
        return max_var


class Solver:
    def __init__(self, list_equations, max_order=7, fixed_scaling=None):
        if len(list_equations) == 0:
            raise Exception(f'Empty list of equations')
        while {} in list_equations:
            list_equations.remove({})
        num_variables = len(tuple(list_equations[0])[0])
        for eq in list_equations[1:]:
            if len(tuple(eq)[0]) != num_variables:
                raise Exception(f'Inconsistent number of variables')

        self.num_variables = list(list_equations[0])[0]
        if fixed_scaling is not None:
            self.fixed_scaling = fixed_scaling
        else:
            self.fixed_scaling = False
            for eq in list_equations:
                if not all(sum(list(eq)[0]) == sum(var) for var in eq):
                    self.fixed_scaling = True
                    break
        self.max_order = max_order

        self.list_systems = [System(list_equations, self.fixed_scaling)]  # [1/math.sqrt(3), -1/math.sqrt(3), 1/math.sqrt(3), 1/math.sqrt(3), None, None], [None, None, None, None, None, None]
        self.Partial_Solutions = []
        self.Complete_Solutions = []
        self.Undetermined_Solutions = []

    def get_Solutions(self, max_num_of_steps=-1):
        while len(self.list_systems) != 0 and max_num_of_steps != 0:
            print(f'{len(self.Complete_Solutions)} solutions found, {len(self.list_systems)} remaining systems')
            self.__step()
            max_num_of_steps -= 1
        return self.Complete_Solutions, self.Undetermined_Solutions

    # Do one step of the XL-algorithm to a system
    def __step(self):
        chosen_index = 0
        num_eq = len(self.list_systems[0].list_eq)
        for index, system in enumerate(self.list_systems):
            if len(system.list_eq) < num_eq:
                chosen_index = index
                num_eq = len(system.list_eq)

        to_reduce_systems = []
        list_systems = [self.list_systems.pop(chosen_index)]
        # After new solutions are found for a system check if others are present
        while len(list_systems) != 0:
            new_systems = []
            for system in list_systems:
                if system.order > self.max_order:
                    self.Undetermined_Solutions.append(system.Sol)
                else:
                    # Get the new solutions
                    new_Solutions, to_reduce = system.find_new_Sol()
                    if to_reduce:
                        # If no more solutions can be extracted from the system save it
                        to_reduce_systems.append(system)
                    else:
                        # Create the new systems from the solutions
                        for new_Sol in new_Solutions:
                            new_system = System(system.list_eq.copy(), fixed_scaling=system.fixed_scaling, Sol=new_Sol)
                            if new_system.solvable:
                                if len(new_system.list_eq) == 0:
                                    if self.__is_new(new_Sol, 'c'):
                                        self.Complete_Solutions.append(new_Sol)
                                elif self.__is_new(new_Sol, 'p'):
                                    self.Partial_Solutions.append(new_Sol)
                                    new_systems.append(new_system)
            list_systems = new_systems

        for index_system, system in enumerate(to_reduce_systems):
            # simplify all the systems
            system.reduce()
            if system.solvable:
                if not system.determined:
                    self.Undetermined_Solutions.append(system.Sol)
                else:
                    # If the previous search of solutions and reduction were unsuccessful step up the system
                    if system.need_to_step_up:
                        system.step_up()
                    self.list_systems.append(system)
                    print(f'Solution to elaborate: {system.Sol}, order of the system: {system.order}, number of equations: {len(system.list_eq)}')

    # Check if the solution is already present in the corresponding list
    def __is_new(self, Sol, type_Sol):
        list_to_check = []
        if type_Sol == 'c':
            list_to_check = self.Complete_Solutions
        elif type_Sol == 'p':
            list_to_check = self.Partial_Solutions

        for Sol_ in list_to_check:
            are_equal = True
            for index, elem in enumerate(Sol_):
                if (Sol[index] is None) == (elem is not None):
                    are_equal = False
                    break
                if elem is not None and (Sol[index] - elem < -zero_tolerance or zero_tolerance < Sol[index] - elem):
                    are_equal = False
                    break
            if are_equal:
                return False
        return True
