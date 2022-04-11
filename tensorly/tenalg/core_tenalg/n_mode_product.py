from re import M
from ... import backend as T
from ... import unfold, fold, vec_to_tensor
# special gift from Cem Bassoy --> todo: move to backend
try:
    import ttvpy as tp
except:
    print('No local ttvpy package detected')

def mode_dot(tensor, matrix_or_vector, mode, transpose=False, fast=False):
        """n-mode product of a tensor and a matrix or vector at the specified mode

        Mathematically: :math:`\\text{tensor} \\times_{\\text{mode}} \\text{matrix or vector}`


        Parameters
        ----------
        tensor : ndarray
            tensor of shape ``(i_1, ..., i_k, ..., i_N)``
        matrix_or_vector : ndarray
            1D or 2D array of shape ``(J, i_k)`` or ``(i_k, )``
            matrix or vectors to which to n-mode multiply the tensor
        mode : int
        transpose : bool, default is False
            If True, the matrix is transposed. 
            For complex tensors, the conjugate transpose is used. 
        fast : bool or "legacy"
            Controls if we are running optimized ttv or not. for testing only
            "legacy" computes the regular nmode product from tensorly
            'ttv' uses Cem Bassoy's TTV implementation (restrictions: numpy and vector contraction only)
            "ttvs" for full c implementation of several ttv (Cem Bassoy's TTVs), requires skip. Uses ttv if ttvs do not make sense.
            'tensordot' uses backend tensordot

        Returns
        -------
        ndarray
            `mode`-mode product of `tensor` by `matrix_or_vector`
            * of shape :math:`(i_1, ..., i_{k-1}, J, i_{k+1}, ..., i_N)` if matrix_or_vector is a matrix
            * of shape :math:`(i_1, ..., i_{k-1}, i_{k+1}, ..., i_N)` if matrix_or_vector is a vector

        See also
        --------
        multi_mode_dot : chaining several mode_dot in one call
        """

        if fast=="legacy":
            # the mode along which to fold might decrease if we take product with a vector
            fold_mode = mode
            new_shape = list(tensor.shape)
            if T.ndim(matrix_or_vector) == 2:  # Tensor times matrix
                # Test for the validity of the operation
                dim = 0 if transpose else 1
                if matrix_or_vector.shape[dim] != tensor.shape[mode]:
                    raise ValueError(
                        'shapes {0} and {1} not aligned in mode-{2} multiplication: {3} (mode {2}) != {4} (dim 1 of matrix)'.format(
                            tensor.shape, matrix_or_vector.shape, mode, tensor.shape[mode], matrix_or_vector.shape[dim]
                        ))
                
                if transpose:
                    matrix_or_vector = T.conj(T.transpose(matrix_or_vector))

                new_shape[mode] = matrix_or_vector.shape[0]
                vec = False

            elif T.ndim(matrix_or_vector) == 1:  # Tensor times vector
                dim = 0
                if matrix_or_vector.shape[0] != tensor.shape[mode]:
                    raise ValueError(
                        'shapes {0} and {1} not aligned for mode-{2} multiplication: {3} (mode {2}) != {4} (vector size)'.format(
                            tensor.shape, matrix_or_vector.shape, mode, tensor.shape[mode], matrix_or_vector.shape[0]
                        ))
                if len(new_shape) > 1:
                    new_shape.pop(mode)
                else:
                    # Ideally this should be (), i.e. order-0 tensors
                    # MXNet currently doesn't support this though..
                    new_shape = []
                vec = True


            else:
                raise ValueError('Can only take n_mode_product with a vector or a matrix.'
                                'Provided array of dimension {} not in [1, 2].'.format(T.ndim(matrix_or_vector)))
            
            res = T.dot(matrix_or_vector, unfold(tensor, mode))

            if vec: # We contracted with a vector, leading to a vector
                return vec_to_tensor(res, shape=new_shape)
            else: # tensor times vec: refold the unfolding
                return fold(res, fold_mode, new_shape)

        else:
            if T.ndim(matrix_or_vector) == 2:  # Tensor times matrix
                # Test for the validity of the operation
                dim = 0 if transpose else 1
                if matrix_or_vector.shape[dim] != tensor.shape[mode]:
                    raise ValueError(
                        'shapes {0} and {1} not aligned in mode-{2} multiplication: {3} (mode {2}) != {4} (dim 1 of matrix)'.format(
                            tensor.shape, matrix_or_vector.shape, mode, tensor.shape[mode], matrix_or_vector.shape[dim]
                        ))
                
                if transpose:
                    # we should support conjugaison without transposition, to avoid moving the data around here.
                    matrix_or_vector = T.conj(matrix_or_vector)


            elif T.ndim(matrix_or_vector) == 1:  # Tensor times vector
                dim = 0
                if matrix_or_vector.shape[0] != tensor.shape[mode]:
                    raise ValueError(
                        'shapes {0} and {1} not aligned for mode-{2} multiplication: {3} (mode {2}) != {4} (vector size)'.format(
                            tensor.shape, matrix_or_vector.shape, mode, tensor.shape[mode], matrix_or_vector.shape[0]
                        ))

            else:
                raise ValueError('Can only take n_mode_product with a vector or a matrix.'
                                'Provided array of dimension {} not in [1, 2].'.format(T.ndim(matrix_or_vector)))
            
            return T.tensordot(tensor,matrix_or_vector, axes=([mode],[dim]), fast=fast)


def ttvs(q,a,b):
    # Using Cem Bassoy's implementation of several tensor-times-vector contractions,
    #  here skipping mode q an contraction a with all vectors in list b.
    order = "backward"
    return tp.ttvs(q+1,a,b,order)

def multi_mode_dot(tensor, matrix_or_vec_list, modes=None, skip=None, transpose=False, fast = False):
    """n-mode product of a tensor and several matrices or vectors over several modes

    Parameters
    ----------
    tensor : ndarray

    matrix_or_vec_list : list of matrices or vectors of length ``tensor.ndim``

    skip : None or int, optional, default is None
        If not None, index of a matrix to skip. 
        Note that in any case, `modes`, if provided, should have a length of ``tensor.ndim``

    modes : None or int list, optional, default is None

    transpose : bool, optional, default is False
        If True, the matrices or vectors in in the list are transposed.
        For complex tensors, the conjugate transpose is used. 
        
    fast : bool or "legacy"
        Controls if we are running optimized ttv or not. for testing only
        "legacy" computes the regular nmode product from tensorly
        'ttv' uses Cem Bassoy's TTV implementation (restrictions: numpy and vector contraction only)
        "ttvs" for full c implementation of several ttv (Cem Bassoy's TTVs), requires skip. Uses ttv if ttvs do not make sense.
        'tensordot' uses backend tensordot

    Returns
    -------
    ndarray
        tensor times each matrix or vector in the list at mode `mode`

    Notes
    -----
    If no modes are specified, just assumes there is one matrix or vector per mode and returns:

    :math:`\\text{tensor  }\\times_0 \\text{ matrix or vec list[0] }\\times_1 \\cdots \\times_n \\text{ matrix or vec list[n] }`

    See also
    --------
    mode_dot
    """
    if T.get_backend()=='numpy' and fast=='ttvs' and skip>=0:
        # Calling TTVs
        shapes_are_one = [matrix_or_vec_list[i].ndim==1 for i in range(len(matrix_or_vec_list))]
        if all(shapes_are_one):
            # poping skiped vector
            matrix_or_vec_list.pop(skip)
            # calling ttvs from the backend, actually ttvpy routine
            # using default order
            return ttvs(skip, tensor, matrix_or_vec_list)


    if modes is None:
        modes = range(len(matrix_or_vec_list))

    decrement = 0  # If we multiply by a vector, we diminish the dimension of the tensor

    res = tensor

    # Order of mode dots doesn't matter for different modes
    # Sorting by mode shouldn't change order for equal modes
    factors_modes = sorted(zip(matrix_or_vec_list, modes), key=lambda x: x[1])
    for i, (matrix_or_vec, mode) in enumerate(factors_modes):
        if (skip is not None) and (i == skip):
            continue

        if transpose:
            res = mode_dot(res, T.conj(T.transpose(matrix_or_vec)), mode - decrement, fast=fast)
        else:
            res = mode_dot(res, matrix_or_vec, mode - decrement, fast=fast)

        if T.ndim(matrix_or_vec) == 1:
            decrement += 1


    #if modes is None:
        #modes = range(len(matrix_or_vec_list))

    #decrement = [0 for i in modes]  # If we multiply by a vector, we diminish the dimension of the tensor

    #res = tensor

    ## Order of mode dots doesn't matter for different modes
    ## Sorting by mode shouldn't change order for equal modes
    ## list of (array, mode)
    ##factors_modes = sorted(zip(matrix_or_vec_list, modes), key=lambda x: x[1])

    ## here we should order modes by increasing ratio dim[1]/dim[0] (if contraction on second mode) (ask ref Bora)
    #ratio = []
    #for matrix_or_vec in matrix_or_vec_list:
        #if matrix_or_vec.ndim==1:
            #ratio.append(matrix_or_vec.shape[0])
        #else:
            #if transpose:
                #ratio.append(matrix_or_vec.shape[0]/matrix_or_vec.shape[1])
            #else:
                #ratio.append(matrix_or_vec.shape[1]/matrix_or_vec.shape[0])
        #factors_modes = sorted(zip(matrix_or_vec_list, modes, ratio), key=lambda x: x[1])

    ## we now need to handle ndim reduction after each contraction
    #for i, (matrix_or_vec, mode, _) in enumerate(factors_modes):
        #if (skip is not None) and (mode == skip):
            #continue

        #if transpose:
            #res = mode_dot(res, T.conj(T.transpose(matrix_or_vec)), mode - decrement[i], fast=fast)
        #else:
            #res = mode_dot(res, matrix_or_vec, mode - decrement[i], fast=fast)

        #if T.ndim(matrix_or_vec) == 1:
            #decrement = [decrement[j] + 1 if factors_modes[j][1]>mode else decrement[j] for j in modes]
            ##decrement += 1

    return res

