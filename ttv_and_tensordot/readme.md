# A few experiments with HPC and tensorly

This branch is a proof of concept of the importance of careful craft of efficient tensor contractions at the back-end level to speed up algorithms in tensorly.

Three things are added:
1. Tensordot for modedot
2. TTV for modedot with vectors
3. Permutation of tensor to minimize number of flops required.

note: plotting requires `plotly` and possibly `kaleido`

### 1. Tensordot
I replaced calls to tl.dot(unfold(tensor),...)  with tensordot from the backends. The advantages are:
+ No thinking about unfolding/permutations, higher level API
+ Support for any HPC method that would be wired through the tensordot API (think cuTensor in Pytorch...)

But on the downside:
- tensordot is often slower than the naive tensorly method (expecially with numpy)
- Backends currently provide tensordot thinking of contracting two tensors with arbitrary number of modes. Therefore to have a general working rule, the modes of the second tensor contracted are appened at the end of the first tensor after contraction. When performing tensor contractions with matrices, we rather want to insert the resulting mode back where contraction happened, and this yields permutations all over the place in the algorithms (see 3. for a discussion on why permutations are really bad). This could be solved if the backends provided a richer API.

Experiments showing this are in `testing_tensordot_tucker.py` and `testing_tensordot_tucker_pytorch.py` for usage of tensordot in the HOOI algorithm to compute Tucker; and in `testing_ttv_parafac.py` for Parafac, along with the TTV version (see below).

Pre-computed results are shown in `benchmark_tensordot_numpy_tucker.png` and `benmark_tensordot_pytorch_tucker.png` for the Tucker test as well as `benchmark_ttv_numpy.png` for the Parafac test.

### 2. Fast TTV vs Tensordot vs Current "legacy" unfolded transpose dot.
As a proof of concept, during Dagstuhl seminar 22101 on Tensor computations, we tried to connect Cem Bassoy's TTV C/C++ library for fast Tensor-Times-Vector (TTV) with tensorly, to grasp the potential for speed-up if doing tensor contractions efficiently at the back-end level (the current naive version relies on GEMM with folding and permuting being unnecessary overheads in theory).

To run these experiments, you need to install the ttvpy package provided at the [ttv package](https://github.com/bassoy/ttv/tree/master/ttvpy) repo, and follow instructions there on how to install (python wrapping is done through pybind11, credits to Cem Bassoy for setting this up!). Disclaimer: I has to use `sudo pip install -e .`, and also manually add the path to the header `cblas.h` in `ttvpy/setup.py`.

Then anytime the keyword `fast=True` is used in parafac or a mode-product using the Numpy backend, it actually is handled using `ttvpy.ttv`.

Results can be computed using `testing_ttv_parafac.py`, and a precomputed result is shown in `benchmark_ttv_numpy.png`

### 3. Optimal order: theory vs practice
Another quick test on the side: I noticed during the ttv implementation that the order of the contractions in the nmode products (sequences of mode products such as TTV and TTMatrix) was arbitrarily set to [0,1,...,d] where d is the input tensor order. Several things could be said here:
- Using a loop at the Python level to perform several TTVs or TTMs is not efficient. We could hope for far better performances if several TTV (TTVs) and several TTM (TTMs) where provided directly by the backend. Think about the above point 2. but with a ttvpy.ttvs routine (or even better for Parafac would be a MTTKRP routine) instead of ttv.
- Formally speaking, when performing TTVs, the sequence order in which the individual TTV are performed matters a lot. Since each contraction reduces the order of the tensor by one, it is optimal to keep the largest contraction last to minimize the number of products to compute. When TTMs are performed a similar argument holds.

To test out the second bullet point (optimal ordering), I went ahead and permuted the modes of an input tensor in tensorly at the start of the algorithm. Results are computed in `testing_order_nmodedot.py`. It can be seen that this strategy fails quite badly, because the theory did not consider a very important aspect of computation: data layout. By permuting the tensor, the striding is not kept so that last mode is stored first (C ordering), and this makes contractions very slow. On the other hand, contracting with the first mode then second and so on, whatever the sizes (current tensorly implementation), is really hard to beat using only permutations. This means the optimal contraction order competes non-trivially with the striding.

Precomputed results are shown in `benchmark_tensordot_orderopt_numpy.png`.