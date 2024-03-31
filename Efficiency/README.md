# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py

Task 3.1 Parallelization & Task 3.2 Matrix Multiplication

```
MAP
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_map.<locals>._map, 
/Users/jiangguangwei/Desktop/MLE/mle-
module-3-GuangweiJiang312/minitorch/fast_ops.py (154)  
================================================================================


Parallel loop listing for  Function tensor_map.<locals>._map, /Users/jiangguangwei/Desktop/MLE/mle-module-3-GuangweiJiang312/minitorch/fast_ops.py (154) 
-----------------------------------------------------------------------------|loop #ID
    def _map(                                                                | 
        out: Storage,                                                        | 
        out_shape: Shape,                                                    | 
        out_strides: Strides,                                                | 
        in_storage: Storage,                                                 | 
        in_shape: Shape,                                                     | 
        in_strides: Strides,                                                 | 
    ) -> None:                                                               | 
        # TODO: Implement for Task 3.1.                                      | 
        if(                                                                  | 
            len(out_strides) != len(in_strides)                              | 
            or (out_strides != in_strides).any()-----------------------------| #0
            or (out_shape != in_shape).any()---------------------------------| #1
        ):                                                                   | 
            for i in prange(len(out)):---------------------------------------| #3
                out_index = np.empty(MAX_DIMS, np.int32)                     | 
                in_index = np.empty(MAX_DIMS, np.int32)                      | 
                to_index(i, out_shape, out_index)                            | 
                broadcast_index(out_index, out_shape, in_shape, in_index)    | 
                inpos = index_to_position(in_index, in_strides)              | 
                out[i] = fn(in_storage[inpos])                               | 
        else:                                                                | 
            for i in prange(len(out)):---------------------------------------| #2
                out[i] = fn(in_storage[i])                                   | 
        # raise NotImplementedError("Need to implement for Task 3.1")        | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 4 parallel for-
loop(s) (originating from loops labelled: #0, #1, #3, #2).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/Users/jiangguangwei/Desktop/MLE/mle-
module-3-GuangweiJiang312/minitorch/fast_ops.py (169) is hoisted out of the 
parallel loop labelled #3 (it will be performed before the loop is executed and 
reused inside the loop):
   Allocation:: out_index = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/jiangguangwei/Desktop/MLE/mle-
module-3-GuangweiJiang312/minitorch/fast_ops.py (170) is hoisted out of the 
parallel loop labelled #3 (it will be performed before the loop is executed and 
reused inside the loop):
   Allocation:: in_index = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
ZIP
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip, 
/Users/jiangguangwei/Desktop/MLE/mle-
module-3-GuangweiJiang312/minitorch/fast_ops.py (205)  
================================================================================


Parallel loop listing for  Function tensor_zip.<locals>._zip, /Users/jiangguangwei/Desktop/MLE/mle-module-3-GuangweiJiang312/minitorch/fast_ops.py (205) 
---------------------------------------------------------------------------|loop #ID
    def _zip(                                                              | 
        out: Storage,                                                      | 
        out_shape: Shape,                                                  | 
        out_strides: Strides,                                              | 
        a_storage: Storage,                                                | 
        a_shape: Shape,                                                    | 
        a_strides: Strides,                                                | 
        b_storage: Storage,                                                | 
        b_shape: Shape,                                                    | 
        b_strides: Strides,                                                | 
    ) -> None:                                                             | 
        # TODO: Implement for Task 3.1.                                    | 
        if(                                                                | 
            len(out_strides) != len(a_strides)                             | 
            or len(out_strides) != len(b_strides)                          | 
            or (out_strides != a_strides).any()----------------------------| #4
            or (out_strides != b_strides).any()----------------------------| #5
            or (out_shape != a_shape).any()--------------------------------| #6
            or (out_shape != b_shape).any()--------------------------------| #7
        ):                                                                 | 
            for i in prange(len(out)):-------------------------------------| #9
                out_index = np.empty(MAX_DIMS, np.int32)                   | 
                a_index = np.empty(MAX_DIMS, np.int32)                     | 
                b_index = np.empty(MAX_DIMS, np.int32)                     | 
                to_index(i, out_shape, out_index)                          | 
                broadcast_index(out_index, out_shape, a_shape, a_index)    | 
                a = index_to_position(a_index, a_strides)                  | 
                broadcast_index(out_index, out_shape, b_shape, b_index)    | 
                b = index_to_position(b_index, b_strides)                  | 
                out[i] = fn(a_storage[a], b_storage[b])                    | 
        else:                                                              | 
            for i in prange(len(out)):-------------------------------------| #8
                out[i] = fn(a_storage[i], b_storage[i])                    | 
        # raise NotImplementedError("Need to implement for Task 3.1")      | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 6 parallel for-
loop(s) (originating from loops labelled: #4, #5, #6, #7, #9, #8).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/Users/jiangguangwei/Desktop/MLE/mle-
module-3-GuangweiJiang312/minitorch/fast_ops.py (226) is hoisted out of the 
parallel loop labelled #9 (it will be performed before the loop is executed and 
reused inside the loop):
   Allocation:: out_index = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/jiangguangwei/Desktop/MLE/mle-
module-3-GuangweiJiang312/minitorch/fast_ops.py (227) is hoisted out of the 
parallel loop labelled #9 (it will be performed before the loop is executed and 
reused inside the loop):
   Allocation:: a_index = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/jiangguangwei/Desktop/MLE/mle-
module-3-GuangweiJiang312/minitorch/fast_ops.py (228) is hoisted out of the 
parallel loop labelled #9 (it will be performed before the loop is executed and 
reused inside the loop):
   Allocation:: b_index = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
REDUCE
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce, 
/Users/jiangguangwei/Desktop/MLE/mle-
module-3-GuangweiJiang312/minitorch/fast_ops.py (262)  
================================================================================


Parallel loop listing for  Function tensor_reduce.<locals>._reduce, /Users/jiangguangwei/Desktop/MLE/mle-module-3-GuangweiJiang312/minitorch/fast_ops.py (262) 
-------------------------------------------------------------------------|loop #ID
    def _reduce(                                                         | 
        out: Storage,                                                    | 
        out_shape: Shape,                                                | 
        out_strides: Strides,                                            | 
        a_storage: Storage,                                              | 
        a_shape: Shape,                                                  | 
        a_strides: Strides,                                              | 
        reduce_dim: int,                                                 | 
    ) -> None:                                                           | 
        # TODO: Implement for Task 3.1.                                  | 
        for i in prange(len(out)):---------------------------------------| #11
            out_index = np.empty(MAX_DIMS, np.int32)                     | 
            to_index(i, out_shape, out_index)                            | 
            o = index_to_position(out_index, out_strides)                | 
            a = index_to_position(out_index, a_strides)                  | 
            sum = out[o]                                                 | 
            step = a_strides[reduce_dim]                                 | 
            for j in prange(a_shape[reduce_dim]):------------------------| #10
                sum = fn(sum, a_storage[a + j * step])                   | 
            out[o] = sum                                                 | 
        # raise NotImplementedError("Need to implement for Task 3.1")    | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #11, #10).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...
 
+--11 is a parallel loop
   +--10 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--11 (parallel)
   +--10 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--11 (parallel)
   +--10 (serial)


 
Parallel region 0 (loop #11) had 0 loop(s) fused and 1 loop(s) serialized as 
part of the larger parallel loop (#11).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/Users/jiangguangwei/Desktop/MLE/mle-
module-3-GuangweiJiang312/minitorch/fast_ops.py (273) is hoisted out of the 
parallel loop labelled #11 (it will be performed before the loop is executed and
 reused inside the loop):
   Allocation:: out_index = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
MATRIX MULTIPLY
 
================================================================================
 Parallel Accelerator Optimizing:  Function _tensor_matrix_multiply, 
/Users/jiangguangwei/Desktop/MLE/mle-
module-3-GuangweiJiang312/minitorch/fast_ops.py (287)  
================================================================================


Parallel loop listing for  Function _tensor_matrix_multiply, /Users/jiangguangwei/Desktop/MLE/mle-module-3-GuangweiJiang312/minitorch/fast_ops.py (287) 
--------------------------------------------------------------------------------------------|loop #ID
def _tensor_matrix_multiply(                                                                | 
    out: Storage,                                                                           | 
    out_shape: Shape,                                                                       | 
    out_strides: Strides,                                                                   | 
    a_storage: Storage,                                                                     | 
    a_shape: Shape,                                                                         | 
    a_strides: Strides,                                                                     | 
    b_storage: Storage,                                                                     | 
    b_shape: Shape,                                                                         | 
    b_strides: Strides,                                                                     | 
) -> None:                                                                                  | 
    """                                                                                     | 
    NUMBA tensor matrix multiply function.                                                  | 
                                                                                            | 
    Should work for any tensor shapes that broadcast as long as                             | 
                                                                                            | 
    ```                                                                                     | 
    assert a_shape[-1] == b_shape[-2]                                                       | 
    ```                                                                                     | 
                                                                                            | 
    Optimizations:                                                                          | 
                                                                                            | 
    * Outer loop in parallel                                                                | 
    * No index buffers or function calls                                                    | 
    * Inner loop should have no global writes, 1 multiply.                                  | 
                                                                                            | 
                                                                                            | 
    Args:                                                                                   | 
        out (Storage): storage for `out` tensor                                             | 
        out_shape (Shape): shape for `out` tensor                                           | 
        out_strides (Strides): strides for `out` tensor                                     | 
        a_storage (Storage): storage for `a` tensor                                         | 
        a_shape (Shape): shape for `a` tensor                                               | 
        a_strides (Strides): strides for `a` tensor                                         | 
        b_storage (Storage): storage for `b` tensor                                         | 
        b_shape (Shape): shape for `b` tensor                                               | 
        b_strides (Strides): strides for `b` tensor                                         | 
                                                                                            | 
    Returns:                                                                                | 
        None : Fills in `out`                                                               | 
    """                                                                                     | 
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0                                  | 
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0                                  | 
    # TODO: Implement for Task 3.2.                                                         | 
    for N in prange(out_shape[0]):----------------------------------------------------------| #14
        for I in prange(out_shape[1]):------------------------------------------------------| #13
            for J in prange(out_shape[2]):--------------------------------------------------| #12
                o_index = N * out_strides[0] + I * out_strides[-2] + J * out_strides[-1]    | 
                for K in range(a_shape[-1]):                                                | 
                    a_index = N * a_batch_stride + I * a_strides[-2] + K * a_strides[-1]    | 
                    b_index = N * b_batch_stride + K * b_strides[-2] + J * b_strides[-1]    | 
                    out[o_index] += a_storage[a_index] * b_storage[b_index]                 | 
        # raise NotImplementedError("Need to implement for Task 3.2")                       | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #14, #13).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...
 
+--14 is a parallel loop
   +--13 --> rewritten as a serial loop
      +--12 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--14 (parallel)
   +--13 (parallel)
      +--12 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--14 (parallel)
   +--13 (serial)
      +--12 (serial)


 
Parallel region 0 (loop #14) had 0 loop(s) fused and 2 loop(s) serialized as 
part of the larger parallel loop (#14).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None
```

Task 3.4: CUDA Matrix Multiplication

```
Timing summary
Size: 64
    fast: 0.00502
    gpu: 0.00734
Size: 128
    fast: 0.01951
    gpu: 0.01715
Size: 256
    fast: 0.12974
    gpu: 0.05605
Size: 512
    fast: 1.57935
    gpu: 0.26720
Size: 1024
    fast: 12.24200
    gpu: 0.94205
```

Task 3.5: Training

(1) Simple:

​	GPU:

```
gpu --HIDDEN 100 --DATASET simple --RATE 0.05
```



```
Average time is 17.2223699092865 (for 10 epochs)
Average time per epoch 1.72223699092865 (for 10 epochs)
```



```
Epoch  0  loss  5.951921170567859 correct 31
Epoch  1  time  1.3534917831420898
Epoch  2  time  1.4624674320220947
Epoch  3  time  1.3563346862792969
Epoch  4  time  1.3457987308502197
Epoch  5  time  1.3710439205169678
Epoch  6  time  1.344057321548462
Epoch  7  time  1.4391028881072998
Epoch  8  time  1.3635671138763428
Epoch  9  time  1.3942272663116455
Epoch  10  time  1.3836569786071777
Epoch  10  loss  1.5691447454016991 correct 50
Average time is 17.2223699092865 (for 10 epochs)
Average time per epoch 1.72223699092865 (for 10 epochs)
```

​	CPU:

```
cpu --HIDDEN 100 --DATASET simple --RATE 0.05
```



```
Average time is 27.329922676086426 (for 10 epochs)
Average time per epoch 2.7329922676086427 (for 10 epochs)
```



```
Epoch  0  time  26.6065890789032
Epoch  0  loss  5.35140465699975 correct 43
Epoch  1  time  0.06684398651123047
Epoch  2  time  0.06459951400756836
Epoch  3  time  0.0632317066192627
Epoch  4  time  0.0630645751953125
Epoch  5  time  0.13042783737182617
Epoch  6  time  0.06544184684753418
Epoch  7  time  0.07044672966003418
Epoch  8  time  0.06307530403137207
Epoch  9  time  0.07003140449523926
Epoch  10  time  0.06617069244384766
Epoch  10  loss  2.5075737379547003 correct 50
Average time is 27.329922676086426 (for 10 epochs)
Average time per epoch 2.7329922676086427 (for 10 epochs)
```

(2) split:

​	GPU:

```
gpu --HIDDEN 100 --DATASET split --RATE 0.05
```



```
Average time is 131.3513307571411 (for 80 epochs)
Average time per epoch 1.641891634464264 (for 80 epochs)
```



```
Epoch  0  loss  5.112819206998323 correct 34
Epoch  1  time  1.3574199676513672
Epoch  2  time  1.3962600231170654
Epoch  3  time  1.3438389301300049
Epoch  4  time  1.3326702117919922
Epoch  5  time  1.4081974029541016
Epoch  6  time  1.3872840404510498
Epoch  7  time  1.4545836448669434
Epoch  8  time  1.3360710144042969
Epoch  9  time  1.387101173400879
Epoch  10  time  1.3318397998809814
Epoch  10  loss  4.368873459162377 correct 34
Epoch  11  time  1.3548212051391602
Epoch  12  time  1.4186854362487793
Epoch  13  time  1.353877305984497
Epoch  14  time  1.3558459281921387
Epoch  15  time  1.383894681930542
Epoch  16  time  1.4050812721252441
Epoch  17  time  1.997943639755249
Epoch  18  time  2.4028422832489014
Epoch  19  time  2.3684897422790527
Epoch  20  time  2.373248815536499
Epoch  20  loss  9.305652571931365 correct 35
Epoch  21  time  2.3113415241241455
Epoch  22  time  1.6663334369659424
Epoch  23  time  1.4198980331420898
Epoch  24  time  1.3938405513763428
Epoch  25  time  1.352379322052002
Epoch  26  time  1.3403785228729248
Epoch  27  time  1.3569362163543701
Epoch  28  time  1.4138970375061035
Epoch  29  time  1.3345494270324707
Epoch  30  time  1.3589560985565186
Epoch  30  loss  5.09029832807324 correct 42
Epoch  31  time  1.3561997413635254
Epoch  32  time  1.3541505336761475
Epoch  33  time  1.4113998413085938
Epoch  34  time  1.8974237442016602
Epoch  35  time  2.2412619590759277
Epoch  36  time  2.2867679595947266
Epoch  37  time  2.303459644317627
Epoch  38  time  2.3345415592193604
Epoch  39  time  1.957362174987793
Epoch  40  time  1.3259570598602295
Epoch  40  loss  4.578254472646369 correct 38
Epoch  41  time  1.3288743495941162
Epoch  42  time  1.3403894901275635
Epoch  43  time  1.3920416831970215
Epoch  44  time  1.4208133220672607
Epoch  45  time  1.3225584030151367
Epoch  46  time  1.3434820175170898
Epoch  47  time  1.3469555377960205
Epoch  48  time  1.3367643356323242
Epoch  49  time  1.4077718257904053
Epoch  50  time  1.3232812881469727
Epoch  50  loss  4.011326815475327 correct 47
Epoch  51  time  1.3130621910095215
Epoch  52  time  1.3283803462982178
Epoch  53  time  1.324369192123413
Epoch  54  time  1.3844542503356934
Epoch  55  time  1.3378500938415527
Epoch  56  time  1.3267991542816162
Epoch  57  time  1.3577344417572021
Epoch  58  time  1.3325042724609375
Epoch  59  time  1.3607933521270752
Epoch  60  time  1.4128761291503906
Epoch  60  loss  5.189280374439954 correct 44
Epoch  61  time  1.3878123760223389
Epoch  62  time  2.0296988487243652
Epoch  63  time  2.2755095958709717
Epoch  64  time  2.3764827251434326
Epoch  65  time  2.4126853942871094
Epoch  66  time  2.221348285675049
Epoch  67  time  2.2336630821228027
Epoch  68  time  2.3437154293060303
Epoch  69  time  2.2538440227508545
Epoch  70  time  2.397763967514038
Epoch  70  loss  3.657389551459814 correct 45
Epoch  71  time  2.068251848220825
Epoch  72  time  1.3289635181427002
Epoch  73  time  1.335357666015625
Epoch  74  time  1.358245849609375
Epoch  75  time  1.410095453262329
Epoch  76  time  1.3299462795257568
Epoch  77  time  1.327646255493164
Epoch  78  time  1.346930742263794
Epoch  79  time  1.5155878067016602
Epoch  80  time  1.4324290752410889
Epoch  80  loss  1.6971905956530706 correct 50
Average time is 131.3513307571411 (for 80 epochs)
Average time per epoch 1.641891634464264 (for 80 epochs)
```

​	CPU:

```
cpu --HIDDEN 100 --DATASET split --RATE 0.05
```



```
Average time is 28.588314294815063 (for 90 epochs)
Average time per epoch 0.31764793660905627 (for 90 epochs)
```



```
Epoch  0  time  22.29301881790161
Epoch  0  loss  6.768273540542794 correct 33
Epoch  1  time  0.0698843002319336
Epoch  2  time  0.06735801696777344
Epoch  3  time  0.06706428527832031
Epoch  4  time  0.06601452827453613
Epoch  5  time  0.07124686241149902
Epoch  6  time  0.06926894187927246
Epoch  7  time  0.06981921195983887
Epoch  8  time  0.0688178539276123
Epoch  9  time  0.06957507133483887
Epoch  10  time  0.06608247756958008
Epoch  10  loss  3.354159328923782 correct 34
Epoch  11  time  0.07256460189819336
Epoch  12  time  0.0687563419342041
Epoch  13  time  0.06595945358276367
Epoch  14  time  0.08251142501831055
Epoch  15  time  0.07173967361450195
Epoch  16  time  0.06647634506225586
Epoch  17  time  0.06642460823059082
Epoch  18  time  0.06675982475280762
Epoch  19  time  0.07477736473083496
Epoch  20  time  0.06677985191345215
Epoch  20  loss  5.530080315848846 correct 41
Epoch  21  time  0.06464314460754395
Epoch  22  time  0.07094550132751465
Epoch  23  time  0.0657033920288086
Epoch  24  time  0.07918095588684082
Epoch  25  time  0.06790924072265625
Epoch  26  time  0.07059049606323242
Epoch  27  time  0.06595063209533691
Epoch  28  time  0.07340455055236816
Epoch  29  time  0.07987380027770996
Epoch  30  time  0.0712895393371582
Epoch  30  loss  3.7932506606658007 correct 43
Epoch  31  time  0.06423234939575195
Epoch  32  time  0.07338762283325195
Epoch  33  time  0.06615614891052246
Epoch  34  time  0.06455540657043457
Epoch  35  time  0.07080817222595215
Epoch  36  time  0.07213020324707031
Epoch  37  time  0.06503653526306152
Epoch  38  time  0.06463241577148438
Epoch  39  time  0.07059288024902344
Epoch  40  time  0.07994246482849121
Epoch  40  loss  3.761453116811308 correct 49
Epoch  41  time  0.06822085380554199
Epoch  42  time  0.0657198429107666
Epoch  43  time  0.08582425117492676
Epoch  44  time  0.06563234329223633
Epoch  45  time  0.06541895866394043
Epoch  46  time  0.07270288467407227
Epoch  47  time  0.07004833221435547
Epoch  48  time  0.06519818305969238
Epoch  49  time  0.08201241493225098
Epoch  50  time  0.06828522682189941
Epoch  50  loss  2.362799334207385 correct 48
Epoch  51  time  0.07167840003967285
Epoch  52  time  0.06842970848083496
Epoch  53  time  0.06853842735290527
Epoch  54  time  0.06960701942443848
Epoch  55  time  0.0650789737701416
Epoch  56  time  0.0695810317993164
Epoch  57  time  0.06648445129394531
Epoch  58  time  0.08527612686157227
Epoch  59  time  0.06622195243835449
Epoch  60  time  0.07013583183288574
Epoch  60  loss  2.253002124807442 correct 48
Epoch  61  time  0.07979249954223633
Epoch  62  time  0.06514883041381836
Epoch  63  time  0.06698846817016602
Epoch  64  time  0.07606196403503418
Epoch  65  time  0.0650033950805664
Epoch  66  time  0.06666231155395508
Epoch  67  time  0.07105803489685059
Epoch  68  time  0.07540774345397949
Epoch  69  time  0.06465816497802734
Epoch  70  time  0.06554627418518066
Epoch  70  loss  3.494310822475082 correct 49
Epoch  71  time  0.0652925968170166
Epoch  72  time  0.08400774002075195
Epoch  73  time  0.07015609741210938
Epoch  74  time  0.06601667404174805
Epoch  75  time  0.0650017261505127
Epoch  76  time  0.0652000904083252
Epoch  77  time  0.07613301277160645
Epoch  78  time  0.06515860557556152
Epoch  79  time  0.06484746932983398
Epoch  80  time  0.06596994400024414
Epoch  80  loss  2.050515038034241 correct 49
Epoch  81  time  0.07947850227355957
Epoch  82  time  0.0639350414276123
Epoch  83  time  0.07639169692993164
Epoch  84  time  0.06568074226379395
Epoch  85  time  0.06967353820800781
Epoch  86  time  0.07036900520324707
Epoch  87  time  0.08195018768310547
Epoch  88  time  0.06879067420959473
Epoch  89  time  0.06638765335083008
Epoch  90  time  0.06958508491516113
Epoch  90  loss  1.5380026661866275 correct 50
Average time is 28.588314294815063 (for 90 epochs)
Average time per epoch 0.31764793660905627 (for 90 epochs)
```

(3) xor

​	GPU:

```
gpu --HIDDEN 100 --DATASET xor --RATE 0.05
```



```
Average time is 233.22246861457825 (for 150 epochs)
Average time per epoch 1.5548164574305217 (for 150 epochs)
```



```
Epoch  0  loss  5.870454065634552 correct 28
Epoch  1  time  1.3522441387176514
Epoch  2  time  1.4002368450164795
Epoch  3  time  1.3405413627624512
Epoch  4  time  1.3315424919128418
Epoch  5  time  1.3329830169677734
Epoch  6  time  1.3428490161895752
Epoch  7  time  1.434973955154419
Epoch  8  time  1.374145746231079
Epoch  9  time  1.3450067043304443
Epoch  10  time  1.3861258029937744
Epoch  10  loss  6.699285824233026 correct 41
Epoch  11  time  1.362417221069336
Epoch  12  time  1.4066598415374756
Epoch  13  time  1.3652446269989014
Epoch  14  time  1.391019344329834
Epoch  15  time  1.3671197891235352
Epoch  16  time  1.3678874969482422
Epoch  17  time  1.3467822074890137
Epoch  18  time  1.4466536045074463
Epoch  19  time  1.3508310317993164
Epoch  20  time  1.3425297737121582
Epoch  20  loss  4.799489894442299 correct 46
Epoch  21  time  1.39066481590271
Epoch  22  time  1.3496668338775635
Epoch  23  time  2.0418288707733154
Epoch  24  time  2.2902445793151855
Epoch  25  time  2.29771089553833
Epoch  26  time  2.308225631713867
Epoch  27  time  2.3127541542053223
Epoch  28  time  1.9069738388061523
Epoch  29  time  1.3449897766113281
Epoch  30  time  1.360588788986206
Epoch  30  loss  4.158291057811146 correct 40
Epoch  31  time  1.3515138626098633
Epoch  32  time  1.3535137176513672
Epoch  33  time  1.4314939975738525
Epoch  34  time  1.344850778579712
Epoch  35  time  1.368494987487793
Epoch  36  time  1.3459815979003906
Epoch  37  time  1.3736484050750732
Epoch  38  time  1.333749532699585
Epoch  39  time  1.3878755569458008
Epoch  40  time  1.3625617027282715
Epoch  40  loss  3.0961744535651574 correct 46
Epoch  41  time  1.659271478652954
Epoch  42  time  2.136718511581421
Epoch  43  time  2.316922426223755
Epoch  44  time  2.407518148422241
Epoch  45  time  2.297287702560425
Epoch  46  time  2.2966275215148926
Epoch  47  time  1.3391435146331787
Epoch  48  time  1.3480825424194336
Epoch  49  time  1.424210548400879
Epoch  50  time  1.3398840427398682
Epoch  50  loss  2.7572874138751704 correct 44
Epoch  51  time  1.426945686340332
Epoch  52  time  2.0290424823760986
Epoch  53  time  2.305872678756714
Epoch  54  time  2.381768226623535
Epoch  55  time  2.3312292098999023
Epoch  56  time  2.3357603549957275
Epoch  57  time  1.6781537532806396
Epoch  58  time  1.293699026107788
Epoch  59  time  1.3290174007415771
Epoch  60  time  1.391810655593872
Epoch  60  loss  3.5754766589978804 correct 42
Epoch  61  time  1.34065580368042
Epoch  62  time  1.3210461139678955
Epoch  63  time  1.340407133102417
Epoch  64  time  1.334444284439087
Epoch  65  time  1.4183366298675537
Epoch  66  time  1.3324806690216064
Epoch  67  time  1.3827259540557861
Epoch  68  time  1.3713436126708984
Epoch  69  time  1.3412096500396729
Epoch  70  time  1.434450626373291
Epoch  70  loss  2.6668533594682464 correct 45
Epoch  71  time  1.3670692443847656
Epoch  72  time  1.3479149341583252
Epoch  73  time  1.3657441139221191
Epoch  74  time  1.3346223831176758
Epoch  75  time  1.3865439891815186
Epoch  76  time  1.3376655578613281
Epoch  77  time  1.324756383895874
Epoch  78  time  1.3367805480957031
Epoch  79  time  1.3282418251037598
Epoch  80  time  1.3277864456176758
Epoch  80  loss  1.9415432806714648 correct 47
Epoch  81  time  1.3768892288208008
Epoch  82  time  1.318005084991455
Epoch  83  time  1.346846580505371
Epoch  84  time  1.331869125366211
Epoch  85  time  1.5141148567199707
Epoch  86  time  2.2410597801208496
Epoch  87  time  2.3043394088745117
Epoch  88  time  2.291369676589966
Epoch  89  time  2.307926893234253
Epoch  90  time  2.3079636096954346
Epoch  90  loss  2.9521817304280003 correct 45
Epoch  91  time  1.549438714981079
Epoch  92  time  1.35337233543396
Epoch  93  time  1.314452886581421
Epoch  94  time  1.3151347637176514
Epoch  95  time  1.3301818370819092
Epoch  96  time  1.4019434452056885
Epoch  97  time  1.3135788440704346
Epoch  98  time  1.3126590251922607
Epoch  99  time  1.3201682567596436
Epoch  100  time  1.3227128982543945
Epoch  100  loss  1.8059630809875333 correct 48
Epoch  101  time  1.3106825351715088
Epoch  102  time  1.3987743854522705
Epoch  103  time  1.326225757598877
Epoch  104  time  1.3175597190856934
Epoch  105  time  1.3107645511627197
Epoch  106  time  1.3271644115447998
Epoch  107  time  1.3964097499847412
Epoch  108  time  1.327303171157837
Epoch  109  time  1.3422493934631348
Epoch  110  time  1.3618602752685547
Epoch  110  loss  2.707959186660463 correct 48
Epoch  111  time  1.3330812454223633
Epoch  112  time  1.4057340621948242
Epoch  113  time  1.335357666015625
Epoch  114  time  1.3637337684631348
Epoch  115  time  1.3505232334136963
Epoch  116  time  1.3307585716247559
Epoch  117  time  1.386162281036377
Epoch  118  time  1.3253839015960693
Epoch  119  time  1.312997817993164
Epoch  120  time  1.3257842063903809
Epoch  120  loss  3.1314465101503193 correct 48
Epoch  121  time  1.377739667892456
Epoch  122  time  1.3949306011199951
Epoch  123  time  1.3346421718597412
Epoch  124  time  1.3501157760620117
Epoch  125  time  1.3420233726501465
Epoch  126  time  1.3642621040344238
Epoch  127  time  1.3497951030731201
Epoch  128  time  1.3900542259216309
Epoch  129  time  1.3480687141418457
Epoch  130  time  1.3340644836425781
Epoch  130  loss  0.8281657340672242 correct 47
Epoch  131  time  1.3294858932495117
Epoch  132  time  1.3729369640350342
Epoch  133  time  1.4581270217895508
Epoch  134  time  1.3484389781951904
Epoch  135  time  1.561103105545044
Epoch  136  time  2.0899596214294434
Epoch  137  time  2.296189308166504
Epoch  138  time  2.415687322616577
Epoch  139  time  2.2987430095672607
Epoch  140  time  2.2584352493286133
Epoch  140  loss  2.6992143636198493 correct 48
Epoch  141  time  1.444465160369873
Epoch  142  time  1.3288893699645996
Epoch  143  time  1.3924875259399414
Epoch  144  time  1.3503620624542236
Epoch  145  time  1.324997901916504
Epoch  146  time  1.3378572463989258
Epoch  147  time  1.3216991424560547
Epoch  148  time  1.3345022201538086
Epoch  149  time  1.4072833061218262
Epoch  150  time  1.3487720489501953
Epoch  150  loss  1.7481881075619645 correct 50
Average time is 233.22246861457825 (for 150 epochs)
Average time per epoch 1.5548164574305217 (for 150 epochs)
```

​	CPU:

```
cpu --HIDDEN 100 --DATASET xor --RATE 0.05
```

```
Average time is 48.40038275718689 (for 300 epochs)
Average time per epoch 0.16133460919062295 (for 300 epochs)
```

```
Epoch  0  time  27.202035903930664
Epoch  0  loss  8.186013283196115 correct 21
Epoch  1  time  0.07435441017150879
Epoch  2  time  0.06809020042419434
Epoch  3  time  0.07082438468933105
Epoch  4  time  0.06629323959350586
Epoch  5  time  0.0659494400024414
Epoch  6  time  0.06343984603881836
Epoch  7  time  0.0715491771697998
Epoch  8  time  0.06682753562927246
Epoch  9  time  0.06483960151672363
Epoch  10  time  0.0632166862487793
Epoch  10  loss  3.9021113567253707 correct 40
Epoch  11  time  0.0825965404510498
Epoch  12  time  0.07420945167541504
Epoch  13  time  0.06462860107421875
Epoch  14  time  0.07286453247070312
Epoch  15  time  0.06516289710998535
Epoch  16  time  0.0680232048034668
Epoch  17  time  0.07052135467529297
Epoch  18  time  0.06755805015563965
Epoch  19  time  0.06611919403076172
Epoch  20  time  0.06614208221435547
Epoch  20  loss  4.1506538399096 correct 42
Epoch  21  time  0.07308745384216309
Epoch  22  time  0.06766104698181152
Epoch  23  time  0.06393551826477051
Epoch  24  time  0.07274746894836426
Epoch  25  time  0.08331084251403809
Epoch  26  time  0.06547689437866211
Epoch  27  time  0.06505012512207031
Epoch  28  time  0.07054805755615234
Epoch  29  time  0.06400227546691895
Epoch  30  time  0.06548142433166504
Epoch  30  loss  4.1571422007554855 correct 41
Epoch  31  time  0.06817770004272461
Epoch  32  time  0.0637211799621582
Epoch  33  time  0.07091927528381348
Epoch  34  time  0.06643486022949219
Epoch  35  time  0.06567716598510742
Epoch  36  time  0.06746482849121094
Epoch  37  time  0.0740816593170166
Epoch  38  time  0.06524181365966797
Epoch  39  time  0.06404876708984375
Epoch  40  time  0.07930564880371094
Epoch  40  loss  3.4967996704029787 correct 42
Epoch  41  time  0.06845331192016602
Epoch  42  time  0.06537461280822754
Epoch  43  time  0.08086943626403809
Epoch  44  time  0.07058215141296387
Epoch  45  time  0.0672769546508789
Epoch  46  time  0.06983757019042969
Epoch  47  time  0.0686790943145752
Epoch  48  time  0.06621813774108887
Epoch  49  time  0.08598613739013672
Epoch  50  time  0.07110095024108887
Epoch  50  loss  3.260925792985631 correct 42
Epoch  51  time  0.06563425064086914
Epoch  52  time  0.06568408012390137
Epoch  53  time  0.06417655944824219
Epoch  54  time  0.06921577453613281
Epoch  55  time  0.0805661678314209
Epoch  56  time  0.06396150588989258
Epoch  57  time  0.06662392616271973
Epoch  58  time  0.07012033462524414
Epoch  59  time  0.07367134094238281
Epoch  60  time  0.06866216659545898
Epoch  60  loss  6.856275457803777 correct 38
Epoch  61  time  0.07454729080200195
Epoch  62  time  0.06735777854919434
Epoch  63  time  0.06947112083435059
Epoch  64  time  0.06485486030578613
Epoch  65  time  0.06797170639038086
Epoch  66  time  0.06679511070251465
Epoch  67  time  0.07201242446899414
Epoch  68  time  0.06749582290649414
Epoch  69  time  0.08288264274597168
Epoch  70  time  0.0661318302154541
Epoch  70  loss  1.485778550102925 correct 41
Epoch  71  time  0.07039237022399902
Epoch  72  time  0.06511378288269043
Epoch  73  time  0.0699005126953125
Epoch  74  time  0.06781530380249023
Epoch  75  time  0.0656588077545166
Epoch  76  time  0.06996870040893555
Epoch  77  time  0.06435966491699219
Epoch  78  time  0.07070398330688477
Epoch  79  time  0.06608867645263672
Epoch  80  time  0.06881093978881836
Epoch  80  loss  5.074645017491284 correct 42
Epoch  81  time  0.07002449035644531
Epoch  82  time  0.07042169570922852
Epoch  83  time  0.07205915451049805
Epoch  84  time  0.08496212959289551
Epoch  85  time  0.06780529022216797
Epoch  86  time  0.06764960289001465
Epoch  87  time  0.06766533851623535
Epoch  88  time  0.0732569694519043
Epoch  89  time  0.0651388168334961
Epoch  90  time  0.06717681884765625
Epoch  90  loss  4.833606030744569 correct 42
Epoch  91  time  0.07059192657470703
Epoch  92  time  0.06931066513061523
Epoch  93  time  0.06674361228942871
Epoch  94  time  0.06593537330627441
Epoch  95  time  0.06771492958068848
Epoch  96  time  0.07166457176208496
Epoch  97  time  0.06897115707397461
Epoch  98  time  0.09416079521179199
Epoch  99  time  0.06835365295410156
Epoch  100  time  0.07609915733337402
Epoch  100  loss  0.8631732574165969 correct 45
Epoch  101  time  0.06547856330871582
Epoch  102  time  0.06321334838867188
Epoch  103  time  0.06980490684509277
Epoch  104  time  0.07021880149841309
Epoch  105  time  0.06570649147033691
Epoch  106  time  0.0693361759185791
Epoch  107  time  0.09126901626586914
Epoch  108  time  0.06713366508483887
Epoch  109  time  0.06588602066040039
Epoch  110  time  0.07255220413208008
Epoch  110  loss  5.192328244728266 correct 41
Epoch  111  time  0.07309198379516602
Epoch  112  time  0.07523512840270996
Epoch  113  time  0.0756981372833252
Epoch  114  time  0.0719766616821289
Epoch  115  time  0.07031583786010742
Epoch  116  time  0.07020807266235352
Epoch  117  time  0.06829667091369629
Epoch  118  time  0.07147622108459473
Epoch  119  time  0.07368707656860352
Epoch  120  time  0.06565308570861816
Epoch  120  loss  2.7005801895712978 correct 42
Epoch  121  time  0.06849145889282227
Epoch  122  time  0.07066011428833008
Epoch  123  time  0.07053637504577637
Epoch  124  time  0.06654596328735352
Epoch  125  time  0.06732869148254395
Epoch  126  time  0.07818412780761719
Epoch  127  time  0.0823979377746582
Epoch  128  time  0.06963753700256348
Epoch  129  time  0.07048249244689941
Epoch  130  time  0.07495260238647461
Epoch  130  loss  2.2183615478192573 correct 46
Epoch  131  time  0.07514452934265137
Epoch  132  time  0.06741523742675781
Epoch  133  time  0.07407045364379883
Epoch  134  time  0.06661224365234375
Epoch  135  time  0.07525062561035156
Epoch  136  time  0.06745791435241699
Epoch  137  time  0.06636595726013184
Epoch  138  time  0.06878471374511719
Epoch  139  time  0.07012629508972168
Epoch  140  time  0.06709837913513184
Epoch  140  loss  1.3046382989026448 correct 43
Epoch  141  time  0.08264803886413574
Epoch  142  time  0.06481790542602539
Epoch  143  time  0.07062554359436035
Epoch  144  time  0.06444811820983887
Epoch  145  time  0.06854391098022461
Epoch  146  time  0.06744813919067383
Epoch  147  time  0.06411623954772949
Epoch  148  time  0.0704352855682373
Epoch  149  time  0.06945037841796875
Epoch  150  time  0.0654909610748291
Epoch  150  loss  2.1183272640450452 correct 43
Epoch  151  time  0.06992101669311523
Epoch  152  time  0.07082891464233398
Epoch  153  time  0.06586551666259766
Epoch  154  time  0.06453728675842285
Epoch  155  time  0.0641322135925293
Epoch  156  time  0.08908438682556152
Epoch  157  time  0.06969642639160156
Epoch  158  time  0.07070279121398926
Epoch  159  time  0.06589508056640625
Epoch  160  time  0.07241153717041016
Epoch  160  loss  1.9797515755359285 correct 43
Epoch  161  time  0.06966972351074219
Epoch  162  time  0.06493306159973145
Epoch  163  time  0.06939077377319336
Epoch  164  time  0.06696176528930664
Epoch  165  time  0.06876873970031738
Epoch  166  time  0.06593775749206543
Epoch  167  time  0.07037496566772461
Epoch  168  time  0.08716416358947754
Epoch  169  time  0.07968759536743164
Epoch  170  time  0.08098030090332031
Epoch  170  loss  0.33532193107096636 correct 44
Epoch  171  time  0.07401204109191895
Epoch  172  time  0.06914973258972168
Epoch  173  time  0.08031177520751953
Epoch  174  time  0.07248091697692871
Epoch  175  time  0.06431102752685547
Epoch  176  time  0.07005071640014648
Epoch  177  time  0.07413220405578613
Epoch  178  time  0.06853580474853516
Epoch  179  time  0.06565666198730469
Epoch  180  time  0.08089756965637207
Epoch  180  loss  1.91352965112545 correct 43
Epoch  181  time  0.0712442398071289
Epoch  182  time  0.0648195743560791
Epoch  183  time  0.0672459602355957
Epoch  184  time  0.08251237869262695
Epoch  185  time  0.06940793991088867
Epoch  186  time  0.06717157363891602
Epoch  187  time  0.07268333435058594
Epoch  188  time  0.07325625419616699
Epoch  189  time  0.06585574150085449
Epoch  190  time  0.07008647918701172
Epoch  190  loss  2.762445947531726 correct 43
Epoch  191  time  0.06850361824035645
Epoch  192  time  0.06568264961242676
Epoch  193  time  0.07026267051696777
Epoch  194  time  0.07190418243408203
Epoch  195  time  0.06688618659973145
Epoch  196  time  0.06536459922790527
Epoch  197  time  0.06960225105285645
Epoch  198  time  0.07364583015441895
Epoch  199  time  0.07850074768066406
Epoch  200  time  0.06925415992736816
Epoch  200  loss  1.3857612818568492 correct 46
Epoch  201  time  0.07370281219482422
Epoch  202  time  0.07271146774291992
Epoch  203  time  0.06752467155456543
Epoch  204  time  0.06555771827697754
Epoch  205  time  0.07245016098022461
Epoch  206  time  0.07076573371887207
Epoch  207  time  0.06853985786437988
Epoch  208  time  0.06638479232788086
Epoch  209  time  0.06577515602111816
Epoch  210  time  0.06546449661254883
Epoch  210  loss  1.7997449046720622 correct 44
Epoch  211  time  0.07268023490905762
Epoch  212  time  0.06601810455322266
Epoch  213  time  0.08073949813842773
Epoch  214  time  0.06459307670593262
Epoch  215  time  0.07213473320007324
Epoch  216  time  0.06601595878601074
Epoch  217  time  0.06669473648071289
Epoch  218  time  0.06629562377929688
Epoch  219  time  0.06650614738464355
Epoch  220  time  0.06910872459411621
Epoch  220  loss  2.8075816541436307 correct 48
Epoch  221  time  0.06524348258972168
Epoch  222  time  0.06346988677978516
Epoch  223  time  0.06421375274658203
Epoch  224  time  0.07418942451477051
Epoch  225  time  0.06606221199035645
Epoch  226  time  0.06932425498962402
Epoch  227  time  0.07268548011779785
Epoch  228  time  0.08101844787597656
Epoch  229  time  0.063507080078125
Epoch  230  time  0.06850361824035645
Epoch  230  loss  1.0669580415670943 correct 48
Epoch  231  time  0.07060432434082031
Epoch  232  time  0.0650336742401123
Epoch  233  time  0.07014727592468262
Epoch  234  time  0.06751251220703125
Epoch  235  time  0.06381559371948242
Epoch  236  time  0.06555747985839844
Epoch  237  time  0.06976461410522461
Epoch  238  time  0.06651496887207031
Epoch  239  time  0.07112717628479004
Epoch  240  time  0.06755590438842773
Epoch  240  loss  2.4343682044280786 correct 49
Epoch  241  time  0.06541991233825684
Epoch  242  time  0.06982302665710449
Epoch  243  time  0.08843588829040527
Epoch  244  time  0.06410336494445801
Epoch  245  time  0.06553244590759277
Epoch  246  time  0.07193326950073242
Epoch  247  time  0.07482147216796875
Epoch  248  time  0.06812381744384766
Epoch  249  time  0.06698799133300781
Epoch  250  time  0.07140588760375977
Epoch  250  loss  2.1662636368900814 correct 45
Epoch  251  time  0.06987905502319336
Epoch  252  time  0.06452679634094238
Epoch  253  time  0.06510114669799805
Epoch  254  time  0.07741379737854004
Epoch  255  time  0.06937599182128906
Epoch  256  time  0.06880831718444824
Epoch  257  time  0.08353686332702637
Epoch  258  time  0.07610893249511719
Epoch  259  time  0.07253074645996094
Epoch  260  time  0.07518172264099121
Epoch  260  loss  2.449778728160255 correct 49
Epoch  261  time  0.0908210277557373
Epoch  262  time  0.0740973949432373
Epoch  263  time  0.06768488883972168
Epoch  264  time  0.08113646507263184
Epoch  265  time  0.0685577392578125
Epoch  266  time  0.07164549827575684
Epoch  267  time  0.06541585922241211
Epoch  268  time  0.06963419914245605
Epoch  269  time  0.06624102592468262
Epoch  270  time  0.06547307968139648
Epoch  270  loss  0.965857798751809 correct 48
Epoch  271  time  0.0854790210723877
Epoch  272  time  0.07446455955505371
Epoch  273  time  0.06668567657470703
Epoch  274  time  0.06771540641784668
Epoch  275  time  0.07276391983032227
Epoch  276  time  0.07163476943969727
Epoch  277  time  0.07870125770568848
Epoch  278  time  0.1593186855316162
Epoch  279  time  0.06567907333374023
Epoch  280  time  0.09615159034729004
Epoch  280  loss  1.3107337118910851 correct 48
Epoch  281  time  0.06400370597839355
Epoch  282  time  0.06925582885742188
Epoch  283  time  0.06883716583251953
Epoch  284  time  0.08274221420288086
Epoch  285  time  0.06767606735229492
Epoch  286  time  0.06936454772949219
Epoch  287  time  0.06498956680297852
Epoch  288  time  0.06626653671264648
Epoch  289  time  0.06839346885681152
Epoch  290  time  0.08604931831359863
Epoch  290  loss  2.2229764708738586 correct 48
Epoch  291  time  0.07355332374572754
Epoch  292  time  0.06776261329650879
Epoch  293  time  0.06440615653991699
Epoch  294  time  0.06998014450073242
Epoch  295  time  0.06523728370666504
Epoch  296  time  0.06734132766723633
Epoch  297  time  0.06380796432495117
Epoch  298  time  0.08781838417053223
Epoch  299  time  0.07099008560180664
Epoch  300  time  0.09750723838806152
Epoch  300  loss  2.4727719902975838 correct 50
Average time is 48.40038275718689 (for 300 epochs)
Average time per epoch 0.16133460919062295 (for 300 epochs)
```



(4) Big model (Xor)

​	GPU:

```
gpu --HIDDEN 1000 --DATASET xor --RATE 0.05
```

```
Average time per epoch 3.446645374298096 (for 50 epochs)
```



```
Epoch  0  loss  nan correct 30
Epoch  1  time  3.2210776805877686
Epoch  2  time  3.323525905609131
Epoch  3  time  3.1547389030456543
Epoch  4  time  3.1122169494628906
Epoch  5  time  3.222447395324707
Epoch  6  time  3.1306519508361816
Epoch  7  time  3.2122607231140137
Epoch  8  time  3.1515400409698486
Epoch  9  time  3.1679635047912598
Epoch  10  time  5.0065367221832275
Epoch  10  loss  nan correct 30
Average time per epoch 3.888811731338501 (for 10 epochs)
Epoch  11  time  5.205600738525391
Epoch  12  time  3.651594877243042
Epoch  13  time  3.1304996013641357
Epoch  14  time  3.087641716003418
Epoch  15  time  3.1005091667175293
Epoch  16  time  3.1072325706481934
Epoch  17  time  3.119760036468506
Epoch  18  time  3.2104618549346924
Epoch  19  time  3.1645286083221436
Epoch  20  time  3.210355281829834
Epoch  20  loss  nan correct 30
Average time per epoch 3.6438150882720945 (for 20 epochs)
Epoch  21  time  3.131878614425659
Epoch  22  time  3.1261236667633057
Epoch  23  time  3.2279715538024902
Epoch  24  time  3.128322124481201
Epoch  25  time  3.1549010276794434
Epoch  26  time  3.140310764312744
Epoch  27  time  3.141481876373291
Epoch  28  time  3.2294015884399414
Epoch  29  time  3.195484161376953
Epoch  30  time  3.4703140258789062
Epoch  30  loss  nan correct 30
Average time per epoch 3.494083038965861 (for 30 epochs)
Epoch  31  time  5.138761520385742
Epoch  32  time  5.1839001178741455
Epoch  33  time  3.7105135917663574
Epoch  34  time  3.168735980987549
Epoch  35  time  3.186140298843384
Epoch  36  time  3.157824993133545
Epoch  37  time  3.1121914386749268
Epoch  38  time  3.14233660697937
Epoch  39  time  3.1604831218719482
Epoch  40  time  3.1149489879608154
Epoch  40  loss  nan correct 30
Average time per epoch 3.5224581956863403 (for 40 epochs)
Epoch  41  time  3.1049792766571045
Epoch  42  time  3.1136157512664795
Epoch  43  time  3.10756778717041
Epoch  44  time  3.205841541290283
Epoch  45  time  3.1048367023468018
Epoch  46  time  3.163515567779541
Epoch  47  time  3.156094789505005
Epoch  48  time  3.201472282409668
Epoch  49  time  3.1858415603637695
Epoch  50  time  3.0901756286621094
Epoch  50  loss  nan correct 30
Average time per epoch 3.446645374298096 (for 50 epochs)
```

​	CPU:

```
cpu --HIDDEN 1000 --DATASET xor --RATE 0.05
```

```
Average time per epoch 2.269710144996643 (for 50 epochs)
Average time is 113.48550724983215 (for 50 epochs)
```

```
Epoch  0  time  25.498543977737427
Epoch  0  loss  13.815501557895455 correct 34
Epoch  1  time  1.6066124439239502
Epoch  2  time  1.5904741287231445
Epoch  3  time  1.6158740520477295
Epoch  4  time  1.5673127174377441
Epoch  5  time  1.5742466449737549
Epoch  6  time  1.6037712097167969
Epoch  7  time  1.5740587711334229
Epoch  8  time  1.5832464694976807
Epoch  9  time  1.568934440612793
Epoch  10  time  1.5581939220428467
Epoch  10  loss  5.601097018365338 correct 40
Average time per epoch 4.134126877784729 (for 10 epochs)
Epoch  11  time  1.57234787940979
Epoch  12  time  1.5402138233184814
Epoch  13  time  1.548079013824463
Epoch  14  time  1.5820364952087402
Epoch  15  time  1.531482219696045
Epoch  16  time  1.5518057346343994
Epoch  17  time  1.57651948928833
Epoch  18  time  1.556549072265625
Epoch  19  time  1.5712542533874512
Epoch  20  time  1.5520195960998535
Epoch  20  loss  2.6931042124536937 correct 44
Average time per epoch 2.8461788177490233 (for 20 epochs)
Epoch  21  time  4.2611682415008545
Epoch  22  time  8.498952865600586
Epoch  23  time  1.5615453720092773
Epoch  24  time  1.5514187812805176
Epoch  25  time  1.5409059524536133
Epoch  26  time  1.5632286071777344
Epoch  27  time  1.5596132278442383
Epoch  28  time  1.5581188201904297
Epoch  29  time  1.5749988555908203
Epoch  30  time  1.5433156490325928
Epoch  30  loss  9.85313996157889 correct 42
Average time per epoch 2.7378947575887045 (for 30 epochs)
Epoch  31  time  1.549973487854004
Epoch  32  time  1.540698766708374
Epoch  33  time  1.5513620376586914
Epoch  34  time  1.5763039588928223
Epoch  35  time  1.5617961883544922
Epoch  36  time  1.5612328052520752
Epoch  37  time  1.5656383037567139
Epoch  38  time  1.5688257217407227
Epoch  39  time  1.5599422454833984
Epoch  40  time  1.5803196430206299
Epoch  40  loss  2.085957249926969 correct 47
Average time per epoch 2.4438233971595764 (for 40 epochs)
Epoch  41  time  1.5894317626953125
Epoch  42  time  1.5825629234313965
Epoch  43  time  1.578195571899414
Epoch  44  time  1.5619065761566162
Epoch  45  time  1.5723185539245605
Epoch  46  time  1.580730676651001
Epoch  47  time  1.5705647468566895
Epoch  48  time  1.5849580764770508
Epoch  49  time  1.5452055931091309
Epoch  50  time  1.5666968822479248
Epoch  50  loss  1.4580708780711926 correct 50
Average time per epoch 2.269710144996643 (for 50 epochs)
Average time is 113.48550724983215 (for 50 epochs)
```

