# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

For Task 3.1 debugging:

```
python project/parallel_check.py
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

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py


## Task 3.5: Training

### Small Model Result
- Hidden Layers = 100
- Learning Rate = 0.05

### Simple Dataset 
- Backend = GPU
- Time per epoch = 
- Image


- Backend = CPU
- Time per epoch =  
- Image


### Split Dataset 
- Backend = GPU
- Epoch = 490
- Time per epoch = 1.718 seconds
- Training log:
Epoch  0  loss  6.4923732220058765 correct 35
Epoch  10  loss  6.833289866257934 correct 35
Epoch  20  loss  4.551905340255043 correct 40
Epoch  30  loss  4.464068605774136 correct 42
Epoch  40  loss  4.6481584437624806 correct 45
Epoch  50  loss  5.445229862445415 correct 46
Epoch  60  loss  2.9948832239563945 correct 45
Epoch  70  loss  1.7107878357656157 correct 45
Epoch  80  loss  2.334736922224883 correct 48
Epoch  90  loss  1.5537233306902436 correct 48
Epoch  100  loss  3.264140504771218 correct 44
Epoch  110  loss  1.6189889035510612 correct 48
Epoch  120  loss  1.8603073204677638 correct 48
Epoch  130  loss  2.388516165517773 correct 49
Epoch  140  loss  1.0379393496685503 correct 49
Epoch  150  loss  1.6614815118345843 correct 48
Epoch  160  loss  1.7764041193835758 correct 48
Epoch  170  loss  0.6931186248193987 correct 48
Epoch  180  loss  2.146838038308416 correct 49
Epoch  190  loss  1.3302512760277645 correct 49
Epoch  200  loss  0.7911557370333618 correct 45
Epoch  210  loss  0.5033239413433486 correct 49
Epoch  220  loss  1.5566061253103756 correct 48
Epoch  230  loss  1.278374385807387 correct 49
Epoch  240  loss  1.5135514046331537 correct 48
Epoch  250  loss  1.5499531639643842 correct 50
Epoch  260  loss  0.6501698414641939 correct 49
Epoch  270  loss  0.48870824186011885 correct 48
Epoch  280  loss  0.2946488917901418 correct 50
Epoch  290  loss  0.22807708129995932 correct 49
Epoch  300  loss  1.752739454759899 correct 44
Epoch  310  loss  0.7207169013350367 correct 50
Epoch  320  loss  1.0779026808646275 correct 49
Epoch  330  loss  0.8950473379325112 correct 48
Epoch  340  loss  0.6962720973097427 correct 49
Epoch  350  loss  0.21309185712884107 correct 48
Epoch  360  loss  0.365457808513853 correct 50
Epoch  370  loss  0.5876864668827544 correct 48
Epoch  380  loss  2.9878930751268014 correct 48
Epoch  390  loss  0.45287246958731375 correct 49
Epoch  400  loss  0.7757047664479783 correct 50
Epoch  410  loss  1.0824509958551283 correct 48
Epoch  420  loss  0.30596346439142375 correct 50
Epoch  430  loss  0.9161634876550329 correct 49
Epoch  440  loss  2.3209857654754944 correct 46
Epoch  450  loss  0.6809610536774144 correct 48
Epoch  460  loss  0.12695130194178633 correct 48
Epoch  470  loss  0.6797640399646854 correct 50
Epoch  480  loss  0.3143940320005277 correct 48
Epoch  490  loss  0.8152517616701986 correct 50

real	14m1.765s
user	13m51.879s
sys	0m5.488s



- Backend = GPU
- Epoch = 490
- Time per epoch = 1.718 seconds
- Training log:



### Xor Dataset 
- Backend = GPU
- Time per epoch = 
- Image


- Backend = CPU
- Time per epoch =  
- Image


### Bigger Model Result
- Hidden Layers = 200
- Learning Rate = 0.05


### Simple Dataset 
- Backend = GPU
- Time per epoch = 
- Image


