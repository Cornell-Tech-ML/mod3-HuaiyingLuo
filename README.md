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
- Epoch = 490
- Time per epoch = 1.767seconds
- Training log:
```
Epoch  0  loss  3.4642383960351144 correct 46
Epoch  10  loss  2.3278845833380557 correct 49
Epoch  20  loss  0.4651712766371487 correct 50
Epoch  30  loss  0.7545256295023387 correct 49
Epoch  40  loss  0.38729918502821464 correct 49
Epoch  50  loss  0.7292287505254642 correct 50
Epoch  60  loss  0.579246397436999 correct 50
Epoch  70  loss  0.02728924092232261 correct 50
Epoch  80  loss  0.4554630169497637 correct 50
Epoch  90  loss  0.1757181824203757 correct 50
Epoch  100  loss  0.23096190756102103 correct 50
Epoch  110  loss  0.45525324260777 correct 50
Epoch  120  loss  0.028604872999878114 correct 50
Epoch  130  loss  0.29924534374413075 correct 50
Epoch  140  loss  0.3000448448070087 correct 50
Epoch  150  loss  0.6367819988735982 correct 50
Epoch  160  loss  0.0323421870681993 correct 50
Epoch  170  loss  0.3323041901899031 correct 50
Epoch  180  loss  0.06553964594866286 correct 50
Epoch  190  loss  0.09920575219289703 correct 50
Epoch  200  loss  0.09259755941917995 correct 50
Epoch  210  loss  0.3591862976027924 correct 50
Epoch  220  loss  0.242410645221595 correct 50
Epoch  230  loss  0.21316135926988092 correct 50
Epoch  240  loss  0.05983817032748592 correct 50
Epoch  250  loss  0.34602161595952485 correct 50
Epoch  260  loss  0.010057568112297743 correct 50
Epoch  270  loss  0.30427070228470904 correct 50
Epoch  280  loss  0.05136626707554102 correct 50
Epoch  290  loss  0.03269814909039719 correct 50
Epoch  300  loss  0.1392701476622757 correct 50
Epoch  310  loss  0.04531973452194237 correct 50
Epoch  320  loss  0.15319627166003436 correct 50
Epoch  330  loss  0.19953589436601724 correct 50
Epoch  340  loss  0.012127316686109255 correct 50
Epoch  350  loss  0.1588833082192919 correct 50
Epoch  360  loss  0.24189919643330257 correct 50
Epoch  370  loss  0.006061930806990839 correct 50
Epoch  380  loss  0.0006895196378483121 correct 50
Epoch  390  loss  0.14381046749115645 correct 50
Epoch  400  loss  0.17012442626884305 correct 50
Epoch  410  loss  0.03933693990172919 correct 50
Epoch  420  loss  0.013895165671072194 correct 50
Epoch  430  loss  0.09369922507852367 correct 50
Epoch  440  loss  0.22890352490653695 correct 50
Epoch  450  loss  0.05303193768552365 correct 50
Epoch  460  loss  9.881678287902721e-06 correct 50
Epoch  470  loss  0.042230740574841476 correct 50
Epoch  480  loss  0.0908291022392584 correct 50
Epoch  490  loss  0.005245470308977125 correct 50

real	14m25.682s
user	14m15.082s
sys	0m5.895s
```


- Backend = CPU
- Epoch = 490
- Time per epoch = 
- Training log:


### Split Dataset 
- Backend = GPU
- Epoch = 490
- Time per epoch = 1.718 seconds
- Training log:
```
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
```

- Backend = CPU
- Epoch = 490
- Time per epoch = 
- Training log:



### Xor Dataset 
- Backend = GPU
- Epoch = 490
- Time per epoch = 1.739 seconds
- Training log:
```
Epoch  0  loss  7.8198942820439425 correct 22
Epoch  10  loss  5.014634084559113 correct 45
Epoch  20  loss  4.887673865695592 correct 45
Epoch  30  loss  3.9106707699031062 correct 46
Epoch  40  loss  3.57493670214052 correct 48
Epoch  50  loss  2.507456358745772 correct 49
Epoch  60  loss  2.111457221284928 correct 50
Epoch  70  loss  2.993161280284903 correct 49
Epoch  80  loss  1.2071425901711672 correct 49
Epoch  90  loss  0.9660317675073966 correct 50
Epoch  100  loss  1.8335038983987613 correct 50
Epoch  110  loss  0.5383462660125773 correct 48
Epoch  120  loss  2.1849131361076815 correct 50
Epoch  130  loss  0.3444292047195203 correct 50
Epoch  140  loss  0.7235575445728621 correct 50
Epoch  150  loss  0.6735231891257463 correct 50
Epoch  160  loss  1.342505920511953 correct 49
Epoch  170  loss  0.3965424220645676 correct 50
Epoch  180  loss  1.213552020206972 correct 50
Epoch  190  loss  0.38866633827859604 correct 50
Epoch  200  loss  0.8943112498577672 correct 50
Epoch  210  loss  0.2532878646569045 correct 50
Epoch  220  loss  0.4445670027439339 correct 50
Epoch  230  loss  0.6509120443138124 correct 50
Epoch  240  loss  0.901913320535603 correct 50
Epoch  250  loss  0.6439373566951871 correct 50
Epoch  260  loss  0.3588706519383154 correct 50
Epoch  270  loss  0.709448932296606 correct 50
Epoch  280  loss  0.25168863459072355 correct 50
Epoch  290  loss  0.5135055237980736 correct 50
Epoch  300  loss  0.5751478321738481 correct 50
Epoch  310  loss  0.23131963382565734 correct 50
Epoch  320  loss  0.37115732356186104 correct 50
Epoch  330  loss  0.8324795537017685 correct 50
Epoch  340  loss  0.6730471132480412 correct 50
Epoch  350  loss  0.2773811159180983 correct 50
Epoch  360  loss  0.2887648076391624 correct 50
Epoch  370  loss  0.5209435749283122 correct 50
Epoch  380  loss  0.37890858816516676 correct 50
Epoch  390  loss  0.40144748439805833 correct 50
Epoch  400  loss  0.1605697080317301 correct 50
Epoch  410  loss  0.09974888919592005 correct 50
Epoch  420  loss  0.09490225384911632 correct 50
Epoch  430  loss  0.10670741813960223 correct 50
Epoch  440  loss  0.3472029884099912 correct 50
Epoch  450  loss  0.3003881438957835 correct 50
Epoch  460  loss  0.35894166005378686 correct 50
Epoch  470  loss  0.10725837297910024 correct 50
Epoch  480  loss  0.2209247813101442 correct 50
Epoch  490  loss  0.3822252201865803 correct 50

real	14m12.153s
user	14m1.822s
sys	0m5.636s
```


- Backend = CPU
- Epoch = 490
- Time per epoch = 
- Training log:


### Bigger Model Result
- Hidden Layers = 200
- Learning Rate = 0.05


### Simple Dataset 
- Backend = CPU
- Epoch = 490
- Time per epoch = 
- Training log:


