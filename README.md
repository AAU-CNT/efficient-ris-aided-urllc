# efficient-ris-aided-urllc

Code implementing the algorithm and the benchmark of the paper "Efficient URLLC with a Reconfigurable Intelligent Surface and Imperfect Device Tracking"

## Abstract
> The use of Reconfigurable Intelligent Surface (RIS) technology to extend coverage and allow for better control of the wireless environment has been proposed in several use cases, including Ultra-Reliable Low-Latency Communications (URLLC) communications. However, the extremely challenging latency constraint makes explicit channel estimation difficult, so positioning information is often used to configure the RIS and illuminate the receiver device. In this work, we analyze the effect of imperfections in the positioning information on the reliability, deriving an upper bound to the outage probability. We then use this bound to perform power control, efficiently finding the minimum power that respects the URLLC constraints under positioning uncertainty. The optimization is conservative, so that all points respect the URLLC constraints, and the bound is relatively tight, with an optimality gap between 1.5 and 4.5 dB.

The paper is submitted to ICC 2023. A preprint version can be found here: 

The main results are obtainable by running
```
test_grid.py
```
to obtain the heatmap, while
```
test_theta.py
```
for the test varying the elevation angle.
Note that the scripts should be run with the flag ```-r``` to save the results. 

Other flags can be used to change the standard parameters, check ```environment.py``` for the possible parameters. 

Visualization of the already presented results can be obtained running ```visual_results.py```. 
Also in this case flag ```-r``` print the results, in both .jpg and .tex format, in a ```plots``` directory.
Check and edit ```scenario.common.standard_output_dir``` to change the default folder  
