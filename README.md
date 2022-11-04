# efficient-ris-aided-urllc

Code implementing the algorithm and the benchmark of the paper "Efficient URLLC with a Reconfigurable Intelligent Surface and Imperfect Device Tracking"

The paper can be found here: 

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
