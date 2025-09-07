# Multi-objective Simulated Annealing (MOSA)

## Algorithm

For each run:

&nbsp;&nbsp;While current temperature **T<sub>i</sub>** > final temperature:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;For each iteration:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Initialize random parameters set **π** with solution **S**

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Generate candidate parameters set **π<sub>c</sub>** with candidate solution **S<sub>c</sub>**: &nbsp;&nbsp;&nbsp; **π<sub>c</sub>** = **π** + *random* **(-step_size, +step_size)**

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; For each objective **S<sub>c<sub>i</sub></sub>**:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **P = 1** if **S<sub>c<sub>i</sub></sub>** < **S<sub>i</sub>**