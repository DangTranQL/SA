# Multi-objective Simulated Annealing (MOSA)

## Algorithm

For each run:

&nbsp;&nbsp;While current temperature **T<sub>i</sub>** > final temperature:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;For each iteration:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Initialize random parameters set **π** with solution **S**

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Generate candidate parameters set **π<sub>c</sub>** with candidate solution **S<sub>c</sub>**: &nbsp;&nbsp;&nbsp; **π<sub>c</sub>** = **π** + *random* **(-step_size, +step_size)**

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;For each objective **S<sub>c<sub>i</sub></sub>**:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **P = 1** if **S<sub>c<sub>i</sub></sub>** < **S<sub>i</sub>**

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **P = e<sup>-ΔS<sub>i</sub>/T</sup>** if **S<sub>c<sub>i</sub></sub>** > **S<sub>i</sub>**

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Equilibrium condition **γ = α *max*(p<sub>i</sub>) + (1-α) ∏ p<sub>i</sub>**

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**S<sub>c</sub>** is accepted if **γ = 1** or **γ** > random number in (0,1)

The algorithm obtains an unpruned Pareto solutions after each run, then all solutions are then pruned to obtain Pareto front