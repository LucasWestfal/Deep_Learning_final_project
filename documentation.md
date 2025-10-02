# Notes regarding the development of the project

## Good things to analize for the project


- [x] actual database on the chemical reaction over time

- [ ] time comparison of running time between classical methods and PINN

- [ ] analysis on the discrepancies between found solutions
influence of different loss parameters on the setting

- [ ] in depth analysis of real life interpretation of the errors of the solution

- [ ] comparison between long time stability of the methods
application of takens theorem on how to collect data on the run

- [ ] discussion about potential gains of using PINN instead of finite differences

    - [ ] training all parameters under a single optimization problem

    - [ ] viabilizing implementation under very little data

    - [ ] simplicity on the placement of sensors
    
    - [ ] flexibility of mesh free methods

- [ ] Comparing optimal hyperparameters of the PINN loss between different scenarios: the hyperparameters are stable between different settings, when only changing the grid?

- [ ] Does discontinuity ruins everything?

- [ ] Analizing viability of modelling non differential curves