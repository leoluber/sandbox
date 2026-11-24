# leo's notes on the project

a set of ideas, notes, comments and unsorted criteria and requests to be integrated into the other relevant`.md` files.

## gala interface
for no we dont need this step. the pipeline stops at the first energy spectrum and no convolution with an optical kernel is performed.
later this is added as an addidtional layer but for now it is sufficient to predict or inverse the distribution over transition energies.


## population signatures
in this first part of the project we will severely restrict the possible morphology distributions to constrain the problem. this includes
1) no rods/wires. isotropic populations or nanoplatelets where one dimension is strongly confined only (this is still quite impressive if it works since rods can be ruled out in a lot of circumstances from other criteria)
2) the standard deviation of each dimension should be bounded to some relative values of the mean such as 5%-20% (can be adjusted if required)
3) each dimension is limited to 2-50 unit cells for now



## deliverables
1) implement the forward pass for a few reasonable single populations defined by some library
2) solve the inverse problem for these library populations
3) generate new populations and test the reconstruction
4) use this for a hyperparameter optimization of the problem
5) solve 2 and 3 population systems


## additional deliverables
1) design a strong loss function
2) make all methods very modular and general so the simple project can later grow to a larger solver library
3) test a variety of new and popular solver methods and approaches
4) keep visualization steps a elementary part of the project