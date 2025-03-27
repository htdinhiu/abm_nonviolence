This is the source code that reimplements an agent-based model for nonviolence action described in the following article:
https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0269976

To run the model, run the following Pytho script with three parameters:

nonviolence_revolution.py n <y/n> s

where 

n is replaced with the number of simulations you want in the batch run,

y indicates that the simulation results will be saved to a database file batch_run/baseline_n_s.db. Replace 'y' with 'n' if you don't want to save the database file. 

s is a suffix of the database filename. 

Example 1: The following command line will run the model 10 times in a batch run and save the simulation results in the database file at batch_run/baseline_10_test.db:

python nonviolence_revolution.py 10 y test

Example 2: The following command line will run the model 5 times in a batch run and but doesn't save the simulation results in a database file:

python nonviolence_revolution.py 5 n 


The other Jupeter notebook files show the tests conducted to compare the behaviors of the reimplemented model and the original model.
