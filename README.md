This is the source code that reimplements an agent-based model for nonviolence action described in the following article:
https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0269976

To run the model, run the following Pytho script with three parameters:
nonviolence_revolution.py <n> <y/n> <s>
where 
<n> is replaced with the number of simulations you want in the batch run,
<y/n> indicates whether simulation results will be saved to a database file batch_run/baseline_n_s.db. A y means the simulation results will be saved. 
<s> is a suffix of the database filename. 

For example, the following command line
python nonviolence_revolution.py 10 y test
will run the model 10 times in a batch run and save the simulation results in the database file at batch_run/baseline_10_test.db

The other Jupeter notebook files show the tests conducted to compare the behaviors of the reimplemented model and the original model.
