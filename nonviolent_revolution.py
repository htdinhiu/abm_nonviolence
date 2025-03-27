#!/usr/bin/env python
# coding: utf-8

# ## Agent-based Model Framework for Nonviolent Protests
# This notebook is used to show experiments using the framework to model nonviolent protests. The mesa python library is used for its general-use agent-based modeling resources and implements the same rules and logic from Chenoweth et al (https://doi.org/10.1371/journal.pone.0269976). In this simulation, agents are split into four types: Citizens, Activists, Police, and Pillars. 
# 
# Activists, of which there are few, roam the simulation grid and start protesting when feasible. Two additional movement strategies are also available which tailors Activists' movement towards Pillars. 
# 
# Citizens, which also move at random, upon seeing protests may decide to become an active nonviolence member, if they find the benefit to outweigh the cost, given their living situation. Nonviolence members may also start to protest if they are in the proximity of an existing protestor(s), Activist or NV Citizen, with additional protestors increasing the peer pressure to do so. 
# 
# In response, Police agents, moving at random, search for activists and nonviolent citizens. If an Activist is within a defined proximity, they will be targeted if either 1) they are currently protesting or 2) with some probability. Nonviolent citizens may also be targeted with a much smaller probability. When an agent is targeted, they will be jailed for a defined number of simulation time steps, or (with a very small chance) killed entirely. Police agents may also defect from the regime when in proximity of protesting agents. Each police agent is given a value that weighs their alignment to the regime; if the ratio of protestors to all the neighboring agents in their proximity outweighs this value, they will realign and stop policing entirely.
# 
# Pillar agents are the main target of the simulation, representing regime pillars of economic or political value. A small number of these agents are scattered throughout the simulation grid, and do not act, but follow the same defection rule as Police agents.
# 
# The simulation ends if any one of three situations occurs: 1) a threshold of Pillar agents are able to be defected via protest, 2) the number of alive activists is reduced enough where no new protests can be started, or 3) all 200 time steps allotted to the simulation pass. Case 1 is defined as a Nonviolence victory, case 2 is defined as a regime victory, and case 3 is undefined, as either case 1 or case 2 could later occur with more time steps.

# In[3]:


#To run this script, type
#python3 nonviolent_revolution.py <n> y <s>
#where <n> is replaced with the number of simulations you want in the batch run,
#y indicates that simulation results will be saved to a database file batch_run/baseline_n_s.db. 
#<s> is a suffix of the database filename. 
#This simulation is of the baseline model.

import warnings
import os
import sys

warnings.filterwarnings("ignore", category=UserWarning, message=".*JupyterViz: `use_state` found within a nested function.*")
warnings.filterwarnings("ignore", module="solara.*")

import math
import time
startTime = time.time()

from agent_based import *
from IPython.display import Markdown as md
import numpy as np
import dill
import os
import scipy.stats
import sys
import datetime
#get_ipython().run_line_magic('matplotlib', 'inline')


if __name__ == "__main__":
    startTime1 = time.time()
    print("Time taken to import = " + str(startTime1 - startTime) + " seconds")

    #set number of iterations in a batch run
    n = 1000 
    if len(sys.argv)>1:
        n = int(sys.argv[1])

    saveDb = False
    if len(sys.argv)>2:
        saveDb = 'y' in sys.argv[2].lower()

    suffix = ""
    if len(sys.argv)>3:
        suffix = sys.argv[3]
        
    displayMode = ["singleRunVisualize", "batchRun", "loadSession", "loadSensAnalysis"][1]

    override_mesa_shapes() #built-in mesa draw doesn't have shapes - this overwrites with custom shape-including one

    # can reinforcement learning after have cost for parameters and which affect wins most
    # can game theory with temporal approach (series of protests)
        # blue team approach (future) how can we get them to change their policy to something more expensive?
        # raises question of does red team adjust their parameters based on protests for later runs? ie 100 then 100 then 100 or 300 at once
        # so then blue team changes strategy as a result of something 


    # ## Define the 3 types of run scenarios for the model
    # ### Single Run Visualize
    # Enabling this option allows the user to perform a single simulation run and easily visualize the interaction between agents. "Active" agents are shown in less muted colors. Activists are red, or light red when inactive or jailed (using a smaller marker when jailed). Nonviolent citizens are cyan, or light cyan when inactive and disappear entirely when jailed. Both activists and citizens appear as circles. Police appear as black triangles, turning gray when defected. Pillars are shown as a bright pink star that also fades when defected.
    # ### Batch Run
    # For analysis of behavior over multiple runs, the simulation is able to be set up and ran for a given number of times with data recorded within each step of each simulation. Individual run visualization is hidden in this case.
    # ### Load Batch Data
    # Batch run data is automatically stored to disk upon completion. To save on running time-heavy simulations each time, this opts to load the saved data from the last batch run of simulations. 

    # In[4]:


    def agent_portrayal(agent):
        # https://matplotlib.org/stable/gallery/color/named_colors.html
        # https://matplotlib.org/stable/api/markers_api.html
        if agent.is_Citizen:
            #color = "black" if agent.inProtest else "white" if agent.jailTerm else "lightgray"
            color = "cyan" if agent.inProtest else "white" if agent.jailTerm else "lightcyan"
            marker = "o"
        elif agent.is_Activist:
            #color = "red" if agent.inProtest else "white" if agent.jailTerm else "pink"
            color = "red" if agent.inProtest else "lightpink" if agent.jailTerm else "lightpink"
            marker = '.' if agent.jailTerm else 'o'
        elif agent.is_Police:
            #color = "lightblue" if agent.defect else "blue"
            color = "lightgray" if agent.defect else "black"
            marker = "v"
        elif agent.is_Pillar:
            #color = "lightgreen" if agent.defect else "green"
            color = "thistle" if agent.defect else "fuchsia"
            marker = "*"
            
        return {
            "color": color,
            "size": 30,
            "marker": marker
        }

    model_params = {   # values from S1 table, chenoweth initial optimized values
        'MaxSteps':200, 'PctCitizen':70, 'PctActivist':0.8, 'PctPolice':4, 'PctPillar':1.53, 'MaxJailTerm':10,
        'StartLegitimacy':0.56, 'PctFindNV':40, 'PctTargetNV':25, 'PctKillNV':10, 'BackfireCoeff':0.99,
        'valueF':0.0706, 'ProtestCycle':8.9, 'ProtestDuration':1, 'nNVthreshold':1, 'DefectThresh':0.05,
        'PeerPressureNum':3.46, 'DelayStartMax':5, 'NVSuccessPct':10, 'PillarProxStrat':0,
        'DefectThreshStDev':0.01, 'ActivistSearchVision':10, 'PctFickle':100, 'PctImmedProtest':5,
        'dataCollectorType':displayMode
    }

    # overrides model_params, put sliders here
    param_visual_overrides = {
        "NVSuccessPct":{ "type": "SliderFloat", "value": model_params['NVSuccessPct'], "label": "NV Win % (Pillars)", "min": 5, "max": 50, "step": 1}
    }


    if displayMode == "singleRunVisualize":
        for key in param_visual_overrides.keys():
            model_params[key] = param_visual_overrides[key]
        page = JupyterViz(
            NVResistanceModel,
            model_params,
            measures = [
                {
                #"citizens":"cyan",
                "activists":"red",
                "police":"black", 
                "pillars":"fuchsia", 
                "protest":"cyan"},
                make_text(lambda x: f"### Model Seed: {x._seed}"),
            ],
            name = "",
            agent_portrayal = agent_portrayal,
            play_interval=0,
        )
    elif displayMode == "batchRun":
        numMeasures = 4

        today = datetime.datetime.now().strftime("%Y-%m-%d")
        # sends back parameter set and records for the datacollector used
        startTime2 = time.time()
        print("Time elapsed before batch_run starts = " + str(startTime2 -  startTime) + " seconds")
        results = batchrunner.batch_run(
            NVResistanceModel,
            model_params,
            number_processes = None, # use as many cores as we can
            iterations = n,
            max_steps = model_params["MaxSteps"],
            display_progress = True
        )
        # filter down to the results we want
        #measures = ['RunId', 'iteration', 'Step'] + list(results[0].keys())[-numMeasures:]
        #page = [{key: x[key] for key in measures} for x in results]
        #result = sys.argv[2] #input(f'Save in {today}_simulation/multi_run_data_{sys.argv[1]}.db? y/N: ')

        if saveDb:
            #os.makedirs(f'{today}_simulation', exist_ok=True)
            dill.dump_session(f'batch_data/baseline_{n}_{suffix}.db')

        page = results[0]#"Complete"
        
    elif displayMode == "loadSession":

        today = datetime.datetime.now().strftime("%Y-%m-%d")
        dill.load_session(f'batch_data/baseline_{n}_{suffix}.db')
        page = "Loaded Runs"
        
    page


 
    # In[2]:


    import dill


    # In[21]:

    f = open(f"batch_data/baseline_{n}_{suffix}.txt", "w")
    
    #record SLURM_JOB_ID if run on a supercomputer
    job_id = os.environ.get('SLURM_JOB_ID')
    if job_id:
        f.write(f"SLURM JOB ID:  {job_id} \n")

    if saveDb:
        with open(f'batch_data/baseline_{n}_{suffix}.db', 'rb') as file:
            dill.load_session(file)

            nvVictories = 0

            for run in results:
                nvVictories = (nvVictories + 1) if run['victory'] == 2 else nvVictories
            
            nvVictoriesAvg = nvVictories/len(results)

            print(f"Number of NV Victories: " + str(nvVictories))
            print(f"Average of NVVictories per {n} runs: " + str(nvVictoriesAvg))

            f.write("Number of NV Victories: " + str(nvVictories) + "\n")   
            f.write(f"Average of NVVictories per {n} runs: " + str(nvVictoriesAvg) + "\n")
    
    endTime = time.time()
    timeTaken = round(endTime - startTime)        
    hours = math.floor(timeTaken/3600)
    minutes = math.floor((timeTaken - hours*3600)/60)
    seconds = timeTaken - hours*3600 - minutes*60
    f.write(f"Time taken: {timeTaken} seconds = {hours}h {minutes}m {seconds}s\n")    
    print(f"Total time taken: {timeTaken} seconds = {hours}h {minutes}m {seconds}s")





    # In[ ]:




