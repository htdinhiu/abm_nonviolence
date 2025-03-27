import mesa, pandas, enum, numpy as np
from matplotlib import pyplot as plt
from mesa.experimental import JupyterViz, make_text
from mesa import batchrunner

# DYNAMIC METHOD - call override_mesa_shapes to replace the function with this one
# overwrite in matplotlib.py in venv /lib/python../site-packages/mesa/experimental/components to add shapes
#frame_count = 0
def _draw_grid_override(space, space_ax, agent_portrayal):
    def portray(g):
        x = [[]]
        y = [[]]
        s = [[]]  # size
        c = [[]]  # color
        ms = ['o'] # list of markers, aligned to each sublist of coords, default o
        for i in range(g.width):
            for j in range(g.height):
                content = g._grid[i][j]
                if not content:
                    continue
                if not hasattr(content, "__iter__"):
                    # Is a single grid
                    content = [content]
                for agent in content:
                    data = agent_portrayal(agent)
                    if "marker" in data:
                        if data["marker"] not in ms:    # new marker - add groupings
                            ms.append(data["marker"])
                            x.append([])
                            y.append([])
                            s.append([])
                            c.append([])
                        group_idx = ms.index(data["marker"])
                    else:
                        group_idx = 0
                    x[group_idx].append(i)
                    y[group_idx].append(j)
                    if "size" in data:
                        s[group_idx].append(data["size"])
                    if "color" in data:
                        c[group_idx].append(data["color"])

        outs = []
        for m_index, marker in enumerate(ms):
            if len(x[m_index]) == 0:
                continue    # ignore the default if it wasn't used

            out = {"x": x[m_index], "y": y[m_index]}
            if len(s[m_index]) > 0:
                out["s"] = s[m_index]
            if len(c[m_index]) > 0:
                out["c"] = c[m_index]
            out["marker"] = marker
            outs.append(out)
        return outs

    space_ax.set_xlim(-1, space.width)
    space_ax.set_ylim(-1, space.height)

    space_ax.tick_params(left = False, right = False , labelleft = False, labelbottom = False, bottom = False) 
    space_ax.set_xticks([x + 0.5 for x in range(space.width)])
    space_ax.set_yticks([x + 0.5 for x in range(space.height)])
    space_ax.get_figure().tight_layout()
    space_ax.get_figure().set_figwidth(space_ax.get_figure().get_size_inches()[1])
    space_ax.grid(alpha = 0.15)
    for p in portray(space):
        space_ax.scatter(**p)

#    plt.savefig(f"figure/frame_{frame_count}.png") #only need this for animation
#    frame_count += 1

def override_mesa_shapes():
    mesa.experimental.components.matplotlib._draw_grid = _draw_grid_override


ATypes = enum.Enum('ATypes', ['CITIZEN', 'ACTIVIST', 'POLICE', 'PILLAR'])



class NVResistanceModel(mesa.Model):
    
    def __init__(self, MaxSteps = 200, SizeX = 40, SizeY = 40, IsTorus = True,
                 PctCitizen = 70, PctActivist = 0.8, PctPolice = 4, PctPillar = 1.53,
                 AgentVision = 4, MaxJailTerm = 10, StartLegitimacy = 0.6,
                 PctFindNV = 40, PctTargetNV = 25, PctKillNV = 10, BackfireCoeff = 0.99,
                 valueF = 0.0706, ProtestCycle = 8.9, ProtestDuration = 1, nNVthreshold = 1,
                 DefectThresh = 0.05, PeerPressureNum = 3.46, DelayStartMax = 1,
                 NVSuccessPct = 10, PillarProxStrat = 0, DefectThreshStDev = 0.01,
                 ActivistSearchVision = 10, ReorderAgents = 1, PctFickle = 100, PctImmedProtest = 5,
                 dataCollectorType = "singleRunVisualize", seed = None
                 ):
        super().__init__()
        self.grid = mesa.space.SingleGrid(SizeX, SizeY, IsTorus)
        self.schedule = mesa.time.RandomActivation(self)


        if dataCollectorType == "singleRunVisualize":
            self.datacollector = mesa.DataCollector(model_reporters = {
                "citizens": lambda _: len([x for x in self.schedule.agents if (x.is_Citizen and not x.jailTerm)]),
                "activists": lambda _: len([x for x in self.schedule.agents if (x.is_Activist and not x.jailTerm)]),
                "police": lambda _: len([x for x in self.schedule.agents if (x.is_Police and not x.defect)]),
                "pillars": lambda _: len([x for x in self.schedule.agents if (x.is_Pillar and not x.defect)]),
                "protest": lambda _: len([x for x in self.schedule.agents if (x.is_Citizen and x.activeNV)])
            }, agent_reporters = None, tables = None)

        elif dataCollectorType == "batchRun":
            self.datacollector = mesa.DataCollector(model_reporters = {

                "citizenPlot": "citizenPlot",
                "activistPlot": "activistPlot",
                "policePlot": "policePlot",
                "pillarPlot": "pillarPlot",
                "protestPlot": "protestPlot",

                "seed": "_seed",
                "victory": "victory",
                "Max_InProtestPct": "Max_InProtestPct",
                "Max_ActualInProtestPct": "Max_ActualInProtestPct",
                "Max_NVCitizens": "Max_NVCitizens"
            }, agent_reporters = None, tables = None)

        self.const = {
            "MaxSteps": round(MaxSteps),
            "delayStartMax": round(DelayStartMax),
            "vision": round(AgentVision),
            "activistVision": round(ActivistSearchVision),
            "pillarproxstrat": PillarProxStrat,
            "ficklePct": PctFickle/100,
            "PctFindNVResistor": PctFindNV/100,
            "PctTargetNV": PctTargetNV/100,
            "PctKillNV": PctKillNV/100,
            "NVSuccessPct": NVSuccessPct/100,  # percentage of pillars defected needed for NV to win, "Win Defect Percentage"
            "BackfireCoeff": BackfireCoeff,
            "MaxJailTerm": MaxJailTerm,
            "nNVthreshold": round(nNVthreshold),
            "protestCycle": round(ProtestCycle),
            "protestDuration": round(ProtestDuration),
            "PctImmedProtest": PctImmedProtest/100,
            "joinNVThresh": valueF,    #newly added "fixed threshold" to reflect historical data
            "defectThreshold": DefectThresh,
            "defectStDev": DefectThreshStDev,
            "peerPressureNum": round(PeerPressureNum)
            # more
        }

        self.citizenPlot = np.zeros(self.const["MaxSteps"])
        self.activistPlot = np.zeros(self.const["MaxSteps"])
        self.policePlot = np.zeros(self.const["MaxSteps"])
        self.pillarPlot = np.zeros(self.const["MaxSteps"])
        self.protestPlot = np.zeros(self.const["MaxSteps"])


        # initialize params of the simulation
        self.governmentLegitimacy = StartLegitimacy

        # characterize and place agents
        NCitizen = round(SizeX*SizeY*PctCitizen/100)
        NActivist = round(SizeX*SizeY*PctActivist/100)
        NPolice = round(SizeX*SizeY*PctPolice/100) 
        NPillar = round(SizeX*SizeY*PctPillar/100)
        TotalAgents = NCitizen + NActivist + NPolice + NPillar
        TypeLUT = [ATypes.CITIZEN]*NCitizen + [ATypes.ACTIVIST]*NActivist + [ATypes.POLICE]*NPolice + [ATypes.PILLAR]*NPillar
        for id, type in enumerate(TypeLUT):
            agent = NVagent(id, type, self)
            self.grid.move_to_empty(agent)
            self.schedule.add(agent)   
        
        # initialize vars for simulation metrics
        self.Max_InProtestPct = 0
        self.Max_ActualInProtestPct = 0
        self.Max_NVCitizens = 0
        self.victory = 0

        self.running = True           
    #__init__

    def step(self):
        self.protestCount = 0
        self.schedule.step()

        # custom data collector for batch runs
        for a in self.schedule.agents:
            if a.is_Citizen and not a.jailTerm:
                self.citizenPlot[self.schedule.steps - 1] += 1
            if a.is_Activist and not a.jailTerm:
                self.activistPlot[self.schedule.steps - 1] += 1
            if a.is_Police and not a.defect:
                self.policePlot[self.schedule.steps - 1] += 1
            if a.is_Pillar and not a.defect:
                self.pillarPlot[self.schedule.steps - 1] += 1
            if a.is_Citizen and a.activeNV:
                self.protestPlot[self.schedule.steps - 1] += 1

        #if self.saveHistory:
            #pass
            # copy agents to list - TODO mesa_replay package? or is same seed sufficient

        aliveAgents = [x for x in self.schedule.agents if x.is_alive]
        self.Max_InProtestPct = max( len([x for x in aliveAgents if x.inProtest]) / len(aliveAgents), self.Max_InProtestPct)
        self.Max_ActualInProtestPct = max(self.protestCount/len(aliveAgents), self.Max_ActualInProtestPct)
        self.Max_NVCitizens = max(len([x for x in aliveAgents if x.is_Citizen and x.activeNV]), self.Max_NVCitizens)


        # check for victory conditions - enough pillars flipped, or regime killed all the activists/activeNV needed to start more joining
        pillarAgents = [x for x in self.schedule.agents if x.is_Pillar]
        if (len([x for x in pillarAgents if x.defect])/len(pillarAgents)) > self.const["NVSuccessPct"]:
            self.victory = 2
        aliveNV = [x for x in self.schedule.agents if x.is_alive and (x.is_Activist or (x.is_Citizen and x.activeNV)) ]
        if len(aliveNV) <= self.const["nNVthreshold"]:
            self.victory = 3

        self.datacollector.collect(self)

        # ran out of time or somebody won
        if (self.schedule.steps >= self.const["MaxSteps"]) or (self.victory != 0):
            #final data collection then end simulation
            self.running = False

    #step

#NVResistanceModel



class NVagent(mesa.Agent):
    def __init__(self, unique_id, agent_type, model):
        super().__init__(unique_id, model)
        # TODO organize these; ALSO - there are other base constants look at agent tables initialization
        self.type = agent_type
        self.is_alive = True
        self.delayStart = self.model.random.randrange(self.model.const["delayStartMax"])
            # model steps starts at 0 -> this = 0; matlab starts at 1 (that delay is set to 1)

        # lazy consts for quick true false of agent types
        self.is_Citizen = (self.type == ATypes.CITIZEN)
        self.is_Activist = (self.type == ATypes.ACTIVIST)
        self.is_Police = (self.type == ATypes.POLICE)
        self.is_Pillar = (self.type == ATypes.PILLAR)
        
        self.jailTerm = 0
        self.inProtest = False
        self.protestCounter = 0 if (self.is_Citizen or self.is_Activist) else float('nan')
        #self.fickle = False

        if self.is_Citizen:
            # calculate my hardship and cost for joining protest
            self.activeNV = False
            self.fickle = self.model.random.random() < self.model.const["ficklePct"]
                    # weighted coin flip - should I do rule NV (decide to protest) or rule C (decide to become NV) first?
            self.immediateProtest = self.model.random.random() < self.model.const["PctImmedProtest"]
            self.k1 = 2.3
            k2 = 0.025 * 3    # 3 is included so max jail term could drop from 30 to 10 (speed sim up) but match jail cost
            
            a, b = (0.5, 0.5)
            Ey = np.exp(a + 0.5*(b**2))
            logy = a + b*self.model.random.normalvariate()
            y = np.exp( logy )
            self.hardship = np.exp(Ey - y) / (1 + np.exp(Ey - y))

            # "risk aversion" of agent to starting to protest, fear of being jailed
                # costA is original, higher for rich; costB is new, higher for poor, let's combine them
            costA = 2 * np.exp(k2 * self.model.const["MaxJailTerm"] * y) / (1 + np.exp(k2 * self.model.const["MaxJailTerm"] * y)) - 1
            costB = 2 * np.exp(k2 * self.model.const["MaxJailTerm"] / y) / (1 + np.exp(k2 * self.model.const["MaxJailTerm"] / y)) - 1
            self.arrestCost = (costA**4 + costB**4)**0.25

        if self.is_Police or self.is_Pillar:
            # how much pressure do i need to defect
            self.defect = False
            #self.defectThreshold = max(0.01, self.model.random.normalvariate(self.model.const["defectThreshold"], self.model.const["defectStDev"]) )
            self.defectThreshold = self.model.const["defectThreshold"] + (self.model.random.normalvariate() * self.model.const["defectStDev"])
            self.defectThreshold = max(0.01, self.defectThreshold)
    #__init__

    def get_visible_agents(self):
        # get other agents within my vision, used for if i'm police (looking for activist) or nv C
        adjX, adjY = self.model.grid.width/2, self.model.grid.height/2
        big_neighborhood = self.model.grid.get_neighbors(pos = self.pos,moore = True,
                                include_center = False, radius = self.model.const["vision"])
        # grid being torus means we have to adjust the coords so we can do math and ignore the wrap-around (add half the grid)
        torus_adjusted = [[(a.pos[0]+adjX)%(adjX*2),(a.pos[1]+adjY)%(adjY*2)] for a in big_neighborhood]
        self_adj = [(self.pos[0]+adjX)%(adjX*2),(self.pos[1]+adjY)%(adjY*2)]
        # find "real" radial distance as opposed to moore distance
        dists = [np.linalg.norm(np.array(self_adj) - np.array(adj_pos)) for adj_pos in torus_adjusted]
        # dists line up w adjusted pos, lines up w real pos; now we can compile real pos if adjusted pos distance fits
        neighbor_agents = [big_neighborhood[idx] for idx, dist in enumerate(dists) if dist <= self.model.const["vision"]]
        
        return neighbor_agents
    #get_visible_agents

    def ruleM(self):
        # get full neighborhood in radius (include corners), filter by if available (empty), then filter by if actually fits in distance
            # since grid is a torus, adjust pos (so that high and low positions play nicely), distance unchanged
        adjX, adjY = self.model.grid.width/2, self.model.grid.height/2
        
        big_neighborhood = self.model.grid.get_neighborhood(pos = self.pos, moore = True,
                                        include_center = False, radius = self.model.const["vision"])
        big_neighborhood = [x for x in big_neighborhood if self.model.grid.is_cell_empty(x)]
        
        torus_adjusted = [[(x[0]+adjX)%(adjX*2),(x[1]+adjY)%(adjY*2)] for x in big_neighborhood]
        self_adj = [(self.pos[0]+adjX)%(adjX*2),(self.pos[1]+adjY)%(adjY*2)]
        dists = [np.linalg.norm(np.array(self_adj) - np.array(adj_pos)) for adj_pos in torus_adjusted]
        available_spaces = [big_neighborhood[idx] for idx, dist in enumerate(dists) if dist <= self.model.const["vision"]]

        pillarProxStrat = self.model.const['pillarproxstrat']

        if any(available_spaces):
            chosen = None
            if not (self.is_Activist and pillarProxStrat):
                # non-activist, or pillar strat 0, so pick random
                chosen = self.model.random.choice(available_spaces)

            else:
                # activist and we have some pillar strat
                # activists see further with strat 2
                pillarVision = self.model.const["activistVision"] if (pillarProxStrat == 2) else self.model.const["vision"]

                neighbors = self.model.grid.get_neighbors(pos = self.pos, moore = True,
                    include_center = False, radius = pillarVision)
                bigger_torus_adj = [[(x.pos[0]+adjX)%(adjX*2),(x.pos[1]+adjY)%(adjY*2)] for x in neighbors]
                bigger_dists = [np.linalg.norm(np.array(self_adj) - np.array(adj_pos)) for adj_pos in bigger_torus_adj]
                pillarList = [neighbors[idx] for idx, dist in enumerate(dists) if neighbors[idx].is_Pillar and (dist <= self.model.const["activistVision"]) ]
                #if len(pillarList) > 0:
                #    raise Exception(f'act ({self.pos[0]},{self.pos[1]}) found pillar ({pillarList[0].pos[0]},{pillarList[0].pos[1]})')
                
                if any(pillarList):
                    # if we're strat 2 - pick the weakest one we can see
                    targetPillar = None
                    minDist = float('inf')
                    chosen = None
                    if (pillarProxStrat == 2) and (len(pillarList)>2):
                        targetPillar = min(pillarList, key = lambda x: x.defectThreshold)
                        pillarList = [targetPillar]
                        # TODO - i have no idea if this is how the matlab intends to do it (limit to 1 element)
                    # find closest space to any pillar we can reach
                    available_spaces.append(self.pos)
                    minDists = []
                    for space in available_spaces:
                        # for each space, compare dists to available pillars, store closest for each space
                        # then pick the space thats closest
                        pilDists = [np.linalg.norm(np.array(space) - np.array(pil.pos)) for pil in pillarList]
                        minDistidx = np.argmin(pilDists)
                        minDists.append([ pilDists[minDistidx], space ])
                    # of these combinations of [bestDist, spaceXY], get the spaceXY of closest space-pillar dist pilDists
                    # including our pos means min will give at least 1
                    chosen = min(minDists, key = lambda x: x[0])[1]
                else:
                    chosen = self.model.random.choice(available_spaces)

            self.model.grid.move_agent(self, chosen)
    #ruleM

    def ruleP(self):
        # police rule, get nearby neighbors within vision radius, arrest/kill if applicable
        neighbor_agents = self.get_visible_agents()       

        nNVall = len([a for a in neighbor_agents if
                  ((a.is_Activist) and not a.jailTerm) or (a.is_Citizen and a.activeNV)])
        nNV = len([a for a in neighbor_agents if a.inProtest])
        
        CanFindResistor = self.model.random.random() < self.model.const["PctFindNVResistor"]
        CanTargetNV = self.model.random.random() < self.model.const["PctTargetNV"]

        if nNVall and CanTargetNV and (CanFindResistor or nNV):    #will they target someone near them
            # (any kind of resistor nearby) and (will target someone) and (activist nearby or will target NVC)
            if nNV:
                NVNeighbors = [a for a in neighbor_agents if a.inProtest]
            else:
                NVNeighbors = [a for a in neighbor_agents if (a.is_Activist and not a.jailTerm) or (a.is_Citizen and a.activeNV)]
                
            NVTarget = self.model.random.choice(NVNeighbors)
            KillNV = self.model.random.random() < self.model.const["PctKillNV"]
            if KillNV:
                NVTarget.is_alive = False
                if NVTarget.is_Citizen:    # idk if needs to check for citizen first
                    NVTarget.activeNV = False
                self.model.grid.remove_agent(NVTarget)
                self.model.schedule.remove(NVTarget)
                
                self.model.governmentLegitimacy *= self.model.const["BackfireCoeff"]    # backfire govt result of killing agent
            else:
                # arrest them instead
                NVTarget.jailTerm = self.random.randint(1, self.model.const["MaxJailTerm"]-1)
                NVTarget.inProtest = False
                if NVTarget.is_Citizen:
                    NVTarget.activeNV = False
    #ruleP

    def ruleNV(self):
        # get agents within my vision
        neighbor_agents = self.get_visible_agents()

        # count how many police, NV citizens, and activists (i can see ie unjailed)
        policeSum, nvcSum, activistSum = (0, 0, 0)
        for neighbor in neighbor_agents:
            policeSum += neighbor.is_Police
            nvcSum += (neighbor.is_Citizen and neighbor.activeNV)
            activistSum += (neighbor.is_Activist and not neighbor.jailTerm)

        # MATLAB has 3 different rules but 2 are commented, select a rule? read paper
            # the one picked doesnt make any sense but works - if theres at least 1 activist or nvC i protest
            # otherwise its (if 0 police or more A+NVC than police, ie count) or (if more A+NVC than all neighbors i have?)
            # i think the nNV param changes between count/percentage
        #if (not policeSum) or (((activistSum + nvcSum)/policeSum) > self.model.const["nNVthreshold"]):
        #if (activistSum + nvcSum)/len(neighbor_agents) > self.model.const["nNVthreshold"]:
        if (activistSum + nvcSum) >= self.model.const["nNVthreshold"]:
            self.inProtest = True
            self.protestCounter = self.model.const["protestCycle"]
    #ruleNV

    def ruleC(self):
        # will citizen actively resist? also choose V or NV
        self.grievance = self.hardship * (1 - self.model.governmentLegitimacy)
        neighbor_agents = self.get_visible_agents()
        policeSum, activistSum, nvcSum, protestSum = (0, 0, 0, 0)
        for neighbor in neighbor_agents:
            policeSum += neighbor.is_Police
            activistSum += (neighbor.is_Activist and not neighbor.jailTerm)
            nvcSum += (neighbor.is_Citizen and neighbor.activeNV)
            protestSum += neighbor.inProtest

        # "IMPORTANT" choose arrest probability (matlab says, again theres multiple rules)
            # apparently moro netlogo and moro paper give different formulas? this is the one from the paper, not the netlogo (matlab)
        ArrestProbability = 1 - np.exp(-self.k1 * (policeSum/(1 + nvcSum + activistSum)))
        PeerPressure = min( 0.5 * protestSum / self.model.const["peerPressureNum"],  1)
        NVChoice = self.grievance*PeerPressure - ArrestProbability*self.arrestCost
        
        self.activeNV = (NVChoice > self.model.const["joinNVThresh"])
    #ruleC

    def ruleD(self):
        neighbor_agents = self.get_visible_agents()
        activistSum, nvcSum = (0, 0)
        for neighbor in neighbor_agents:
            activistSum += (neighbor.is_Activist and neighbor.inProtest)
            nvcSum += (neighbor.is_Citizen and neighbor.inProtest)
        NVratio = (nvcSum + activistSum) / len(neighbor_agents)
        if NVratio > self.defectThreshold:
            self.defect = True
    #ruleD

    def step(self):
        # read the matlab so that right conditionals are made and ordered as expected
        if self.is_alive and (self.model.schedule.steps >= (self.delayStart+1)):

            # am i tired of protesting
            if self.protestCounter > 0:
                self.protestCounter -= 1
                if self.protestCounter <= (self.model.const["protestCycle"] - self.model.const["protestDuration"]):
                    self.inProtest = False

            # can i move around
            if not self.is_Pillar and (not self.jailTerm):
                # NOTE: matlab checks based on jailterm = 0 or nan(jailterm)
                self.ruleM()

            # can i do police activities
            if self.is_Police and (not self.defect):
                self.ruleP()

            # can i protest or citizen resist
            # rule NV first if immediateProtest is false, otherwise rule C
            # each citizen this is true false at random
            doNVfirst = self.is_Citizen and self.immediateProtest
            for NVfirst in [False, True]:
                # NV rule: if i'm a NV member, should I protest?
                if doNVfirst == NVfirst:
                    if ( (self.is_Activist and not self.jailTerm) or (self.is_Citizen and self.activeNV) ) and not self.protestCounter:
                        self.ruleNV()

                    # also count how many protestors per step
                    if self.inProtest:
                        self.model.protestCount += 1
                else:
                # C rule: should i become a NV member?
                    if (self.is_Citizen and not self.jailTerm) and not (self.activeNV and not self.fickle):
                        self.ruleC()

            # can i, will i defect
            if self.is_Police or self.is_Pillar:
                # if not self.defect
                self.ruleD()

            if self.jailTerm > 0:
                self.jailTerm -= 1
            
    #step

#NVagent

