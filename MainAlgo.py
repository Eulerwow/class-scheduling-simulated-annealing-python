# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 12:54:48 2021

@author: Pengyuan Ding
"""
'''==========Solving Class Scheduling Problem by SA Algorithm in python======='''

# importing required modules
import pandas as pd
import numpy as np
import time
import copy
import random
import math
import gurobipy as grb

''' ================= initialization setting ======================'''
Workdays = 2
Dayhours = 10
Dayminutes = Dayhours*60

BigM = 10**4

n_iterations = 1000

rm_tmp = pd.read_csv("Room_Input_Data.csv", header=None)
Room_capa = np.array(rm_tmp)
Room_capa = [int(c) for c in Room_capa[0]]
Num_Room = len(Room_capa)
Num_cutters = Num_Room*Workdays - 1
Room_capa.extend(Room_capa)
#print(Room_capa)
#print(Num_Room)

cl_tmp = pd.read_csv("Class_Input_Data.csv", header=None)
Main_info = np.array(cl_tmp)
Main_info[2] = 60*Main_info[2]
# in Main_info, line[0] is the class id, line[1] is class size, 
# line[2] is class length, line[3] class's assigned tutor id

Num_Class = len(Main_info[0])
Class_list = list(range(Num_Class))
Num_Tutor = len(set(Main_info[3]))
#print(Class_list)
#print(Num_Tutor)

Tutor_Class = []
for t in range(Num_Tutor):
    tclass = [int(Main_info[0][i]) for i in range(Num_Class) if int(Main_info[3][i]) == t]
    Tutor_Class.append(tclass)

fulllist = list(range(Num_Class + Num_cutters))

''' ===== generating random schedule of classes without buffer times ==========='''

def random_sequence(Num_Class, Num_cutters):
    sequence = [0 for i in fulllist]
    #cutter_lctn = random.sample(fulllist, Num_cutters)
    cutter_lctn = [4, 9, 14, 19, 24, 29, 34]
    cutter_lctn.sort()
    for i in range(Num_cutters):
        sequence[cutter_lctn[i]] = i
    blist = [item for item in fulllist if item not in cutter_lctn]
    random.shuffle(blist)
    for i in range(Num_Class):
        sequence[blist[i]] = i + Num_cutters
    return sequence

''' ======== MIP to generate buffer times for a pre-determined sequence ===='''
def gettimetable(sequence):
    
    maxspan = 0
    for r in range(Num_cutters):
        if r ==0:
            maxspan = sum(Main_info[2][sequence[i]-(Num_cutters)] for i in range(sequence.index(r)))
        else:
            temp = sum(Main_info[2][sequence[i]-(Num_cutters)] for i in range(sequence.index(r-1)+1, sequence.index(r)))
            if temp > maxspan:
                maxspan = temp
        if r==Num_cutters-1:
            temp = sum(Main_info[2][sequence[i]-(Num_cutters)] for i in range(sequence.index(r)+1, len(sequence)))
            if temp > maxspan:
                maxspan = temp
    
    higherminutebound = max(Dayminutes, maxspan)
    
    daylength_penalty = 0
    if maxspan > Dayminutes:
        daylength_penalty = 1
    
    MIP = grb.Model()
    MIP.Params.LogToConsole = 0
    MIP.setParam('NonConvex', 2)
    MIP.setParam('TimeLimit', 0.5)
    
    #real number decision varialbes: 'buffer time' before each class  
    buffer = MIP.addVars(Num_Class, vtype=grb.GRB.INTEGER, name="buffer")
    #binary decision variables to detect tutor clashes
    overlap = MIP.addVars(Num_Tutor,Num_Class, Num_Class, vtype=grb.GRB.BINARY, name="overlap")
    #binary decision variables to detect tutor concurrent clashes
    concur = MIP.addVars(Num_Tutor,Num_Class, Num_Class, vtype=grb.GRB.BINARY, name="concur")
    
    timetable = [[0, 0] for cl in range(Num_Class)]
    
    #calculate penalty for total time of a classroom longer than dayhours, meanwhile create a timetable with decision variables
    rtimes = []
    for r in range(Num_cutters):
        rtime = 0
        rlctn = int(sequence.index(r))
        if r == 0:
            if rlctn > 0:
                for i in range(rlctn):
                    classid = sequence[i]-(Num_cutters)
                    rtime += buffer[classid]
                    #timetable[classid][0] = copy.deepcopy(rtime)
                    timetable[classid][0] = grb.quicksum(buffer[sequence[j]-(Num_cutters)] for j in range(i+1)) + grb.quicksum(Main_info[2][sequence[j]-(Num_cutters)] for j in range(i))
                    rtime += Main_info[2][classid]
                    #timetable[classid][1] = copy.copy(rtime)
                    timetable[classid][1] = grb.quicksum(buffer[sequence[j]-(Num_cutters)] for j in range(i+1)) + grb.quicksum(Main_info[2][sequence[j]-(Num_cutters)] for j in range(i+1))
        
        elif sequence.index(r) - sequence.index(r-1) > 1:
            rm1lctn = sequence.index(r-1)
            for i in range(rm1lctn+1, rlctn):
                classid = sequence[i]-(Num_cutters)
                rtime += buffer[classid]
                timetable[classid][0] = grb.quicksum(buffer[sequence[j]-(Num_cutters)] for j in range(rm1lctn+1, i+1)) + grb.quicksum(Main_info[2][sequence[j]-(Num_cutters)] for j in range(rm1lctn+1, i))
                rtime += Main_info[2][classid]
                timetable[classid][1] = grb.quicksum(buffer[sequence[j]-(Num_cutters)] for j in range(rm1lctn+1, i+1)) + grb.quicksum(Main_info[2][sequence[j]-(Num_cutters)] for j in range(rm1lctn+1, i+1))
        rtimes.append(rtime)
            
        if r == Num_cutters-1:
            if rlctn < len(fulllist)-1:
                rtime = 0
                for i in range(rlctn+1, len(fulllist)):
                    classid = sequence[i]-(Num_cutters)
                    rtime += buffer[classid]
                    timetable[classid][0] = grb.quicksum(buffer[sequence[j]-(Num_cutters)] for j in range(rlctn+1, i+1)) + grb.quicksum(Main_info[2][sequence[j]-(Num_cutters)] for j in range(rlctn+1, i))
                
                    rtime += Main_info[2][classid]
                    timetable[classid][1] = grb.quicksum(buffer[sequence[j]-(Num_cutters)] for j in range(rlctn+1, i+1)) + grb.quicksum(Main_info[2][sequence[j]-(Num_cutters)] for j in range(rlctn+1, i+1))
                rtimes.append(rtime)
                
        MIP.addConstr(rtimes[r] <= higherminutebound)
        
    MIP.addConstr(rtimes[Num_cutters] <= higherminutebound)
    
    #concurrence counting                                                
    for tutor in range(Num_Tutor): 
        Tutor_Class_SAT = [clas for clas in Tutor_Class[tutor] if sequence.index(clas+Num_cutters) < sequence.index(3)]
        for clas in Tutor_Class_SAT:
            for otherclas in Tutor_Class_SAT[Tutor_Class_SAT.index(clas)+1 :]:#
                if clas != otherclas:
                    MIP.addConstr(8100*concur[tutor,clas,otherclas] >=8100- (timetable[clas][0] - timetable[otherclas][0])*(timetable[clas][0] - timetable[otherclas][0]))
                    
        Tutor_Class_SUN = [clas for clas in Tutor_Class[tutor] if clas not in Tutor_Class_SAT]
        for clas in Tutor_Class_SUN:
            for otherclas in Tutor_Class_SUN[Tutor_Class_SUN.index(clas)+1 :]:#
                if clas != otherclas:
                    MIP.addConstr(8100*concur[tutor,clas,otherclas] >=8100- (timetable[clas][0] - timetable[otherclas][0])*(timetable[clas][0] - timetable[otherclas][0]))
                           
    
    #overlap counting   
    for tutor in range(Num_Tutor): 
        Tutor_Class_SAT = [clas for clas in Tutor_Class[tutor] if sequence.index(clas+Num_cutters) < sequence.index(3)]
        for clas in Tutor_Class_SAT:
            for otherclas in Tutor_Class_SAT:
                if clas != otherclas:
                    MIP.addConstr(BigM*overlap[tutor,clas,otherclas] >= -(timetable[clas][0] - timetable[otherclas][0])*(timetable[clas][0] - timetable[otherclas][1]),  "c1-"+str(tutor)+"-"+str(clas)+"-"+str(otherclas))
                    #MIP.addConstr(BigM*(timetable[clas][0] - timetable[otherclas][0])*(timetable[clas][0] - timetable[otherclas][0]) >= 1)
        Tutor_Class_SUN = [clas for clas in Tutor_Class[tutor] if clas not in Tutor_Class_SAT]
        for clas in Tutor_Class_SUN:
            for otherclas in Tutor_Class_SUN:
                if clas != otherclas:
                    MIP.addConstr(BigM*overlap[tutor,clas,otherclas] >= -(timetable[clas][0] - timetable[otherclas][0])*(timetable[clas][0] - timetable[otherclas][1]))
                    #MIP.addConstr(BigM*(timetable[clas][0] - timetable[otherclas][0])*(timetable[clas][0] - timetable[otherclas][0]) >= 1)
    
    MIP.setObjective(grb.quicksum(overlap[tutor,c1,c2]+concur[tutor,c1,c2] for tutor in range(Num_Tutor) for c1 in range(Num_Class) for c2 in range(Num_Class)), grb.GRB.MINIMIZE)
    
    MIP.addConstrs(buffer[c] >=0 for c in range(Num_Class))
        
    MIP.optimize()
    
    obj = MIP.objVal
    #print(obj)   
    
    Varvalues = []
    for v in MIP.getVars():
        Varvalues.append(v.x)
        #if v.x != 0:
            #print('%s %g' % (v.varName, v.x))    
    
    resulttable = [[0, 0] for cl in range(Num_Class)]
    for r in range(Num_cutters):
        rtime = 0
        rlctn = int(sequence.index(r))
        if r == 0:
            if rlctn > 0:
                for i in range(rlctn):
                    classid = sequence[i]-(Num_cutters)
                    rtime += Varvalues[classid]
                    resulttable[classid][0] = rtime
                    rtime += Main_info[2][classid]
                    resulttable[classid][1] = rtime
        
        elif sequence.index(r) - sequence.index(r-1) > 1:
            rm1lctn = sequence.index(r-1)
            for i in range(rm1lctn+1, rlctn):
                classid = sequence[i]-(Num_cutters)
                rtime += Varvalues[classid]
                resulttable[classid][0] = rtime
                rtime += Main_info[2][classid]
                resulttable[classid][1] = rtime
        rtimes.append(rtime)
            
        if r == Num_cutters-1:
            if rlctn < len(fulllist)-1:
                rtime = 0
                for i in range(rlctn+1, len(fulllist)):
                    classid = sequence[i]-(Num_cutters)
                    rtime += Varvalues[classid]
                    resulttable[classid][0] = rtime
                    rtime += Main_info[2][classid]
                    resulttable[classid][1] = rtime
                rtimes.append(rtime)                
       
    '''  
    reserve for debugging  
    print(MIP.getAttr(grb.GRB.Attr.NumConstrs))
    
    c0 = MIP.getConstrByName("c1-0-23-24")
    print(c0)   
    '''
    return obj, daylength_penalty, resulttable

''' ======== outer Simulated Annealing algo for optimal schedule ========='''
def getfitness(sequence):
    # max day length constraint penalty
    room_size_penalty = 0
    for r in range(Num_cutters):
        rlctn = int(sequence.index(r))
        if r == 0:
            if rlctn > 0:
                for i in range(rlctn):
                    classid = sequence[i]-(Num_cutters)
                    if Main_info[1][classid] > Room_capa[r]:
                        room_size_penalty += 1
                        
        elif sequence.index(r) - sequence.index(r-1) > 1:
            rm1lctn = sequence.index(r-1)
            for i in range(rm1lctn+1, rlctn):
                classid = sequence[i]-(Num_cutters)
                if Main_info[1][classid] > Room_capa[r]:
                    room_size_penalty += 1
            
        if r == Num_cutters-1:
            if rlctn < len(fulllist)-1:
                for i in range(rlctn+1, len(fulllist)):
                    classid = sequence[i]-(Num_cutters)
                    if Main_info[1][classid] > Room_capa[r+1]:
                        room_size_penalty += 1
                        
    penal_1, penal_2, timetable = gettimetable(sequence)
    penal_3 = room_size_penalty    
    total_penal = penal_1 + penal_2 + penal_3
    
    return total_penal, timetable

def mutate(sequence, num_mutate):
    mutated = copy.deepcopy(sequence)
    m_chg = random.sample(Class_list, num_mutate)
    m_chg.sort()
    m_chg = [i + Num_cutters for i in m_chg]
    inxlist = []
    for gene in m_chg:
        inxlist.append(mutated.index(gene))
    inxlist.sort()
    temp = mutated[inxlist[0]]
    for i in range(len(inxlist)-1):
        mutated[inxlist[i]] = mutated[inxlist[i+1]]
    mutated[inxlist[-1]] = temp
    return mutated

'''reserve for debugging
initial_seq = random_sequence(Num_Class, Num_cutters)
#print(initial_seq)

#test on one sequence
sequence = initial_seq
sequence = [30, 7, 28, 26, 0, 31, 19, 18, 34, 1, 10, 12, 14, 23, 2, 15, 9, 13, 27, 3, 25, 24, 22, 20, 4, 32, 11, 33, 29, 5, 16, 17, 21, 35, 6, 8]
    
fit = getfitness(sequence)
print(fit)
'''

# simulated annealing algorithm
def simulated_annealing(objective, num_mutate, temp):
	# generate an initial point
    best = random_sequence(Num_Class, Num_cutters)
	# evaluate the initial point
    best_eval, best_timetable = objective(best)
	# current working solution
    curr, curr_eval, curr_timetable = best, best_eval, best_timetable
    scores = list()
	# run the algorithm
    for i in range(n_iterations):
		# take a step
        candidate = mutate(curr, num_mutate)
		# evaluate candidate point
        candidate_eval, candidate_timetable = objective(candidate)
		# check for new best solution
        if candidate_eval < best_eval:
			# store new best point
            best, best_eval, best_timetable = candidate, candidate_eval, candidate_timetable
			# keep track of scores
            scores.append(best_eval)
			# report progress
            print('>%d %d' % (i,  best_eval))
		# difference between candidate and current point evaluation
        diff = candidate_eval - curr_eval
		# calculate temperature for current epoch
        t = temp / float(i + 1)
		# calculate metropolis acceptance criterion
        metropolis = math.exp(-diff / t)
		# check if we should keep the new point
        if diff < 0 or np.random.rand() < metropolis:
			# store the new current point
            curr, curr_eval = candidate, candidate_eval
        if curr_eval == 0:
            break
    return [best, best_eval, best_timetable, scores]

''' ======== run SA algorithm ========='''
start_time = time.time()
best_sequence, lowest_penalty, the_timetable, history = simulated_annealing(getfitness, 2, 100)
end_time = time.time()
time = end_time - start_time

''' ======== print result ========='''

print('Algorithm find optimal schedule in %d seconds' %time)
#print(best_sequence)
#print(the_timetable)

def printtitle(room):
    print("Classroom %d:" %room)
    print("Class ID    Tutor ID    Start Time   End Time")
    
def printtime(position):
    classid = best_sequence[position]-(Num_cutters)
    tutor = Main_info[3][classid]
    start_hr = int(9+the_timetable[classid][0]/60)
    start_min = round((9+the_timetable[classid][0]/60 - start_hr)*60)
    end_hr = int(9+the_timetable[classid][1]/60)
    end_min = round((9+the_timetable[classid][1]/60 - end_hr)*60)
    print( "   %02d          %02d        %02d:%02d        %02d:%02d" %(classid, tutor, start_hr, start_min, end_hr, end_min))

print("Sat:")    
for r in range(4):
    room = r +1
    printtitle(room)
    rlctn = int(best_sequence.index(r))
    if r == 0:
        for i in range(rlctn):   
            printtime(i)            
    if r > 0:
        rm1lctn = int(best_sequence.index(r-1))
        for i in range(rm1lctn+1, rlctn):
            printtime(i)
            
print("Sun:")    
for r in range(4, Num_cutters):
    room = r%4 +1
    printtitle(room) 
    rlctn = int(best_sequence.index(r))
    rm1lctn = int(best_sequence.index(r-1))
    for i in range(rm1lctn+1, rlctn):
        printtime(i)          
    if r == Num_cutters-1:
        room = Num_Room
        printtitle(room) 
        for i in range(rlctn+1, len(fulllist)):            
            printtime(i)
    
