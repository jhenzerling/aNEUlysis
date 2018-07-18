#Helper to analysis module
from larcv import larcv
from larcv.dataloader2 import larcv_threadio

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import ROOT as ROOT

import os,sys,time

import tensorflow.contrib.slim as slim
import tensorflow.python.platform
import tensorflow as tf

#from Logging import mod_log as log
from Configs import config_main as con
from Loaders import mod_psingle as ps
from Machine_Learning import Network as MLN
from Logging import mod_log as log

#Create Grid to look at results


#Grapher for softmax probs from CSV's
#Which takes 'pred' or 'true'
def Grapher(fpath,which):
    df = pd.read_csv(fpath)
    #df.describe()
    
    #Plot the results to see softmax for different PIDS
    particles = ['Electron','Muon','Photon','Pion','Proton']
    fig,ax = plt.subplots(figsize=(12,8),facecolor='w')
    for index, particle in enumerate(particles):
	if which == 'pred':
        	sub_df = df.query('prediction==%d' % index)
		hist, _ = np.histogram(sub_df.probability.values, bins=25, range=(0.,1.),
				weights=[1./sub_df.index.size] * sub_df.index.size)
	elif which == 'true':
        	sub_df = df.query('label==%d' % index)
        	hist, _ = np.histogram(sub_df.label_probability.values, 
                       	       	bins=25, range=(0.,1.), 
                       	       	weights=[1./sub_df.index.size] * sub_df.index.size)
	elif which == 'all':
		sub_df = df.query('label in [0,2]')
		surb_df = df.query('label not in [0,2]')
		hist, _ = np.histogram(sub_df.probability.values, bins=25, range=(0.,1.),
				weights=[1./sub_df.index.size] * sub_df.index.size)
		hist2, _2 = np.histogram(surb_df.probability.values, bins=25, range=(0.,1.),
				weights=[1./surb_df.index.size] * surb_df.index.size)
	else:
		print('Bad type selection on 2nd input')
	
        
        # Plot
	if which == 'all':
		label1 = 'Showers'
		label2 = 'Tracks'
		plt.plot(np.arange(0.,1.,1./25.),hist,  marker='o')
		plt.plot(np.arange(0.,1.,1./25.),hist2,  marker='o')	
		
	else:
        	label = '%s (%d events)' % (particle,sub_df.index.size)
        	plt.plot(np.arange(0.,1.,1./25.),hist, label=label, marker='o')
        # Decoration!
        plt.tick_params(labelsize=20)
        plt.xlabel('Softmax Probability',fontsize=20,fontweight='bold')
        plt.ylabel('Fraction of Events',fontsize=20,fontweight='bold')
    leg=plt.legend(fontsize=16,loc=2)
    leg_frame=leg.get_frame()
    leg_frame.set_facecolor('white')
    plt.grid()
    plt.show()

def GrapherANA(fpath,which):
    df = pd.read_csv(fpath)
    #df.describe()
    
    #Plot the results to see softmax for different PIDS
    particles = ['Electron','Muon','Photon','Pion','Proton']
    fig,ax = plt.subplots(figsize=(12,8),facecolor='w')
    for index, particle in enumerate(particles):
        sub_df = df.query('label==%d and prediction==%d' % (which,index))
        hist, _ = np.histogram(sub_df.probability.values, bins=25, range=(0.,1.),
                               weights=[1./sub_df.index.size] * sub_df.index.size)
        
        # Plot
        label = 'Predicted %s for True Label %s (%d events)' % (particle,particles[which],sub_df.index.size)
        plt.plot(np.arange(0.,1.,1./25.),hist, label=label, marker='o')
        # Decoration!
        plt.tick_params(labelsize=20)
        plt.xlabel('Softmax Probability',fontsize=20,fontweight='bold')
        plt.ylabel('Fraction of Events',fontsize=20,fontweight='bold')
    leg=plt.legend(fontsize=16,loc=2)
    leg_frame=leg.get_frame()
    leg_frame.set_facecolor('white')
    plt.grid()
    plt.show()

#AOJAOIJDOI
def GrapherANA2(fpath,which):
    df = pd.read_csv(fpath)
    #df.describe()
    
    #Plot the results to see softmax for different PIDS
    particles = ['Electron','Muon','Photon','Pion','Proton']
    fig,ax = plt.subplots(figsize=(12,8),facecolor='w')
    sub_df = df.query('label==%d' % (which))
    hist0, _0 = np.histogram(sub_df.arr0.values, bins=25, range=(0.,1.),
                               weights=[1./sub_df.index.size] * sub_df.index.size)
    hist1, _1 = np.histogram(sub_df.arr1.values, bins=25, range=(0.,1.),
                               weights=[1./sub_df.index.size] * sub_df.index.size)
    hist2, _2 = np.histogram(sub_df.arr2.values, bins=25, range=(0.,1.),
                               weights=[1./sub_df.index.size] * sub_df.index.size)
    hist3, _3 = np.histogram(sub_df.arr3.values, bins=25, range=(0.,1.),
                               weights=[1./sub_df.index.size] * sub_df.index.size)
    hist4, _4 = np.histogram(sub_df.arr4.values, bins=25, range=(0.,1.),
                               weights=[1./sub_df.index.size] * sub_df.index.size)
        
    # Plot
    for i in range(0,4):
        label = 'Predicted %s for True Label %s (%d events)' % (particles[i],particles[which],sub_df.index.size)
        plt.plot(np.arange(0.,1.,1./25.),hist, label=label, marker='o')
        # Decoration!
        plt.tick_params(labelsize=20)
        plt.xlabel('Softmax Probability',fontsize=20,fontweight='bold')
        plt.ylabel('Fraction of Events',fontsize=20,fontweight='bold')
    leg=plt.legend(fontsize=16,loc=2)
    leg_frame=leg.get_frame()
    leg_frame.set_facecolor('white')
    plt.grid()
    plt.show()

def GrapherSingle(fpath,part):
    df = pd.read_csv(fpath)
    #df.describe()
    
    #Plot the results to see softmax for different PIDS
    index = part
    particles = ['Electron','Photon','Muon','Pion','Proton']
    fig,ax = plt.subplots(figsize=(12,8),facecolor='w')
    sub_df = df.query('prediction==%d' % index)
    hist, _ = np.histogram(sub_df.probability.values, bins=25, range=(0.,1.),
                           weights=[1./sub_df.index.size] * sub_df.index.size)
    sub_df2 = df.query('label==%d' % index)
    hist2, _2 = np.histogram(sub_df2.label_probability.values, 
                             bins=25, range=(0.,1.), 
                             weights=[1./sub_df2.index.size] * sub_df2.index.size)
	
        
    # Plot
    label = '%s (%d events) - Pred' % (particles[index],sub_df.index.size)
    label2 = '%s (%d events) - True' % (particles[index],sub_df2.index.size)
    plt.plot(np.arange(0.,1.,1./25.),hist, label=label, marker='o')
    plt.plot(np.arange(0.,1.,1./25.),hist2, label=label2, marker='o')
    # Decoration!
    plt.tick_params(labelsize=20)
    plt.xlabel('Softmax Probability',fontsize=20,fontweight='bold')
    plt.ylabel('Fraction of Events',fontsize=20,fontweight='bold')
    leg=plt.legend(fontsize=16,loc=2)
    leg_frame=leg.get_frame()
    leg_frame.set_facecolor('white')
    plt.grid()
    plt.show()


#Calculate Correct Fractions for Geometry Types
def GeomFrac(fpath):
    
	df = pd.read_csv(fpath)
        
	emshower = df.query('label in [0,2]').index.size
	emshowerc = df.query('label in [0,2] and prediction in [0,2]').index.size
        ec = df.query('prediction in [0,2]').index.size
        emw = df.query('label in [0,2] and prediction not in [0,2]').index.size
	print("EM Shower correct fraction:",emshowerc,"/",emshower,"=",float(emshowerc)/emshower*100.,"%")
       

	track = df.query('label not in [0,2]').index.size
	trackc = df.query('label not in [0,2] and prediction not in [0,2]').index.size
        tc = df.query('prediction not in [0,2]').index.size
        trw = df.query('label not in [0,2] and prediction in [0,2]').index.size
	print("Track correct fraction:",trackc,"/",track,"=",float(trackc)/track*100.,"%")



        print('Efficiency')
	print(emshowerc,'/',emshower,'=',float(emshowerc)/emshower,'showright')
        print(emw,'/',emshower,'=',float(emw)/emshower*100.,'% wrong')
	print(trackc,'/',track,'=',float(trackc)/track,'trackright')
        print(trw,'/',track,'=',float(trw)/track*100.,'% wrong')

        print('Purity')
	print(emshowerc,'/',ec,'=',float(emshowerc)/ec,'showright')
        print(emw,'/',ec,'=',float(emw)/ec*100.,'% wrong')
	print(trackc,'/',tc,'=',float(trackc)/tc,'trackright')
        print(trw,'/',tc,'=',float(trw)/tc*100.,'% wrong')
        
        

	return [emshowerc/emshower,trackc/track]

#Calculate Correct Fractions for PID
def PIDFrac(fpath):
	df = pd.read_csv(fpath)
	sp = 0
	sl = 0
	particles = ['Electron','Muon','Photon','Pion','Proton']
	for index, particle in enumerate(particles):
		pred = df.query('label==%d' % index).index.size
		labe = df.query('label==%d and prediction==%d' % (index,index)).index.size
		sp += pred
		sl += labe
		frac = float(labe)/pred
		st = float(sl)/sp
		print('Fraction Correct of ',particles[index],' is: ',
				labe,'/',pred,'=',frac*100,'%')

	print('Fraction Correct of Total is: ',
			sl,'/',sp,'=',st*100,'%')

	return st

def PIDFrac2(fpath):
	df = pd.read_csv(fpath)
	m = np.empty(shape = [5,5])
        n = np.empty(shape = [5,5])
	particles = ['Electron','Muon','Photon','Pion','Proton']
	for index,particle in enumerate(particles):
		for index2,particle in enumerate(particles):
			pred = df.query('label==%d' % index).index.size
			labe = df.query('label==%d and prediction==%d' % (index,index2)).index.size
                        labe2 = df.query('prediction==%d' % (index2)).index.size
			frac = float(labe)/pred
                        frac2 = float(labe)/labe2
			m[index,index2] = frac
                        n[index,index2] = frac2
			print(particles[index],' is counted as ', particles[index2],
				labe,'/',pred,'=',frac*100,'% Efficiency')

                        #print(particles[index],' is counted as ', particles[index2],
                        #      labe,'/',labe2,'=',frac2*100,'% Purity')

                        
	return [m,n]


#Plot something you want
def AnaPlot(batch,part):
    c = 1
    d = 1
    proc = ps.IOPrep('train',batch)
    [im,la] = ps.allocate('train',proc)
    dim = proc.fetch_data('train_image').dim()
    while d == 1:
        if ps.OHtoName(la[c]) == part:
            ps.OHtoName(la[c])
            log.PSingle_plot(im,la,dim,c)
            d += 1
        else:
            if c == batch:
                break
            else:
                c +=1


#Plot something you want
def AnaPlot2(batch,part):
    c = 1
    d = 1
    proc = ps.IOPrep('train',batch)
    [im,la] = ps.allocate('train',proc)
    dim = proc.fetch_data('train_image').dim()
    while d == 1:
        if la[c][part] == 1:
	    print(la[c])
            ps.OHtoName(la[c])
            log.PSingle_plot(im,la,dim,c)
            d += 1
        else:
            if c == batch:
                break
            else:
                c +=1




def GrapherFSM(fpath,choice):
    df = pd.read_csv(fpath)
    
    #Plot the results to see softmax for different PIDS
    particles = ['Electron','Muon','Photon','Pion','Proton']
    fig,ax = plt.subplots(figsize=(12,8),facecolor='w')
    sub_df = df.query('label==%d' % choice)
        
    hist, _ = np.histogram(sub_df.arr0, 
                           bins=25, range=(0.,1.), 
                           weights=[1./sub_df.index.size] * sub_df.index.size)
        
    label = '%s' % (particles[0])
    plt.plot(np.arange(0.,1.,1./25.),hist, label=label, marker='o')
        
    hist, _ = np.histogram(sub_df.arr1, 
                           bins=25, range=(0.,1.), 
                           weights=[1./sub_df.index.size] * sub_df.index.size)
        
    label = '%s' % (particles[1])
    plt.plot(np.arange(0.,1.,1./25.),hist, label=label, marker='o')


    hist, _ = np.histogram(sub_df.arr2, 
                           bins=25, range=(0.,1.), 
                           weights=[1./sub_df.index.size] * sub_df.index.size)
       
    label = '%s' % (particles[2])
    plt.plot(np.arange(0.,1.,1./25.),hist, label=label, marker='o')
       
    hist, _ = np.histogram(sub_df.arr3, 
                           bins=25, range=(0.,1.), 
                           weights=[1./sub_df.index.size] * sub_df.index.size)

        
    label = '%s' % (particles[3])
    plt.plot(np.arange(0.,1.,1./25.),hist, label=label, marker='o')
        
    hist,  _ = np.histogram(sub_df.arr4, 
                            bins=25, range=(0.,1.), 
                            weights=[1./sub_df.index.size] * sub_df.index.size)
	
    label = '%s' % (particles[4])
    plt.plot(np.arange(0.,1.,1./25.),hist, label=label, marker='o')

        # Plot
        #label = '%s (%d events)' % (particle,sub_df.index.size)
        #plt.plot(np.arange(0.,1.,1./25.),hist, label=label, marker='o')
        # Decoration!
    plt.tick_params(labelsize=20)
    plt.xlabel('Softmax Probability',fontsize=20,fontweight='bold')
    plt.ylabel('Fraction of Events',fontsize=20,fontweight='bold')
    leg=plt.legend(fontsize=16,loc=2)
    leg_frame=leg.get_frame()
    leg_frame.set_facecolor('white')
    plt.grid()
    plt.show()






########

def DataInfo(choice):
    ROOT.TFile.Open('/user/jhenzerling/work/neunet/Data/PSingle/train_50k.root').ls()
    ROOT.TFile.Open('/user/jhenzerling/work/neunet/Data/PSingle/test_40k.root').ls()
    fchoice = choice
    print('choice is')
    print(fchoice)
    
    # Create TChain, count # of entries
    chain_image2d = ROOT.TChain('image2d_data_tree')
    chain_image2d.AddFile(fchoice)
    print(chain_image2d.GetEntries(),'image entries found!')
    
    # Create TChain, count # of entries
    chain_particle = ROOT.TChain('particle_mctruth_tree')
    chain_particle.AddFile(fchoice)
    print(chain_image2d.GetEntries(),'part mctruth entries found!')
    
    pdg_array      = np.zeros([chain_particle.GetEntries()],dtype=np.int32)
    energy_array   = np.zeros([chain_particle.GetEntries()],dtype=np.float64)
    momentum_array = np.zeros([chain_particle.GetEntries()],dtype=np.float64)
    
    for index in range(chain_particle.GetEntries()):
        chain_particle.GetEntry(index)
        particle = chain_particle.particle_mctruth_branch.as_vector().front()
        pdg = int(particle.pdg_code())
        total_energy   = particle.energy_init() * 1000.
        kinetic_energy = total_energy - larcv.ParticleMass(pdg)
        momentum = np.sqrt(np.power(total_energy,2) - np.power(larcv.ParticleMass(pdg),2))
        
        pdg_array[index]      = pdg
        energy_array[index]   = kinetic_energy
        momentum_array[index] = momentum
        
        #if momentum > 800:
        #    print(pdg,kinetic_energy,momentum)
        
    df = pd.DataFrame(data={'pdg' : pdg_array, 'energy' : energy_array, 'momentum' : momentum_array})
        
    pdg_list, pdg_counts = np.unique(df.pdg.values,return_counts=True)
    
    print('PDGs found: {}'.format(pdg_list))
    print('PDG counts: {}'.format(pdg_counts))
    
    PDG2NAME = {11   : 'electron',
                22   : 'gamma',
                13   : 'muon',
                211  : 'pion',
                2212 : 'proton'}
    print('')
    for pdg in pdg_list:
        sub_df = df.query('pdg=={}'.format(pdg))
        min_value = sub_df.momentum.values.min()
        max_value = sub_df.momentum.values.max()
        print('{:10s} momentum range: {:.3g} => {:.3g} MeV/c'.format(PDG2NAME[pdg], min_value, max_value))
            
    print('')
    for pdg in pdg_list:
        sub_df = df.query('pdg=={}'.format(pdg))
        mineng = sub_df.energy.values.min()
        maxeng = sub_df.energy.values.max()
        print('{:10s} kinenergy range: {:.3g} => {:.3g} MeV'.format(PDG2NAME[pdg], mineng, maxeng))
    print('')

def GetKinetics(choice):
    ROOT.TFile.Open('/user/jhenzerling/work/neunet/Data/PSingle/train_50k.root').ls()
    ROOT.TFile.Open('/user/jhenzerling/work/neunet/Data/PSingle/test_40k.root').ls()
    fchoice = choice
    print('choice is')
    print(fchoice)
    
    # Create TChain, count # of entries
    chain_image2d = ROOT.TChain('image2d_data_tree')
    chain_image2d.AddFile(fchoice)
    print(chain_image2d.GetEntries(),'image entries found!')
    
    # Create TChain, count # of entries
    chain_particle = ROOT.TChain('particle_mctruth_tree')
    chain_particle.AddFile(fchoice)
    print(chain_image2d.GetEntries(),'part mctruth entries found!')
    
    pdg_array      = np.zeros([chain_particle.GetEntries()],dtype=np.int32)
    energy_array   = np.zeros([chain_particle.GetEntries()],dtype=np.float64)
    momentum_array = np.zeros([chain_particle.GetEntries()],dtype=np.float64)
    id_array       = np.zeros([chain_particle.GetEntries()],dtype=np.int32)
    
    for index in range(chain_particle.GetEntries()):
        chain_particle.GetEntry(index)
        particle = chain_particle.particle_mctruth_branch.as_vector().front()
        pdg = int(particle.pdg_code())
        total_energy   = particle.energy_init() * 1000.
        kinetic_energy = total_energy - larcv.ParticleMass(pdg)
        momentum = np.sqrt(np.power(total_energy,2) - np.power(larcv.ParticleMass(pdg),2))
        
        pdg_array[index]      = pdg
        energy_array[index]   = kinetic_energy
        momentum_array[index] = momentum
        id_array[index] = index
        
        #if momentum > 800:
        #    print(pdg,kinetic_energy,momentum)
        
    df = pd.DataFrame(data={'entry' : id_array, 'pdg' : pdg_array, 'energy' : energy_array, 'momentum' : momentum_array})

    return df


#Softmax of Particle w/res kinetics
def KineticGrapher(fpath,choice):
    df = pd.read_csv(fpath)
    ki = GetKinetics('test_40k.root')
    
    #Plot the results to see softmax for different PIDS
    particles = ['Electron','Muon','Photon','Pion','Proton']
    fig,ax = plt.subplots(figsize=(12,8),facecolor='w')
    #sub_df = df.query('label==%d' % choice)
    #merge data
    meg = pd.merge(df,ki,left_index=True,right_index = True)
    sub_df = meg.query('prediction==%d' % choice)
    sub_df2 = meg.query('label==%d and prediction==%d' % (choice,choice))
    x = sub_df.momentum
    y = sub_df.energy
    x2 = sub_df2.momentum
    y2 = sub_df2.energy

    z = y2
    z2 = y

    hist,  _ = np.histogram(z, 
                        bins=25, range=(0.,800.))
                        #weights=[1./z.index.size] * z.index.size)

    hist2,  _2 = np.histogram(z2, 
                        bins=25, range=(0.,800.))
                        #weights=[1./z2.index.size] * z2.index.size)

    i=1
    ration = np.zeros((25))
    while i<25:
        ration[i] = float(hist[i])/hist2[i]
        i=i+1
    print(hist)
    print(hist2)
    print(ration)

    # Plot
    label = '%s (%d predicted events)' % (particles[choice],z2.index.size)
    plt.plot(np.arange(0.,800.,800./25.),ration, label=label, marker='o',color='red')
        # Decoration!
    plt.tick_params(labelsize=20)
    #plt.title('True %s',fontsize=20,fontweight='bold')
    plt.xlabel('Energy',fontsize=20,fontweight='bold')
    plt.ylabel('Purity',fontsize=20,fontweight='bold')
    leg=plt.legend(fontsize=16,loc=2)
    leg_frame=leg.get_frame()
    leg_frame.set_facecolor('white')
    plt.grid()
    plt.show()
