#First Physics Analysis Module, Inference Study a la Kazu
from larcv import larcv
from larcv.dataloader2 import larcv_threadio
import ROOT as R

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

import os,sys,time
start = time.time()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow.contrib.slim as slim
import tensorflow.python.platform
import tensorflow as tf

#from Logging import mod_log as log
from Configs import config_analysis as con
from Loaders import mod_psingle as ps
from Machine_Learning import Network as MLN
from Physics import ana_graph as ag
from Logging import mod_log as log

#######

#Path to Weights
wpath = con.PATH['Home']
ppath = con.PATH['Physics']
snap = con.PATH['Snap']

#Infor for IO
batchsize = con.LEARN['Batch']
cnumb = con.DATA['Classes']
cfg = ps.get_CFG('test')

########
#Create a CSV of the output for probability analysis
fname = con.FNAME['Half']
fname2= con.FNAME['Full']
fname3 = 'inf3.csv'
fname4 = 'inf4.csv'

fpath = wpath + ppath + fname
fpath2 = wpath + ppath + fname2
fpath3 = wpath + ppath + fname3
fpath4 = wpath + ppath + fname4

fpathC = fpath3
fnameC = fname3
    
    #Check if inference already done
if os.path.exists(fpathC) != True:
    print('File does not exist yet')

    #Set up IO
    tep = larcv_threadio()
    tep.configure(cfg)
    tep.start_manager(batchsize)
    time.sleep(2)
    tep.next(store_entries=True,store_event_ids=True)

    #Call the dimensions of the data
    tedim = tep.fetch_data('test_image').dim()

    ########

    #Set input
    rawinput = tf.placeholder(tf.float32,[None,tedim[1]*tedim[2]*tedim[3]],name='raw')
    input2d = tf.reshape(rawinput,[-1,tedim[1],tedim[2],tedim[3]],name='input')
    
    #Build the net
    net,endp = MLN.build2(input_tensor = input2d, num_class=cnumb, trainable=False, debug = False)
    print('Built Net')
    #Define Softmax
    softmax = tf.nn.softmax(logits=net)
    
    sess = tf.InteractiveSession()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        init = tf.global_variables_initializer()
        print('Hit Sess')
        # Load weights
        saver = tf.train.Saver()
        saver.restore(sess, snap)
        print('Loaded Weights')

        fout = open(fnameC,'w')
        if fpathC == fpath3:
            fout.write('entry,run,subrun,event,prediction,label,arr0,arr1,arr2,arr3,arr4\n')
        else:
            fout.write('entry,run,subrun,event,prediction,probability,label,label_probability\n')

        ctr = 0

        #Call number of events
        events = tep.fetch_n_entries()
        print('Using Events: ', events)
        print('Opened File')
        while ctr < events:
            #Set feed for data
            tedat = tep.fetch_data('test_image').data()
            telab = tep.fetch_data('test_label').data()
            tefeed = {rawinput:tedat}
        
            #Run Softmax
            softmaxb = sess.run(softmax,feed_dict=tefeed)
            if (ctr % 1000) == 0:
                print('%g %% completed in %f seconds' % ((100*ctr/events), time.time() - start))
                prevents = tep.fetch_event_ids()
                prentries = tep.fetch_entries()
        
            #Store in csv
            for i in xrange(len(softmaxb)):
                softmaxarr = softmaxb[i]
                sentry = prentries[i]
                sevent = prevents[i]
            
                pred = np.argmax(softmaxarr)
                pprob = softmaxarr[pred]
                labe = np.argmax(telab[i])
                plab = softmaxarr[labe]
            
                if fpathC == fpath3:
                    dstring = '%d,%d,%d,%d,%d,%d,%g,%g,%g,%g,%g\n' % (sentry,sevent.run(),
                            sevent.subrun(),sevent.event(),pred,labe,
                            softmaxarr[0],softmaxarr[1],softmaxarr[2],softmaxarr[3],softmaxarr[4])
                else:
                    dstring = '%d,%d,%d,%d,%d,%g,%d,%g\n' % (sentry,sevent.run(),sevent.subrun(),
                                                             sevent.event(),pred,pprob,labe,plab)
                fout.write(dstring)
            
                ctr+=1
                if ctr == events:
                    break
            if ctr == events:
                break
                    
            tep.next(store_entries=True,store_event_ids=True)
                    
        fout.close()
        print('Closed File')
        coord.request_stop()
        print('Coord Finished')
        coord.join(threads)
        print('Coord Join')

else:
    print('File Already Exists')
    
    #ag.DataInfo('test_40k.root')
    #dar = ag.GetKinetics('test_40k.root')
    #i=0
    #while i<5:
    #    ag.KineticGrapher(fpathC,i)
    #    i=i+1

#LABEL2 IS PHOTON?
#ag.PIDFrac2(fpath2)
#ag.PIDFrac2(fpath2)
#ag.GeomFrac(fpath2)


#ag.AnaPlot2(batchsize,2)
	

    #print('fpath1 Half')
ag.Grapher(fpath2,'true')
    #ag.Grapher(fpath,'pred')
    #print('fpath2 Full')
    #ag.Grapher(fpath2,'true')
    #ag.Grapher(fpath2,'pred')
    #print('fpath1 = Half Fracs')
    #b = ag.GeomFrac(fpath)
#ag.GrapherFSM(fpath3,4)
    #a = ag.PIDFrac(fpath)
    #print('fpath2 = Full fracs')
    #c = ag.GeomFrac(fpath2)
    #d = ag.PIDFrac(fpath2)

    #Plot 1 of type
#ag.AnaPlot(batchsize,'Muon')
    #[a,b] = ag.PIDFrac2(fpath)
    #print('')
    #print('full now')
    #[m,n] = ag.PIDFrac2(fpath2)
    #print(n)
    #ag.PIDFrac(fpath2)

#ag.GrapherANA(fpath2,0)
    #for i in range (0,5):
    #    ag.GrapherSingle(fpath2,i)
     #   ag.GrapherANA2(fpath3,i)
        
#bums = 1    
#if bums == 1:
    #asdfoij = ag.GeomFrac(fpath)
#    red = np.array([1.,0.]) ; green = np.array([1.,0.5]) ; blue = np.array([1.,0.50])
#    stops = np.array([0.,1.])
#    R.TColor.CreateGradientColorTable(2,stops,red,green,blue,1000)
#    R.gStyle.SetOptStat(0)
#    m = R.TMatrixD(5,5)
#    #n = R.TMatrixD(5,5)
#    l = R.TMatrixD(2,2)
#
#    l[1][0] = 01.53
#    l[1][1] = 97.73
##
#    l[0][0] = 98.47
#    l[0][1] = 02.27

    #m[4][0]=00.06
    #m[4][1]=01.28
    #m[4][2]=00.19
    #m[4][3]=26.75
    #m[4][4]=71.69

    #m[3][0]=00.21
    #m[3][1]=01.20
    #m[3][2]=00.09
    #m[3][3]=83.51
    #m[3][4]=14.97

    #m[2][0]=07.23
    #m[2][1]=10.28
    #m[2][2]=81.83
    #m[2][3]=00.48
    #m[2][4]=00.16

    #m[1][0]=03.31	
    #m[1][1]=70.63	
    #m[1][2]=20.20	
    #m[1][3]=03.33	
    #m[1][4]=02.50	

    #m[0][0]=93.49	
    #m[0][1]=03.96	
    #m[0][2]=02.19	
    #m[0][3]=00.33	
    #m[0][4]=00.01
	
    
    

    

    


    #i=0
    #j=0
    #while i<0:
    #    while j<0:
    #        m[i][j]=n[i][j]

    #print(m)

#    canv = R.TCanvas()
#    l.Draw("colz text")
#    h = R.TH2D(canv.GetListOfPrimitives()[0])##
##
 #   for axis in [h.GetXaxis(),h.GetYaxis()]:
        
        #axis.SetBinLabel(1,"Proton")
        #axis.SetBinLabel(2,"Pion")
        #axis.SetBinLabel(3,"Muon")
        #axis.SetBinLabel(4,"Photon")
        #axis.SetBinLabel(5,"Electron")
 #       axis.SetBinLabel(1,'Shower')
 #       axis.SetBinLabel(2,'Track')
        
  #  h.GetXaxis().SetTitle("True") ; h.GetYaxis().SetTitle("Reco")
  #  h.SetMarkerSize(2.5)
  #  h.SetMaximum(100.) ; h.SetMinimum(0.)
  #  h.Draw("colz text")
  #  raw_input()
    
    
   # end = time.time()
   # print('Run Time: %d seconds' % (end - start))
