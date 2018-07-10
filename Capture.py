import RPi.GPIO as GPIO
import time
import Adafruit_GPIO.SPI as SPI
import Adafruit_MCP3008
import matplotlib.pyplot as plt
import csv
import sys
import os
from time import sleep
import numpy as np
import pandas as pd
from scipy import signal
import math
from numpy import NaN, Inf, arange, isscalar, asarray, array

GPIO.setmode(GPIO.BCM)

GPIO.setup(6, GPIO.IN, pull_up_down=GPIO.PUD_UP)
print('code is running...')

while True:
    input_state = GPIO.input(6)
    
    if input_state == False:
        csvfile = "elbin.csv"
        # Hardware SPI configuration:
        SPI_PORT   = 0
        SPI_DEVICE = 0
        mcp = Adafruit_MCP3008.MCP3008(spi=SPI.SpiDev(SPI_PORT, SPI_DEVICE))
        print('Reading MCP3008 values, press Ctrl-C to quit...')
        x = 5000
        t = 1
        while x!=0:
            values =0
            values= mcp.read_adc(0)
            print('| {0:>4} |'.format(values))
            data=[t,values]
            with open(csvfile, "a")as output:
                writer = csv.writer(output, delimiter=",", lineterminator = '\n')
                writer.writerow(data)
            x=x-1;
            t=t+1;
            time.sleep(0.001)
        x = []
        y = []
        with open('elbin.csv','r') as csvfile:
            plots = csv.reader(csvfile, delimiter=',')
            for row in plots:
                x.append(int(row[0]))
                y.append(int(row[1]))
        plt.plot(x,y, label='Loaded from file!')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('ECG Graph\nCheck it out')
        plt.legend()
        x = []
        y = []
        with open('elbin.csv','r') as csvfile:
            plots = csv.reader(csvfile, delimiter=',')
            for row in plots:
                x.append(int(row[0]))
                y.append(int(row[1]))
        def butter_lowpass(cutoff, fs, order=4):
            nyq = 0.5 * fs
            normal_cutoff = cutoff / nyq
            b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
            return b, a
        def butter_lowpass_filter(data, cutoff, fs, order=4):
            b, a = butter_lowpass(cutoff, fs, order=order)
            y = signal.filtfilt(b, a, data)
            return y
        def butter_highpass(cutoff, fs, order=3):
            nyq = 0.5 * fs
            normal_cutoff = cutoff / nyq
            b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
            return b, a
        def butter_highpass_filter(data, cutoff, fs, order=3):
            b, a = butter_highpass(cutoff, fs, order=order)
            y = signal.filtfilt(b, a, data)
            return y
        def peakdet(v, delta, x = None):
            maxtab = []
            mintab = []
            if x is None:
                x = arange(len(v))
            v = asarray(v)
            if len(v) != len(x):
                sys.exit('Input vectors v and x must have same length')
            if not isscalar(delta):
                sys.exit('Input argument delta must be a scalar')
            if delta <= 0:
                sys.exit('Input argument delta must be positive')
            mn, mx = Inf, -Inf
            mnpos, mxpos = NaN, NaN
            lookformax = True
            for i in arange(len(v)):
                this = v[i]
                if this > mx:
                    mx = this
                    mxpos = x[i]
                if this < mn:
                    mn = this
                    mnpos = x[i]
                if lookformax:
                    if this < mx-delta:
                        maxtab.append((mxpos, mx))
                        mn = this
                        mnpos = x[i]
                        lookformax = False
                else:
                    if this > mn+delta:
                        mintab.append((mnpos, mn))
                        mx = this
                        mxpos = x[i]
                        lookformax = True
            return array(maxtab), array(mintab)
        fps = 340
        0 #Sampling rate
        filtered_sine = butter_highpass_filter(y,1,fps)
        filtered_sine1 = butter_lowpass_filter(filtered_sine,20,fps)
        delt=int((max(filtered_sine1)-min(filtered_sine1))*.55)
        plt.subplot(211)
        plt.plot(x,y)
        plt.title('Original ECG signal')
        plt.subplot(212)
        plt.plot(range(len(filtered_sine1)),filtered_sine1)
        plt.title('Filtered ECG signal')
        
        if __name__=="__main__":
            maxtab, mintab = peakdet(filtered_sine1,delt)
        print('heart_rate | epoc_duration | ppos | qpos | rpos| spos | tpos | P_wave | Q_wave | qrs_complex | R_wav | S_wav | T_wav | PR_int | QT_int | PR_seg | ST_seg')
        heart_rate1=0
        epoc_duration1=0
        ppos1=0
        qpos1=0
        rpos1=0
        spos1=0
        tpos1=0
        P_wave1=0
        Q_wave1=0
        qrs1=0
        R_wave1=0
        S_wave1=0
        T_wave1=0
        PR_interval1=0
        QT_interval1=0
        PR_segment1=0
        ST_segment1=0
        k=1
        for i in range(0,10):
            t=math.floor(maxtab[i][0])
            s=math.floor(maxtab[i+1][0])
            half=int((s-t)/2)
            #r wave peaks
            epoc=filtered_sine1[t+half-40:s+half+40]
            rpeak=0
            for i in range(0,len(epoc)):
                if epoc[i]>rpeak:
                    rpeak=epoc[i]
                    rpos=i
            #Q wave peaks
            i=rpos
            while(epoc[i]>epoc[i-1]):
                i=i-1
            qpeak=epoc[i]
            qpos=i

            #average
            avg=0
            for i in range(1,50):
                avg=epoc[i]+avg
            avg=avg/50
 
            #S wave peaks
            i=rpos
            while(epoc[i]>epoc[i+1]):
                i=i+1
            speak=epoc[i]
            spos=i
            i=qpos
            j=spos
            qr=[]
            g=0
            while(epoc[i]<epoc[i-1] and g<=9):
                i=i-1
                g=g+1
            printerval2=i
            l=0
            while(epoc[j]<epoc[j+1] and l<=9):
                j=j+1
                l=l+1
            stsegment1=j
            qrs=((j-i)/fps)*1000
            R_wave=((spos-qpos)/fps)*1000
            Q_wave=(((qpos-i)+((qpos-i-1)/2))/fps)*1000
            S_wave=(((j-spos)+((j-spos-1)/2))/fps)*1000
            while(i!=j):
                qr.append([i, epoc[i]])
                i=i+1

            #P wave peaks
            ppeak=0
            for i in range(50,qpos):
                if epoc[i]>ppeak:
                    ppeak=epoc[i]
                    ppos=i
            i=ppos
            j=i
            pp=[]
            if(epoc[ppos]>avg):
                while((epoc[i]>epoc[i-1] or epoc[i]>epoc[j+8] )and i>(ppos-25)):
                    pp.append([i,epoc[i]])
                    i=i-1
                while(((epoc[j]>epoc[j+1] or epoc[j]>epoc[j+8])and i<(ppos+24)) and j<(printerval2-2)):
                    pp.append([j,epoc[j]])
                    j=j+1
            else:
                i=0
                j=o
                pp.append([0,0])
            printerval1=i
            prsegment=j
            P_wave=((j-i)/fps)*1000

            #T wave peaks
            tpeak=0
            for i in range(spos,len(epoc)):
                if epoc[i]>tpeak:
                    tpeak=epoc[i]
                    tpos=i
            i=tpos
            j=i
            tt=[]
            if(epoc[tpos]>avg):
                while(((epoc[i]>epoc[i-1] or epoc[i]>epoc[j+9] )and i>(tpos-55)) and i>(stsegment1-2)):
                    tt.append([i,epoc[i]])
                    i=i-1
                while((epoc[j]>epoc[j+1] or epoc[j]>epoc[j+7])and i<(tpos+28)):
                    tt.append([j,epoc[j]])
                    j=j+1
            else:
                i=0
                j=0
                tt.append([0,0])
            stsegment2=i
            qtinterval=j
            T_wave=((j-i)/fps)*1000
            v=(s-t)/fps
            PR_interval=((printerval2-printerval1)/fps)*1000
            QT_interval=((qtinterval-printerval2)/fps)*1000
            PR_segment=((printerval2-prsegment)/fps)*1000
            ST_segment=((stsegment2-stsegment1)/fps)*1000
            heart_rate=int(60/v)
            epoc_duration=int(v*1000)
            print('----------Next Epoc-----------')
            #print ('The Heart Rate is',60/v,'bpm')
            #print ('The Epoc Duaraion is',v, 'sec')
            #print ('P Peak Position=',ppos,'ms')
            #print ('Q Peak Position=',qpos,'ms')
            #print ('R Peak Position=',rpos,'ms')
            #print ('S Peak Position=',spos,'ms')
            #print ('T Peak Position=',tpos,'ms')
            #print ("QRS Complex Duration=",qrs,"ms")
            #print ('P Wave Duration=',P_wave,'ms')
            #print ('Q Wave Duration=',Q_wave,'ms')
            #print ('R Wave Duration=',R_wave,'ms')
            #print ('S Wave Duration=',S_wave,'ms')
            #print ('T Wave Duration=',T_wave,'ms')
            #print ('PR Interval Duration=',PR_interval,'ms')
            #print ('QT Interval Duration=',QT_interval,'ms')
            #print ('PR Segment Duration=',PR_segment,'ms')
            #print ('ST Segment Duration=',ST_segment,'ms')
            example_ECG1 = [[heart_rate, epoc_duration,ppos,qpos,rpos,spos,tpos, P_wave, Q_wave, qrs, R_wave,S_wave, T_wave, PR_interval, QT_interval, PR_segment, ST_segment]]
            print(heart_rate, epoc_duration,int(ppos),int(qpos),int(rpos),int(spos),int(tpos), int(P_wave), int(Q_wave), int(qrs), int(R_wave),int(S_wave), int(T_wave), int(PR_interval), int(QT_interval), int(PR_segment), int(ST_segment))
            heart_rate1=(heart_rate1*(k-1)+heart_rate)/k
            epoc_duration1=(epoc_duration1*(k-1)+epoc_duration)/k
            ppos1=(ppos1*(k-1)+ppos)/k
            qpos1=(qpos1*(k-1)+qpos)/k
            rpos1=(rpos1*(k-1)+rpos)/k
            spos1=(spos1*(k-1)+spos)/k
            tpos1=(tpos1*(k-1)+tpos)/k
            P_wave1=(P_wave1*(k-1)+P_wave)/k
            Q_wave1=(Q_wave1*(k-1)+Q_wave)/k
            qrs1=(qrs1*(k-1)+qrs)/k
            R_wave1=(R_wave1*(k-1)+R_wave)/k
            S_wave1=(S_wave1*(k-1)+S_wave)/k
            T_wave1=(T_wave1*(k-1)+T_wave)/k
            PR_interval1=(PR_interval1*(k-1)+PR_interval)/k
            QT_interval1=(QT_interval1*(k-1)+QT_interval)/k
            PR_segment1=(PR_segment1*(k-1)+PR_segment)/k
            ST_segment1=(ST_segment1*(k-1)+ST_segment)/k

            plt.figure(i)

            plt.subplot(221)
            plt.plot(epoc, label='selection of single Epoc with Peaks')
            plt.scatter(rpos, rpeak, color='blue')
            plt.scatter(qpos, qpeak, color='blue')
            plt.scatter(ppos, ppeak, color='blue')
            plt.scatter(spos, speak, color='blue')
            plt.scatter(tpos, tpeak, color='blue')
            plt.legend()

            plt.subplot(222)
            plt.plot(epoc, label='QRS complex Detection')
            plt.scatter(array(qr)[:,0], array(qr)[:,1], color='red')
            plt.legend()

            plt.subplot(223)
            plt.plot(epoc, label='P_wave Detection')
            plt.scatter(array(pp)[:,0], array(pp)[:,1], color='red')
            plt.legend()

            plt.subplot(224)
            plt.plot(epoc, label='T_wave detection')
            plt.scatter(array(tt)[:,0], array(tt)[:,1], color='red')
            plt.legend()
            #plt.savefig("signal.png")
            k=k+1
            
        print('===============================================average value=================================')
        print(int(heart_rate1), int(epoc_duration1),int(ppos1),int(qpos1),int(rpos1),int(spos1),int(tpos1), int(P_wave1), int(Q_wave1), int(qrs1), int(R_wave1),int(S_wave1), int(T_wave1), int(PR_interval1), int(QT_interval1), int(PR_segment1), int(ST_segment1))

        from sklearn.model_selection import train_test_split
        np.set_printoptions(precision=2)
        ECG = pd.read_table('ECG_dataset.txt')
        feature_names_ECG = ['P_wave', 'Q_wave', 'QRS_complex', 'R_wave', 'T_wave', 'heart_rate', 'ST_segment', 'QT_interval', 'PR_interval']
        X_ECG = ECG[feature_names_ECG]
        y_ECG = ECG['diseases_label']
        target_names_ecg = ['Normal','Bradycardia', 'Tachycardia']
        print('  ')
        print('============Machine Learning Output=====================')
        print('  ')

        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(X_ECG, y_ECG, test_size=0.3, random_state=0)

        #Scaling data
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import cross_val_score
        sc = StandardScaler()
        sc.fit(X_train)
        X_train_std = sc.transform(X_train)
        X_test_std = sc.transform(X_test)

        from sklearn.svm import SVC
        svm = SVC(kernel='rbf', random_state=0, gamma=.10, C=1.0)
        svm.fit(X_train_std, y_train)
        example_ECG = [[P_wave1, Q_wave1, qrs1, R_wave1, T_wave1, heart_rate1, ST_segment1, QT_interval1, PR_interval1]]
        #example_ECG = [[114,10,165,54,230,65,250,34,104,300,144]]
        #example_ECG = [[190,83,200,111,222,55,10,55,111,120,115]]
        #example_ECG = [[33,5,40,25,25,115,95,165,145]]
        example_ECG_scaled = sc.transform(example_ECG)
        print(svm.predict(example_ECG_scaled))
        print('Classified Output for given signal is',
              target_names_ecg[svm.predict(example_ECG_scaled)[0]-1])
        plt.show()


