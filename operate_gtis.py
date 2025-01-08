#updated 03.03.2018 with numpy version of in-intersection

from astropy.io import fits as pyfits
import re
from astropy.time import Time
import numpy

def Intersect2Segments(x,y):
        '''Intersects 2 segments: result is array [ok, min, max] '''
        xmin = x[0]
        xmax = x[1]
        ymin = y[0]
        ymax = y[1]
        ok = True
        nmin = max(xmin,ymin)
        nmax = min(xmax,ymax)
        if(nmin>=nmax):ok=False
        return [ok,nmin,nmax]

def Cut2Segments(x,y):
        ''' Cuts the second segment from the first one '''
        xmin = x[0] 
        xmax = x[1] 
        ymin = y[0] 
        ymax = y[1]
        
        ok = -1 # number of segments after cuting
        ans = [ok]

        if((ymin<xmax)and(ymin>xmin)and(ymax>xmin)and(ymax<xmax)): # y is totally inside x:
                ok = 2
                nmin1 = xmin
                nmax1 = ymin
                nmin2 = ymax
                nmax2 = xmax
                ans = [ok,[nmin1,nmax1],[nmin2,nmax2]] # this to return
        else: # y is not totally inside x
                if((ymin>xmin)and(ymin<xmax)):#left end of y is in x
                        ok = 1
                        ans = [ok,[xmin,ymin]]
                elif((ymax>xmin)and(ymax<xmax)):# right end of y is in x
                        ok = 1
                        ans = [ok,[ymax,xmax]]
                elif((ymax<xmin)or(ymin>xmax)): # y is outside of x and does not intersects with x
                        ok = 1
                        ans = [ok,[xmin,xmax]]
        return ans
                

def MJD2Fermi(x):
        ''' MJD to FERMI seconds '''
        t0gps = 662342413.0 
        t = Time(x, format='mjd')
        return t.gps - t0gps

def Fermi2MJD(x):
        ''' FERMI seconds to MJD '''
        t0gps = 662342413.0
        t = Time(x+t0gps, format='gps')
        return t.utc.mjd

def Intersect1WithAll( fermi_gtis, tmin, tmax   ):
        ''' Intersects all fermi_gtis with 1 user gti [tmin,tmax] '''
        #Basic idea: let us find intersection of 2 segments X=[x1,x2], Y=[y1,y2]
        #we will need to define z1 = max(x1,y1) ; z2 = min(x2,y2). Z -- is intersection of X and Y
        #Z is a real intersection if z2>z1

        starts = numpy.maximum( fermi_gtis['START'], tmin) # array [max(fermi_start1, tmin), max(fermi_start2, tmin), ... ]
        stops = numpy.minimum( fermi_gtis['STOP'], tmax  ) # array [min(fermi_stop1, tmax), min(fermi_stop2, tmax), ... ]

        #now starts and stops are candidates for intersection of fermi_gtis with user gti
        #but it can happen, that NOT all fermi_gtis intersects with user ones
        #for such "bad" candidates we will have stop<start. Let us filter these out
        lengths = stops - starts
        ok = numpy.where( lengths>0 )#indexes of all "real" intersections
        return starts[ok], stops[ok]



def UpdateGTIs(fitsfile,textfile, method='in',times_in_mjd=True):
        ''' Updates fits file with the GTIs from text file '''
        ''' method=in(default) -- to use simple intersection of text and fits GTIs'''
        ''' method=out -- to REMOVE text gtis from fits ones '''

        textgtis = []
        fitsgtis = []
        newgtis = []
        #get text gtis
        with open(textfile) as f:
                for ss in f.readlines():
                        ss = re.split(r'\ +',ss)
                        tmin = float(ss[0]) 
                        tmax = float(ss[1]) 
                        
                        if(times_in_mjd):
                                tmin = MJD2Fermi( tmin )
                                tmax = MJD2Fermi( tmax )
                        textgtis.append([tmin,tmax])
        #get fits gtis
        with pyfits.open(fitsfile) as f:
                fitsgtis1 = f[2].data.copy() #to save
                fitsgtis = f[2].data.copy() # to work

        #process gtis
        if(method == 'in'):
                new_starts = []
                new_stops = []
                for tt in range(len(textgtis)):
                        #print '[',tstarts[tt],' , ' ,tstops[tt],']'
                        t1, t2 = Intersect1WithAll( fitsgtis, textgtis[tt][0], textgtis[tt][1]   )
                        new_starts = numpy.append(new_starts, t1)
                        new_stops = numpy.append(new_stops, t2) 
                for ii in range(len(new_starts)):
                        newgtis.append( [ new_starts[ii], new_stops[ii] ] )
                        

        if(method=='out'):
                tt = textgtis[0]
                print( tt )
                # cut-out first gti from fitsgti-s
                for ff in fitsgtis:
                                res = Cut2Segments(ff,tt) # cut tt from ff
                                if(res[0]==2):
                                        newgtis.append(res[1])
                                        newgtis.append(res[2])
                                if(res[0]==1):
                                        newgtis.append(res[1])          
                #cut-out the rest from what we got on the previous step:
                for ii in range(1,len(textgtis)):
                        tt = textgtis[ii]
                        print( tt )
                        newgtis1 = []
                        for ff in newgtis:
                                res = Cut2Segments(ff,tt) # cut tt from ff
                                if(res[0]==2): 
                                        newgtis1.append(res[1]) 
                                        newgtis1.append(res[2]) 
                                if(res[0]==1): 
                                        newgtis1.append(res[1])

                        newgtis = newgtis1 


        print(( 'Text GTIs size: ', len(textgtis) ))
        print(( 'Fits GTIs size: ',fitsgtis.shape[0] ))
        print(('Resulted GTIs size: ',len(newgtis) ))

        fitsgtis1.resize(len(newgtis))
        ontime = 0
        for ii in range(len(newgtis)):
                fitsgtis1[ii] = newgtis[ii]
                ontime += newgtis[ii][1]-newgtis[ii][0]
        #replacing old fits gtis with the new ones
        with pyfits.open(fitsfile,'update') as f:
                f[2].data = fitsgtis1
                f[2].header['ONTIME'] = ontime #replace sum of all gtis just in case...

##############################################
#UpdateGTIs('test.fits','h.dat',method='out')
