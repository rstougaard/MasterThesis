import pyLikelihood
from BinnedAnalysis import *
import numpy as np
nn = '4FGL J0617.7-1715'
emin = 100
emax = 300
irf='CALDB'
edisp_bins = -3
ebin = 1
tbin = 1
optimizer = 'MINUIT'
'''
scdatafile=glob.glob(datapath+'/*SC*.fits')[0]; # this is "SC" file downloaded from Fermi/LAT site
eventsfile=str(ebin)+'_'+str(tbin)+'_events_gti.fits'
expmapfile=str(ebin)+'_'+str(tbin)+'_expMap.fits'
expcubefile=prefix+'_expCube_'+str(tbin)+'.fits'
cmapfile=str(ebin)+'_'+str(tbin)+'_SrcMap.fits'
bexpmapfile=str(ebin)+'_'+str(tbin)+'_BexpMap.fits'
model_back = str(ebin)+'_'+str(tbin)+'_back_map.fits'
model_all = str(ebin)+'_'+str(tbin)+'_model_map.fits'

cfg = BinnedConfig(edisp_bins=edisp_bins)
print( 'Will launch analysis with edisp_bins=',cfg.edisp_bins() )
analysis = binnedAnalysis (config=cfg, irfs=irf,cmap=cmapfile,bexpmap=bexpmapfile,expcube=expcubefile,srcmdl=model, optimizer=optimizer)
'''
#this is a change0
prefix = "./Rikke/"
srcmap =prefix+str(ebin)+'_'+str(tbin)+'_SrcMap.fits' 
binexpmap = prefix+str(ebin)+'_'+str(tbin)+'_BexpMap.fits'
ltcube = prefix+'J0617_expCube_'+str(tbin)+'.fits'
input_model ="src_model_const.xml"

obs = BinnedObs(srcMaps=srcmap, binnedExpMap=binexpmap, expCube=ltcube, irfs='CALDB')
like = BinnedAnalysis(obs, input_model, optimizer='NewMinuit')
#cfg = BinnedConfig(edisp_bins=edisp_bins)
#print( 'Will launch analysis with edisp_bins=',cfg.edisp_bins() )
#like = binnedAnalysis (config=cfg, irfs=irf,cmap=srcmap,bexpmap=binexpmap,expcube=ltcube,srcmdl=input_model, optimizer=optimizer)
likeobj = pyLikelihood.NewMinuit(like.logLike)
like.fit(verbosity=0, covar=True, optObject=likeobj)
TS = like.Ts(nn) #also include in output file
convergence = likeobj.getRetCode()  #also include in output file
print('Conergence = ',convergence)
#root = tree.getroot()

# Save successful bin details
#successful_bins[(emin, emax)] = writexml

#flux_tot_value = like.flux(source_name, emin=emin, emax=emax)
#flux_tot_error = like.fluxError(source_name, emin=emin, emax=emax)
arg = pyLikelihood.dArg( (emin*emax)**0.5 ) # Emin, Emax are in MeV
flux = like.model.srcs[nn].src.spectrum()(arg) *emin*emax*1.6e-6  # differential flux in erg/cm2/s ; source -- the name of the source
coeff = flux / like.flux(nn,emin,emax)

dflux = like.fluxError(nn,emin,emax)*coeff # flux error, erg/cm2/s

E = (like.energies[:-1] + like.energies[1:]) / 2.
nobs = like.nobs
geometric_mean = (emin*emax)**0.5

data = np.column_stack((
                        geometric_mean, 
                        flux, 
                        dflux,
                        emin, 
                        emax
                        ))
# Define a header for clarity in the text file
header = "geometric_mean flux flux_error emin emax"

# Save the data to a text file. Adjust the format (here '%f') if you need different precision.
np.savetxt("output_denysdata_rikkeanalysis.txt", data, header=header, fmt='%s')