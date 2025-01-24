#Base
import argparse
from glob import glob
from os.path import join
import json, os, time, sys
import numpy as np
import xarray as xr
import pandas as pd
import scipy.stats as st
from tqdm.notebook import trange, tqdm
import ray

#Plot
PlotDir = '/home/david.wyrick/projects/zap-n-zip/plots'
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from ssm.plots import gradient_cmap, white_to_color_cmap

#Project
from tbd_eeg.data_analysis.eegutils import EEGexp
import tbd_eeg.data_analysis.Utilities.utilities as util

#SSM
import ssm
from sklearn.model_selection import StratifiedKFold

#Allen
from allensdk.brain_observatory.ecephys.lfp_subsampling.subsampling import remove_lfp_offset
from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache

mcc = MouseConnectivityCache(resolution=10)
str_tree = mcc.get_structure_tree()

#Read in allen CCF
ccfsum = pd.read_csv('/home/david.wyrick/projects/zap-n-zip/data/mouse_connectivity/ontology_v2.csv')

#User
from util import *

#CCM
from delay_embedding import ccm
from delay_embedding import evaluation as E
from delay_embedding import helpers as H
from delay_embedding import surrogate as S
import ray
import kedm
ProjDir = '/home/david.wyrick/projects/zap-n-zip/'
ServDir = '/data/projects/DOI/'
PlotDir = os.path.join(ProjDir,'plots','ccm')

##===== ============================ =====##
##===== Parse Command Line Arguments =====##
parser = argparse.ArgumentParser(description='FCF')

##===== Data Options =====##
parser.add_argument('--mID',type=str, default='G62FFF7TT',
                    help='mouse to perform analysis on')

parser.add_argument('--time_bin_ms',type=int, default=100,
                    help='time_bin_ms')

parser.add_argument('--delay',type=int, default=1,
                    help='tau')

parser.add_argument('--rand_proj',type=int, default=0,
                    help='random projection of delay vectors?')

parser.add_argument('--use_kedm',type=int, default=1,
                    help='Use Amin or kEDM software')

parser.add_argument('--nKfold',type=int, default=10,
                    help='# of folds')

parser.add_argument('--zscore',type=int, default=0,
                    help='zscore spike counts?')

parser.add_argument('--fr_thresh',type=float, default=2.5,
                    help='Firing rate threshold for neurons to include in analysis')

parser.add_argument('--rand_proj_dist',type=str, default='normal',
                    help='Distribution to draw random matrix')

dFC_threshold = 0.02
from scipy.signal import find_peaks
from kneed import KneeLocator

def determine_optimal_FCF(FCF_takens,takens_range):
    nTakens, N, N = FCF_takens.shape
    FCF_optimal = np.zeros((N,N))
    complexity = np.zeros((N,N))
    
    for i in range(N):
        for j in range(N):
            if i == j:
                continue

            ##============================================= First pass 
            ##===== Find the "complexity" =====##
            # Elbow method, or when the FCF first reaches 95% of it's maximum
            fcf_ij = FCF_takens[:,i,j]

            if all(np.isnan(fcf_ij)):
                complexity[i,j] = np.nan
                FCF_optimal[i,j] = np.nan
                continue
            elif np.nanmax(fcf_ij) < 0:
                indy = np.nanargmax(fcf_ij)
                embedding_dim = takens_range[indy]
                embeddedness = fcf_ij[indy]
            else:
                indy = np.where(fcf_ij >= 0.95*np.nanmax(fcf_ij))[0][0]
                embedding_dim = takens_range[indy]
                embeddedness = fcf_ij[indy]                
            
            complexity[i,j] = embedding_dim
            FCF_optimal[i,j] = embeddedness
            
            ##============================================= second pass
            takens_ij = embedding_dim
            if np.isnan(takens_ij):
                complexity[i,j] = np.nan
                FCF_optimal[i,j] = np.nan
                continue
            
            #Plot interpolated fcf values
            max_fcf = np.max(fcf_ij)

            #Plot select takens dimension based on 95% max
            takens_idx = np.where(takens_range == embedding_dim)[0]
            elbow_indices = [takens_idx[0]]
            
            #Plot takens dimension based on max fcf value
            max_fcf_idx = np.argmax(fcf_ij)
            
            #Find if there are local maxima
            peak_indices, _ = find_peaks(fcf_ij)
            
            if all(fcf_ij < 0):
                elbow_indices.append(max_fcf_idx)
                elbow_indices = np.array(elbow_indices)
            else:
                #Plot takens dimension based on knee method 
                kneedle = KneeLocator(np.arange(len(fcf_ij)), fcf_ij, S=20, curve="concave", direction="increasing",online=True)
                if kneedle.knee is not None:
                    elbow_indices.append(kneedle.knee)
                
                if len(peak_indices) > 1:
                    #Sort potential FCF values
                    elbow_indices = np.concatenate((peak_indices,elbow_indices))
                else:
                    # print('no peaks')/
                    #Find elbow based on FCF percent change 
                    # tmp = pd.DataFrame(fcf_ij).pct_change().values[:,0]
                    # kk = np.where(np.abs(tmp) < 0.01)[0][0]
                    # ax.vlines(takens_50_1[kk],fcf_ij[kk],1,ls='--',color=cc[3],label='Elbow based on % change')
                    # elbow_indices.append(kk)
                    elbow_indices = np.array(elbow_indices)
            
            #Sort potential FCF values
            fcf_sub = fcf_ij[elbow_indices]
            fcf_sub_sorted = np.sort(fcf_sub)
            ei_sorted = elbow_indices[np.argsort(fcf_sub)]
            
            #For now, choose the highest FCF value
            fcf_select = fcf_sub_sorted[-1]
            ei_select = ei_sorted[-1]
            dim_select = takens_range[ei_select]
            
            #Loop through potential FCF values and find the lowest dimensionality that results in high FCF
            prev_fcf_max = fcf_select; dFC2 = []
            nElbows = len(elbow_indices)
            for e in range(1,nElbows):
                ei = -1*(e+1)

                #Calculate absolute difference between selected and current FCF
                dFC = np.abs(fcf_select - fcf_sub_sorted[ei])
                dFC2.append(np.abs(prev_fcf_max - fcf_sub_sorted[ei]))
                
                # print(fcf_select,fcf_sub_sorted[ei],prev_fcf_max,dFC,dFC2)
                #If the relative increase in FCF is less than 0.05 and the dimensionality is lower, 
                if (dFC < dFC_threshold) & (ei_sorted[ei] < ei_select) & (fcf_sub_sorted[ei] > 0) & (np.all(np.array(dFC2) < dFC_threshold)):
                    # print('.')
                    prev_fcf_max = fcf_select
                    fcf_select = fcf_sub_sorted[ei]
                    ei_select = ei_sorted[ei]
                    dim_select = takens_range[ei_select]
                # print(fcf_select,dim_select,fcf_select,prev_fcf_max, dFC2)

            
            FCF_optimal[i,j] = fcf_select
            complexity[i,j] = dim_select
    return FCF_optimal, complexity

def plot_CCM_results(corr_mat,fcf,flow,takens,fname_suffix,title=None,mask=None,save=True,rBool=True,ticks=None,labels=None,boundaries=None):

    fig, axes = plt.subplots(1,5,figsize=(20,4))#,gridspec_kw={'wspace': 0.1,'hspace':0.1})
    plt.suptitle(title,fontsize=16,fontweight='bold',y=0.96)

    ax = axes[0]
    ax.set_title('Correlation Matrix',fontsize=14,fontweight='bold')
    vmax = np.round(np.nanpercentile(corr_mat.ravel(),99),2)
    vmin = np.round(np.nanpercentile(corr_mat.ravel(),1),2)
    sns.heatmap(corr_mat,square=True,annot=False,cmap='coolwarm',vmin=vmin,center=0, vmax=vmax,ax=ax,cbar_kws={'shrink':0.5,'label': 'Correlation','ticks':[vmin,vmax]},rasterized=rBool) #cmap=sns.color_palette("vlag", as_cmap=True),center=0,ax=ax,cbar_kws={'shrink':0.5,'label': 'Correlation'})

    ax = axes[1]
    ax.set_title('Functional causal flow',fontsize=14,fontweight='bold')
    vmax = np.round(np.nanpercentile(fcf.ravel(),99),2)
    sns.heatmap(fcf,square=True,annot=False,cmap='viridis',vmin=0, vmax=vmax,ax=ax,mask=mask,cbar_kws={'shrink':0.5,'label': 'FCF','ticks':[0,vmax]},rasterized=rBool)

    # ax = axes[2]
    # vmax = np.round(np.nanpercentile(fcf - corr_mat,99),2)
    # vmin = np.round(np.nanpercentile(fcf - corr_mat,0.5),2)
    # ax.set_title('FCF - Correlation',fontsize=14,fontweight='bold')
    # sns.heatmap(fcf - corr_mat,square=True,annot=False,cmap='RdBu_r',center=0, vmin=vmin,vmax=vmax,ax=ax,mask=mask,cbar_kws={'shrink':0.5,'label': 'FCF - Corr','ticks':[vmin,0,vmax]},rasterized=rBool)

    ax = axes[2]
    ax.set_title('Directionality',fontsize=14,fontweight='bold')
    if mask is None:
        dir_mask = None
    else:
        dir_mask = mask | mask.T
    vmax = np.round(np.nanpercentile(flow,97.5),2); vmin = np.round(np.nanpercentile(flow,2.5),2) #-1*vmax
    sns.heatmap(flow,square=True,annot=False,cmap='RdBu_r',vmin=vmin,vmax=vmax,mask=dir_mask,ax=ax,cbar_kws={'shrink':0.5,'label': 'Directionality','ticks':[vmin,0,vmax]},rasterized=rBool)
    
    ax = axes[3]
    ax.set_title('Complexity',fontsize=14,fontweight='bold')
    vmax = np.round(np.nanpercentile(takens,99),2)
    sns.heatmap(takens,square=True,annot=False,cmap='rocket',ax=ax,mask=mask,vmax=vmax,vmin=0,cbar_kws={'shrink':0.5,'label': 'Complexity','ticks':[0,vmax]},rasterized=rBool)

    if ticks is None:
        for ax in axes[:-1]:
            ax.set_xticks([]);ax.set_yticks([])
    else:
        for ii, ax in enumerate(axes):
            ax.set_xticks(ticks);ax.set_yticks(ticks)
            ax.set_xticklabels(labels,rotation=90)
            if ii == 0:
                ax.set_yticklabels(labels)
            else:
                ax.set_yticklabels([])
        
    if boundaries is not None:
        for ii, ax in enumerate(axes[:-1]):
            if (ii == 0) | (ii == 2):
                c = 'k'
            else:
                c = 'w'
            ax.vlines(boundaries,*ax.get_ylim(),color=c,lw=0.5,alpha=0.5)
            ax.hlines(boundaries,*ax.get_xlim(),color=c,lw=0.5,alpha=0.5)

    ax = axes[4]
    ax.autoscale(tight=True)

    x = np.abs(corr_mat.ravel())
    y = fcf.ravel()
    if mask is None:
        mask2 = (~np.isnan(x)) & (~np.isnan(y)) &(~np.isinf(x)) & (~np.isinf(y))
    else:
        mask2 = (~np.isnan(x)) & (~np.isnan(y)) &(~np.isinf(x)) & (~np.isinf(y)) & (~mask)
    using_datashader(ax, x[mask2], y[mask2])
    ax.plot([-0.5,1],[-0.5,1],'-k')
    ax.set_xlim([-0.5,1])
    ax.set_ylim([-0.5,1])
    ax.set_xticks(np.arange(-0.5,1.1,0.25)); ax.set_xticklabels(np.arange(-0.5,1.1,0.25))
    ax.set_yticks(np.arange(-0.5,1.1,0.25)); ax.set_yticklabels(np.arange(-0.5,1.1,0.25))
    ax.vlines(0,-0.5,1,color='r'); ax.hlines(0,-0.5,1,color='r')
    ax.set_xlabel('Correlation')
    ax.set_ylabel('Functional causal flow')

    if save:
        plt.savefig(os.path.join(PlotDir,f'FCF_probe_{fname_suffix}.pdf'))
        
        
group_names = ['MO-L','MO-R','SS-L','SS-R','VIS-L','VIS-R','RS-L','RS-R']
order_areas = [[15,21],[-15,-21],[29,36,43,50,57,64,71,78],[-29,-36,-43,-50,-57,-64,-71,-78],
                [129,136,143,150,157,164,171,178],[-129,-136,-143,-150,-157,-164,-171,-178],
                [249,255,261,268,275],[-249,-255,-261,-268,-275]]

def get_plotting_stuff(region_ids):
    
    nBlocks = len(region_ids)
    #For plotting purposes
    tickmarks_group = []
    ticklabels_group = []
    boundaries_group = [0]
    for group_name, aIDs_group in zip(group_names,order_areas):

        cntr = 0
        for aID in aIDs_group:
            pos = np.where(region_ids== aID)[0]
            if len(pos) > 0:
                cntr += len(pos)
        tickmarks_group.append(boundaries_group[-1]+cntr/2)
        boundaries_group.append(boundaries_group[-1]+cntr)
        ticklabels_group.append(group_name)

    ticklabels = []
    for aID in region_ids:
        tmp = ccfsum.loc[ccfsum['id'] == np.abs(aID)]['abbreviation'].values[0]
        if aID < 0:
            areaname = '{}_R'.format(tmp)
        else:
            areaname = '{}_L'.format(tmp)
        ticklabels.append(areaname)
    tickmarks = np.arange(nBlocks) 
    
    return (tickmarks, ticklabels, tickmarks_group, ticklabels_group, boundaries_group)
    
if __name__ == '__main__':

    # Parse the arguments
    args = parser.parse_args()
    mouse_name = args.mID 
    mID = args.mID
    time_bin_ms = args.time_bin_ms
    time_bin = time_bin_ms/1000
    delay = args.delay
    
    rand_proj = bool(args.rand_proj)
    zscore = bool(args.zscore)
    use_kedm = bool(args.use_kedm)
    nKfold = args.nKfold
    fr_thresh = args.fr_thresh
    rand_proj_dist = args.rand_proj_dist
    
    base_dir = '/data/Niell' #local
    rec_name = f'DOI'

    #Create directory for saving to
    TempDir = os.path.join(ServDir,'results','ccm',mID)
    if not os.path.exists(TempDir):
        os.makedirs(TempDir)

    tmp_list = sorted(glob(join(TempDir,f'{rec_name}_run_*')))
    if len(tmp_list) == 0:
        curr_run = 0
    else:
        last_run = int(tmp_list[-1][-1])
        curr_run = last_run+1
    folder = f'{rec_name}_run_{curr_run}'

    SaveDir = os.path.join(TempDir,folder)
    if not os.path.exists(SaveDir):
        os.makedirs(SaveDir)
        
    PlotDir = os.path.join(SaveDir,'plots')
    if not os.path.exists(PlotDir):
        os.makedirs(PlotDir)

    #Save model parameters
    args_dict = args.__dict__
    args_dict['rec_name'] = rec_name
    args_dict['SaveDir'] = SaveDir
    with open(join(SaveDir,f"ccm_parameters_run_{curr_run}.json"), "w") as outfile:
        json.dump(args_dict, outfile)

    #Let's loop over taken's dimension to see what the optimal embedding dimension is
    
    import h5py
    with h5py.File(os.path.join(base_dir,f'dfof_bpf_avg-per-CCF-area_{mID}.h5'),'r') as h5file:
        # A list of numpy arrays of size TxN; list is of length 8, which corresponds to each experimental condition
        data_list = list(h5file['avg_trace_bpf_list']) # mean dfof activity for a given ccf area, bandpass filtered
        # avg_trace_list = list(h5file['avg_trace_list'])         # mean dfof activity for a given ccf area

        #List of condition names 
        filename_list = []
        for ii in range(8):
            tmp = str(np.array(h5file[f'uniqID_s{ii}']))
            filename_list.append(tmp[2:-1])
        
        brainmask = np.array(h5file['brainmask'])    #Brain mask of exposed cortical surface
        dorsalMap = np.array(h5file['dorsalMap'])    #Cortical surface with Allen CCF overlaid
        region_ids = np.array(h5file['region_ids'])  #Region IDs corresonding to mean activity in avg_trace_list
        running_list = list(h5file['running_list'])  #Running speed per condition
        
    tickmarks, ticklabels, tickmarks_group, ticklabels_group, boundaries_group =  get_plotting_stuff(region_ids)
    processes = []
    import pdb;pdb.set_trace()
    #Loop over different time blocks
    for ii, X in enumerate(data_list):

        T,N = X.shape
        print('\t# of time points and neurons: ',X.shape)
        t0 = time.perf_counter()

        if use_kedm:
            takens_range = np.arange(1,31)
            nTakens = len(takens_range)
            FCF_takens = np.zeros((nTakens,N,N))
            for jj in trange(1,31):
                edims = np.repeat(jj,N)
                FCF_takens[jj-1] = kedm.xmap(X,edims,tau=1,Tp=0)
        else:
            takens_range = np.concatenate((np.arange(1,11),np.arange(12,31,2)))
            #Parallelize
            ray.init(num_cpus=26,include_dashboard=True, dashboard_port=5432)
            X_id = ray.put(X)

            #Create progress bar

            num_ticks = len(takens_range)
            pb = H.ProgressBar(num_ticks); actor = pb.actor

            ## Loop over taken's dimension
            obj_list = [ccm.FCF.remote(X=X_id,delay=delay,dim=dim,rand_proj=rand_proj,n_neighbors=dim+1,rand_proj_dist=rand_proj_dist) for dim in takens_range]

            #Start a series of remote Ray tasks 
            processes = [fcf.calculate_connectivity.remote(pba = actor) for fcf in obj_list]

            #And then print progress bar until done
            pb.print_until_done()

            #Initiate parallel processing
            FCF_takens = np.array(ray.get(processes))
            ray.shutdown()

        tE = (time.perf_counter() - t0)/60

        #Save!!
        fsuffix = f'{filename_list[ii]}_{time_bin_ms}ms-bins_tau-{delay}'
        if zscore:
           fsuffix = f'{fsuffix}_z' 
        if rand_proj:
            fsuffix = f'{fsuffix}_randproj'

        np.save(os.path.join(SaveDir,f'FCF_{fsuffix}.npy'),FCF_takens)
        
        

    print('Done!')
    
    #Preallocate arrays
    dFC_threshold = 0.02
    takens_50 = np.arange(1,31)
    nCond = len(filename_list)

    correlation = np.full((nCond,N,N),np.nan)
    FCF_optimal = np.full((nCond,N,N),np.nan)
    complexity = np.full((nCond,N,N),np.nan)
    directionality = np.full((nCond,N,N),np.nan)
    FCF_takens2 = np.full((nCond,len(takens_50),N,N),np.nan)

    for ii, X in enumerate(data_list):

        t0 = time.perf_counter()
        print(filename_list[ii])

        T,N = X.shape
        print('\t# of time points and neurons: ',X.shape)
        
        #Load
        fsuffix = f'{filename_list[ii]}_{time_bin_ms}ms-bins_tau-{delay}'
        if zscore:
           fsuffix = f'{fsuffix}_z' 
        if rand_proj:
            fsuffix = f'{fsuffix}_randproj'

        FCF_takens = np.load(os.path.join(SaveDir,f'FCF_{fsuffix}.npy'))

        if np.sum(np.isinf(FCF_takens)) > 0:
            indy = np.where(np.isinf(FCF_takens))
            FCF_takens[indy] = np.nan

        # takens_range = np.concatenate((np.arange(1,11),np.arange(12,31,2)))
        mean_dim = 15
        tShift = delay*(mean_dim-1)    #Max time shift
        tDelay = T - tShift        #Length of delay vectors
        test_ratio = 0.1

        #How much data are we going to try and reconstruct?
        tTest = np.max([1.0,np.min([np.floor(test_ratio*tDelay),tDelay-tShift-1.0])]).astype(int)

        #Get indices for training and test datasets
        iStart = tDelay - tTest; iEnd = tDelay
        test_indices = np.arange(iStart,iEnd)

        spks_target = X[test_indices]
    
        #Calculate correlation matrix
        correlation[ii] = E.correlation_FC(spks_target,transform='identity')
        FCF_optimal[ii], complexity[ii] = determine_optimal_FCF(FCF_takens,takens_range)
        directionality[ii] = FCF_optimal[ii] - FCF_optimal[ii].T

        titlestr = f'{rec_name} {filename_list[ii]}\n {time_bin_ms} ms bins, delay = {delay}, zscore = {zscore}, rand_proj = {rand_proj}'
        plot_CCM_results(correlation[ii],FCF_optimal[ii],directionality[ii],complexity[ii],fsuffix,title=titlestr,mask=None,save=True,rBool=True,ticks=tickmarks,labels=ticklabels,boundaries=boundaries_group)
        
        tE = (time.perf_counter() - t0)/60
        print('{}  {:.2f} mins'.format(filename_list[ii],tE), end=', ')
    print('Saving') #15minwindows

    #Save!!
    fsuffix = f'{rec_name}_{time_bin_ms}ms-bins_tau-{delay}'
    if zscore:
        fsuffix = f'{fsuffix}_z' 
    if rand_proj:
        fsuffix = f'{fsuffix}_randproj'

    # np.savez(os.path.join(SaveDir,f'results_rand_proj_{rec_name}_{time_bin_ms}ms-bins_tau-{delay}.npz'),FCF_takens=FCF_takens2,FCF=FCF_optimal,complexity=complexity,directionality=directionality,correlation=correlation)
    np.savez(os.path.join(SaveDir,f'results_{fsuffix}.npz'),FCF_takens=FCF_takens2,FCF=FCF_optimal,complexity=complexity,directionality=directionality,correlation=correlation)
    print('Done!')

    cmap = np.concatenate((sns.color_palette('Blues',6),sns.color_palette('Reds',2)))
    def calculate_average(mat,filename_list,b, axis=0,ax=None):
        tmp_list = []
        for jj, fname in enumerate(filename_list):
            tmp = np.nanmean(mat[jj],axis=axis)
            tmp_list.append(tmp)
        mat_j = np.array(tmp_list)
        mat_bluepill = np.mean(mat_j[:b],axis=0)
        mat_redpill = np.mean(mat_j[b:],axis=0)
        if ax is not None:
            sns.kdeplot(mat_bluepill,ax=ax,color=cmap[b-1],ls=ls,label='Non-psi average',lw=2)
            sns.kdeplot(mat_redpill,ax=ax,color=cmap[-1],ls=ls,label='Psi average',lw=2)
        return mat_bluepill, mat_redpill

    fig, axes = plt.subplots(1,4,figsize=(16,4))

    for jj, fname in enumerate(filename_list):

        ls = '-'
        lw = 1.5
        zo = 0

        fcf_tmp = FCF_optimal[jj]
        tmp = np.nanmean(fcf_tmp,axis=0)
        sns.kdeplot(tmp,ax=axes[0],color=cmap[jj],ls=ls,label=fname,lw=lw,zorder=zo)

        dir_tmp = fcf_tmp - fcf_tmp.T
        tmp = np.nanmean(dir_tmp,axis=0)
        sns.kdeplot(tmp,ax=axes[1],color=cmap[jj],ls=ls,label=fname,lw=lw,zorder=zo)

        tmp = np.nanmean(complexity[jj],axis=0)
        sns.kdeplot(tmp,ax=axes[2],color=cmap[jj],ls=ls,label=fname,lw=lw,zorder=zo)

        tmp = np.nanmean(complexity[jj],axis=1)
        sns.kdeplot(tmp,ax=axes[3],color=cmap[jj],ls=ls,label=fname,lw=lw,zorder=zo)

    ax_in = axes[0].inset_axes([0.7, 0.7, 0.3, 0.3])
    fcf_j_bluepill, fcf_j_redpill = calculate_average(FCF_optimal,filename_list,6,ax=ax_in)
    # ax_in.set_xlabel('FCF_j')
    ax_in.set_ylabel('')

    ax = axes[0]
    ax.set_xlabel('FCF_j')
    ax.set_title('Functional causal flow')
    # ax.set_xlim([-0.01,0.25])

    ax_in = axes[1].inset_axes([0.7, 0.7, 0.3, 0.3])
    dir_j_bluepill, dir_j_redpill = calculate_average(directionality,filename_list,6,ax=ax_in)
    # ax_in.set_xlabel('Directionality_j')
    ax_in.set_xlim([-0.15,0.15])
    ax_in.set_ylabel('')

    ax = axes[1]
    ax.set_xlabel('Directionality_j')
    ax.set_title('Directionality')
    ax.set_xlim([-0.15,0.15])

    ax_in = axes[2].inset_axes([0.7, 0.7, 0.3, 0.3])
    cpx_j_bluepill, cpx_j_redpill = calculate_average(complexity,filename_list,6,ax=ax_in)
    # ax_in.set_xlabel('Takens_j')
    ax_in.set_ylabel('')

    ax = axes[2]
    ax.set_xlabel('Takens_j')
    ax.set_title('Complexity')

    ax_in = axes[3].inset_axes([0.7, 0.7, 0.3, 0.3])
    cpx_j_bluepill, cpx_j_redpill = calculate_average(complexity,filename_list,6,axis=1,ax=ax_in)
    # ax_in.set_xlabel('Takens_i')
    ax_in.set_ylabel('')

    ax = axes[3]
    ax.set_xlabel('Takens_i')
    ax.set_title('Complexity')
    plt.savefig(os.path.join(PlotDir,f'hubness_{mID}.pdf'))
