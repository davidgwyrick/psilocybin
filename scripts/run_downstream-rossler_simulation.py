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
import yaml
import kedm

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
from tbd_eeg.data_analysis.Utilities.behavior_movies import Movie

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
from plotting import *

#CCM
from delay_embedding import ccm
from delay_embedding import evaluation as E
from delay_embedding import helpers as H
from delay_embedding import surrogate as S
import simulator.networks as net
import ray

ProjDir = '/home/david.wyrick/projects/zap-n-zip/'
ServDir = '/data/projects/zap-n-zip/'
SimDir = '/data/clustered_NN_sim'

##===== ============================ =====##
##===== Parse Command Line Arguments =====##
parser = argparse.ArgumentParser(description='FCF')

##===== Data Options =====##
parser.add_argument('--T',type=int, default=6,
                    help='minutes to simulate')

parser.add_argument('--time_bin_ms',type=int, default=50,
                    help='time_bin_ms')

parser.add_argument('--delay',type=int, default=1,
                    help='tau')

parser.add_argument('--num_simulations',type=int, default=20,
                    help='num_simulations')

# get firing rates for a specific time window
def get_firing_rates2(spike_times, spike_clusters, unit_ids, bins):

    # bins = np.arange(tWindow[0], tWindow[1]+tBin, tBin)
    firingrate = np.empty((len(unit_ids), len(bins)-1))*np.nan

    for indi, uniti in enumerate(unit_ids):
        spikesi = np.squeeze(spike_times[spike_clusters == uniti])
        sp_counts, edges = np.histogram(spikesi, bins)
        firingrate[indi,:] = sp_counts#/tBin

    return firingrate, bins

def usr_zscore(X):
    mx = np.mean(X,axis=0); std = np.std(X,axis=0)
    Xz = np.divide(X-mx,std,out=np.zeros(X.shape),where = std!= 0)
    return Xz

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

def simulate_network(cond='rest', T=300, time_bin=0.05, save=False):

    with open('/home/david.wyrick/Git/FCF/example_configs/clustered_spiking.yaml', 'r') as file:
        pm = yaml.safe_load(file)
    pm['T'] = T; #pm['dt'] = 0.001

    if cond == 'rest':
        #During rest, set std of external inputs to 0
        pm['baseline_c'] = [[0.1027,0],[.0915,0]]
    else:
        pm['baseline_c'] = [[0.125,0.025],[.0915,0.025]]

    #Determine E/I ratio
    E = round(pm['N']*pm['EI_frac'])
    I = round(pm['N']*(1-pm['EI_frac']))
    print(f'# of E cells: {E}, # of I cells: {I}')

    #Set parameters
    pm['theta']   = np.concatenate((np.ones((E))*pm['theta_c'][0],np.ones((I))*pm['theta_c'][1]))
    pm['v_rest']  = np.concatenate((np.zeros((E))*pm['v_rest_c'][0],np.zeros((I))*pm['v_rest_c'][1])) #Resting potential
    pm['tau_syn'] = np.concatenate((np.ones((E))*pm['tau_syn_c'][0],np.ones((I))*pm['tau_syn_c'][1])) #Synaptic time consts
    pm['tau_m'] = np.concatenate((np.ones((E))*pm['tau_m_c'][0],np.ones((I))*pm['tau_m_c'][0]))       #Membrane time constants
    pm['f_mul'] = -1/pm['tau_syn']
    pm['f_add'] = 1/pm['tau_syn']
    factor = ((1000/pm['N'])**(1/2))*5*0.8*E*.2
    pm['baseline'] = np.concatenate((
            factor*pm['baseline_c'][0][0]*(np.ones((E))+(pm['baseline_c'][0][1])*(2*np.random.rand(E)-1)),
            factor*pm['baseline_c'][1][0]*(np.ones((I))+(pm['baseline_c'][1][1])*(2*np.random.rand(I)-1))
        ))

    #Run network
    network = net.ClusteredSpiking(pm['N'], pm)
    t, x, spikes, spikes_flat = network.run(pm['T'],dt=pm['dt'])

    #Separate lists
    spike_times = np.array([s for c, s in spikes_flat])
    spike_clusters = np.array([c for c, s in spikes_flat])

    #Bin spike counts
    bins = np.arange(0,pm['T']+time_bin,time_bin)
    spk_counts, bins = get_firing_rates2(spike_times, spike_clusters, np.arange(pm['N']), bins)
    
    #Subselect neurons from clusters
    cluster_starts = np.hstack((0,np.cumsum(network.pm['cluster_size'].flatten())))
    cluster_intervals = np.vstack([cluster_starts[:-1],cluster_starts[1:]]).T

    #Get firing rate
    FR = np.sum(spk_counts,axis=1)/pm['T']

    #Select subset of neurons per cluster
    neuron_indices = []
    for c in cluster_intervals:
        n_select = int((c[1] - c[0])/5)
        indy = np.arange(*c)

        FR_sub = FR[indy]
        indy_sub = np.where(FR_sub > 1)[0]

        if len(indy_sub) > n_select:
            indy_sub = np.random.choice(indy_sub,n_select)
        else:
            # import pdb; pdb.set_trace()
            indy_sub = np.argsort(FR_sub)[::-1]
            indy_sub = indy_sub[:n_select]
        neuron_indices.append(indy[indy_sub])
    neuron_indices = np.concatenate(neuron_indices)

    X_spks = spk_counts[neuron_indices,:].T
    
    if save:
        #Save instantiation into np file
        
        tmp_list = sorted(glob(join(SimDir,f'network-{cond}-cond_sim-*')))
        if len(tmp_list) == 0:
            curr_run = 0
        else:
            last_run = int(tmp_list[-1][-6:-4])
            curr_run = last_run+1
        filename = f'network-{cond}-cond_sim-{curr_run:02d}.npy'
        np.save(join(SimDir,filename),X_spks)

    # plt.figure()
    # plt.title(cond)
    # # cells
    # indsE = np.nonzero(spike_clusters < E)[0]
    # indsI = np.nonzero(spike_clusters >= E)[0]
    # plt.plot(spike_times[indsE],spike_clusters[indsE], '.', markersize=0.5, color='navy')
    # plt.plot(spike_times[indsI],spike_clusters[indsI], '.', markersize=0.5, color='firebrick')
    # plt.yticks([])
    # plt.xlabel('time [s]')
    # plt.ylabel('neurons')
    # plt.tight_layout()
    # plt.savefig(join(SimDir,f'network-{cond}-cond_run-{curr_run:02d}.pdf'))
    
    return X_spks, FR

def plot_CCM_results(corr_mat,fcf,flow,takens,fname_suffix,vmax_fcf=None,dir_clims=None,vmax_takens=None,title=None,mask=None,save=True,rBool=True,ticks=None,labels=None,boundaries=None):
    fs = 11
    fig, axes = plt.subplots(1,5,figsize=(20,4))#,gridspec_kw={'wspace': 0.1,'hspace':0.1})
    plt.suptitle(title,fontsize=22,fontweight='bold',y=0.98)

    if vmax_fcf is None:
        vmax1 = np.round(np.nanpercentile(np.abs(corr_mat.ravel()),99),2)
        vmax2 = np.round(np.nanpercentile(np.abs(fcf.ravel()),99),2)
        vmax_fcf = np.max([vmax1,vmax2])
    
    ax = axes[0]
    ax.set_title('|Correlation|',fontsize=fs,fontweight='bold')
    sns.heatmap(np.abs(corr_mat),square=True,annot=False,cmap='viridis',vmin=0, vmax=vmax_fcf,ax=ax,cbar_kws={'shrink':0.5,'label': '|Correlation| ','ticks':[0,vmax_fcf]},rasterized=rBool) #cmap=sns.color_palette("vlag", as_cmap=True),center=0,ax=ax,cbar_kws={'shrink':0.5,'label': 'Correlation'})

    ax = axes[1]
    ax.set_title('Functional causal flow',fontsize=fs,fontweight='bold')
    sns.heatmap(fcf,square=True,annot=False,cmap='viridis',vmin=0, vmax=vmax_fcf,ax=ax,mask=mask,cbar_kws={'shrink':0.5,'label': 'FCF','ticks':[0,vmax_fcf]},rasterized=rBool)

    ax = axes[2]
    vmax = np.round(np.nanpercentile(fcf - np.abs(corr_mat),98),2)
    vmin = np.round(np.nanpercentile(fcf - np.abs(corr_mat),2),2)
    ax.set_title('FCF - |Correlation|',fontsize=14,fontweight='bold')
    sns.heatmap(fcf - np.abs(corr_mat),square=True,annot=False,cmap='RdBu_r',center=0, vmin=vmin,vmax=vmax,ax=ax,mask=mask,cbar_kws={'shrink':0.5,'label': 'FCF - |Corr|','ticks':[vmin,0,vmax]},rasterized=rBool)


    ax = axes[3]
    ax.set_title('Directionality',fontsize=fs,fontweight='bold')
    if dir_clims is None:
        vmax = np.round(np.nanpercentile(flow,97.5),2); vmin = np.round(np.nanpercentile(flow,2.5),2) #-1*vmax
    else:
        vmax = dir_clims[1]; vmin = dir_clims[0]
    # pdb.set_trace()
    sns.heatmap(flow,square=True,annot=False,cmap='RdBu_r',center=0,vmin=vmin,vmax=vmax,ax=ax,cbar_kws={'shrink':0.5,'label': 'Directionality','ticks':[vmin,0,vmax]},rasterized=rBool)
    
    ax = axes[4]
    ax.set_title('Embedding dimensionality',fontsize=fs,fontweight='bold')
    if vmax_takens is None:
        vmax_takens = np.round(np.nanpercentile(np.abs(takens.ravel()),98))
    sns.heatmap(takens,square=True,annot=False,cmap='rocket',ax=ax,mask=mask,vmin=0,vmax=vmax_takens,cbar_kws={'shrink':0.5,'label': 'Takens','ticks':[0,vmax_takens]},rasterized=rBool)

    if ticks is None:
        for ax in axes:
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
        for ii, ax in enumerate(axes):
            if (ii == 2) | (ii == 3):
                c = 'k'
            else:
                c = 'w'
            ax.vlines(boundaries,*ax.get_ylim(),color=c,lw=0.5,alpha=0.5)
            ax.hlines(boundaries,*ax.get_xlim(),color=c,lw=0.5,alpha=0.5)
    ax = axes[4]
    ax.autoscale(tight=True)
    return fig
        
    # if save:
    #     plt.savefig(os.path.join(PlotDir,f'FCF_groups_{fname_suffix}.pdf'))

# FCF_optimal, complexity = determine_optimal_FCF(FCF_takens,takens_range)

if __name__ == '__main__':

    # Parse the arguments
    args = parser.parse_args()
    T = args.T*60
    time_bin_ms = args.time_bin_ms
    time_bin = time_bin_ms/1000
    delay = args.delay
    num_simulations = args.num_simulations

    # Create a new PowerPoint presentation to save figures to
    prs = Presentation()

    #Add title slide
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    slide.shapes.title.text = f'Clustered NN runs'
    slide.placeholders[1].text = f'Simulation runs'

    #Run multiple simulations
    for iSim in range(num_simulations):

        tmp_list = sorted(glob(join(SimDir,f'FCF-results_network-cond_sim-*')))
        if len(tmp_list) == 0:
            curr_run = 0
        else:
            last_run = int(tmp_list[-1][-6:-4])
            curr_run = last_run+1

        print(f'Simulation {iSim}, file index: {curr_run}')

        #Simulate data
        T = args.T*60
        print(f'Simulating {T} seconds of data')
        X_rest, FR_rest = simulate_network(cond='rest', T=T, time_bin=time_bin,save=True)
        X_run, FR_run = simulate_network(cond='run', T=T, time_bin=time_bin,save=True)
        T, N = X_run.shape

        #Plot firing rate distributions
        fig, ax = plt.subplots(figsize=(4,4))
        sns.histplot(FR_rest,binwidth=2,element='step',fill=False,lw=2,palette=cc[0],ax=ax,stat='probability',label='Rest')
        sns.histplot(FR_run,binwidth=2,element='step',fill=False,lw=2,palette=cc[0],ax=ax,stat='probability',label='Run')
        ax.legend()
        ax.set_xlabel('Firing rate (Hz)')
        ax.set_ylim([0,0.3]); ax.set_xlim([0,100])
        save_fig_to_pptx(fig, prs)
        plt.savefig(join(SimDir,f'Firing-rate-distros_sim-{curr_run:02d}.pdf'),facecolor='white',bbox_inches='tight')

        #Plot raster plots
        fig, ax = plt.subplots(figsize=(10,10))
        ax.set_title('Raster plot for "rest" condition')

        fig, ax = plt.subplots(figsize=(10,10))
        ax.imshow(X_rest.T,cmap='gray_r')
        ax.set_xticks([0,T/2,T])
        tmax = T*time_bin/60
        ax.set_xticklabels(labels=[0,tmax/2,tmax])
        ax.set_xlabel('Time (min)')
        ax.set_ylabel('Neuron ID')
        save_fig_to_pptx(fig, prs)
        plt.savefig(join(SimDir,f'raster-plot_rest-cond_sim-{curr_run:02d}.pdf'),facecolor='white',bbox_inches='tight')

        #Plot raster plots
        fig, ax = plt.subplots(figsize=(10,10))
        ax.set_title('Raster plot for "run" condition')

        fig, ax = plt.subplots(figsize=(10,10))
        ax.imshow(X_run.T,cmap='gray_r')
        ax.set_xticks([0,T/2,T])
        tmax = T*time_bin/60
        ax.set_xticklabels(labels=[0,tmax/2,tmax])
        ax.set_xlabel('Time (min)')
        ax.set_ylabel('Neuron ID')
        save_fig_to_pptx(fig, prs)
        plt.savefig(join(SimDir,f'raster-plot_run-cond_sim-{curr_run:02d}.pdf'),facecolor='white',bbox_inches='tight')

        ## Calculate FCF
        print('Calculating FCF')
        takens_range = np.arange(1,31)
        FCF_takens = np.zeros((len(takens_range),2,N,N))
        print(f'\t{N} neurons included')
        for ii, data in enumerate([X_rest,X_run]):
            X = usr_zscore(data)
            for jj in takens_range:
                edims = np.repeat(jj,N)
                FCF_takens[jj-1,ii] = kedm.xmap(X,edims,tau=1,Tp=0)

        print('Determining optimal FCF')
        FCF_optimal_rest, complexity_rest = determine_optimal_FCF(FCF_takens[:,0],takens_range)
        FCF_optimal_run, complexity_run = determine_optimal_FCF(FCF_takens[:,1],takens_range)

        print('Plotting')
        #Reformat
        FCF_optimal = np.stack((FCF_optimal_rest,FCF_optimal_run))
        correlation = np.zeros(FCF_optimal.shape)
        complexity = np.stack((complexity_rest,complexity_run))
        directionality = np.zeros(FCF_optimal.shape)
        for ii, FCF in enumerate(FCF_optimal):
            directionality[ii] = FCF - FCF.T

        for ii, data in enumerate([X_rest,X_run]):
            # spks_target = data[test_indices]
            #Calculate correlation matrix
            correlation[ii] = E.correlation_FC(data,transform='identity')
        
        filename = f'FCF-results_network-cond_sim-{curr_run:02d}.npz'
        np.savez(join(SimDir,filename),FCF_optimal=FCF_optimal,correlation=correlation,complexity=complexity,directionality=directionality)

        vmax1 = np.round(np.nanpercentile(np.abs(correlation.ravel()),99),2)
        vmax2 = np.round(np.nanpercentile(np.abs(FCF_optimal.ravel()),99),2)
        vmax_fcf = np.max([vmax1,vmax2])
        vmax_takens = np.round(np.nanpercentile(np.abs(complexity.ravel()),98))
        vmax = np.round(np.nanpercentile(directionality.ravel(),97.5),2); vmin = np.round(np.nanpercentile(directionality.ravel(),2.5),2) #-1*vmax
        dir_clims = [vmin,vmax]

        for ii, rstr in enumerate(['rest','run']):
            titlestr = f'Clustered NN during {rstr} condition'
            fig = plot_CCM_results(correlation[ii],FCF_optimal[ii],directionality[ii],complexity[ii],rstr,vmax_fcf=vmax_fcf,vmax_takens=vmax_takens,dir_clims=dir_clims,title=titlestr,mask=None,save=True,rBool=True)#,ticks=ticks,labels=group_order_labels,boundaries=boundaries)
            save_fig_to_pptx(fig, prs)
            plt.savefig(join(SimDir,f'FCF_arrays_{rstr}-cond_sim-{curr_run:02d}.png'),facecolor='white',bbox_inches='tight')
        
        fig, axes = plt.subplots(1,4,figsize=(16,4))
        for ii, mat in enumerate([FCF_optimal,directionality,complexity]):
            matrix_j = np.nanmean(mat,axis=1)
            ax = axes[ii]
            sns.kdeplot(matrix_j[0],ax=ax,color=cc[1],label='rest')
            sns.kdeplot(matrix_j[1],ax=ax,color=cc[2],label='run')
            ax.legend()
        
        ax = axes[3]
        complexity_i = np.nanmean(complexity,axis=2)
        sns.kdeplot(complexity_i[0],ax=ax,color=cc[1],label='rest')
        sns.kdeplot(complexity_i[1],ax=ax,color=cc[2],label='run')
        ax.legend()
        
        title_strs = ['Functional causal flow','Directionality','Complexity of\nsource','Complexity of\ntarget']
        xlabels = ['FCF_j','Directionality_j','Takens_j','Takens_i']
        for ii, title in enumerate(title_strs):
            axes[ii].set_title(title)
            axes[ii].set_xlabel(xlabels[ii])

        save_fig_to_pptx(fig, prs)
        plt.savefig(join(SimDir,f'FCF-j_sim-{curr_run:02d}.png'),facecolor='white',bbox_inches='tight')
    prs.save(join(SimDir,f'simulations.pptx'))
    print('DONE!!!')