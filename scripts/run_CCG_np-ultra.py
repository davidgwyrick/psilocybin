ProjDir = '/home/david.wyrick/projects/zap-n-zip/'
ServDir = '/data/projects/zap-n-zip/'

#Base
import argparse
from glob import glob
from os.path import join
import json, os, time, sys
import gspread
import pandas as pd
import numpy as np
from tqdm import tqdm

#Scipy
import scipy.signal as sig
import scipy.stats as st
from scipy.ndimage import gaussian_filter

#Plot
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from pptx import Presentation

#Allen
from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
mcc = MouseConnectivityCache(resolution=10)
str_tree = mcc.get_structure_tree()

#Read in allen CCF
ccfsum = pd.read_csv('/home/david.wyrick/projects/zap-n-zip/data/mouse_connectivity/ontology_v2.csv')

#User
sys.path.append(ProjDir)
import util
import plotting as usrplt

#CCG
sys.path.append('/home/david.wyrick/Git/functional-network')
import ccg_library as ccg_lib

##===== ============================ =====##
##===== Parse Command Line Arguments =====##
parser = argparse.ArgumentParser(description='CCG analysis on np-ultra data')

##===== Data Options =====##
parser.add_argument('--sID',type=str, default='2024-06-05_717033',
                    help='session to perform analysis on')

parser.add_argument('--time_bin_ms',type=int, default=1,
                    help='Time bin width')

parser.add_argument('--tWindow_width_s',type=int, default=120,
                    help='Time window over which to calculate FC')

parser.add_argument('--tWindow_shift_s',type=float, default=20,
                    help='Amount of time to shift rolling window by')

parser.add_argument('--fr_thresh',type=float, default=0.5,
                    help='fr_thresh')

##===== ======= Argument End ======= =====##
##===== ============================ =====##

if __name__ == '__main__':

    ## Parse the arguments ----------------------------------------
    args = parser.parse_args()

    #Which experiment?
    sID = args.sID
   
    #How to segment data
    time_bin_ms = args.time_bin_ms
    time_bin = time_bin_ms/1000
    tWindow_width_s = args.tWindow_width_s
    tWindow_width = tWindow_width_s#*60
    tWindow_shift = args.tWindow_shift_s

    #Data preprocessing
    fr_thresh = args.fr_thresh

    ## FOLDERS ----------------------------------------
    #Create directory for saving to
    # '/data/projects/zap-n-zip/results/np_ultra/CCG'
    TempDir = os.path.join(ServDir,'results','np_ultra','CCG',sID)
    if not os.path.exists(TempDir):
        os.makedirs(TempDir)

    tmp_list = sorted(glob(join(TempDir,f'run_*')))
    if len(tmp_list) == 0:
        curr_run = 0
    else:
        last_run = int(tmp_list[-1][-1])
        curr_run = last_run+1

    folder = f'run_{curr_run:02d}'
    SaveDir = os.path.join(TempDir,folder)
    if not os.path.exists(SaveDir):
        os.makedirs(SaveDir)
    
    PlotDir = join(SaveDir,'plots')
    if not os.path.exists(PlotDir):
        os.makedirs(PlotDir)

    #Save model parameters
    args_dict = args.__dict__
    args_dict['SaveDir'] = SaveDir

    with open(join(SaveDir,f"parameters_run_{curr_run}.json"), "w") as outfile:
        json.dump(args_dict, outfile)

    ## Load data ----------------------------------------
    #Read in the data
    data_path = '/data/np_ultra/pilot_psi_exp/2025-01-08_NPUltra_unittable.pkl'
    data = pd.read_pickle(data_path)

    #Insert layer information
    data.insert(10,'area',data['layer'])

    areas = data['area'].values
    groups = data['region'].values  
    #Get layer info
    layers = []
    for a, g in zip(areas,groups):
        if '1' in a:
            layers.append('1')
        elif '2/3' in a:
            layers.append('2/3')
        elif '2' in a:
            layers.append('2')
        elif '3' in a:
            layers.append('3')
        elif '4' in a:
            layers.append('4')
        elif '5' in a: 
            layers.append('5')
        elif '6' in a:
            layers.append('6')
        else:
            layers.append('none')
    layers = np.array(layers)
    data['layer'] = layers

    areas = data['area'].values
    boundaries, yticks, labels, layers, areas, groups, mesogroups, supergroups, order_by_group = util.get_group_plotting_info2(areas)
    data.insert(11,'group',groups)
    data.insert(12,'mesogroup',mesogroups)
    data.insert(13,'supergroup',supergroups)

    #Fix 'Spontaneous_3_spikes','Spontaneous_4_spikes' columns
    n_list = []
    for i, row in data.iterrows():
        for c in ['Spontaneous_3_spikes','Spontaneous_4_spikes']:
            try:
                n = len(data.at[i,c])
                n_list.append(n)
            except:
                data.at[i,c] = np.array([])

    opto_cell = []
    for g in data['genotype'].values:
        if 'Sim1' in g:
            opto_cell.append('Sim1')
        elif 'Tlx3' in g:
            opto_cell.append('Tlx3')
    data['opto_cell'] = opto_cell

    columns = ['spike_times', 'Spontaneous_0_spikes', 'Spontaneous_1_spikes','Spontaneous_2_spikes','Spontaneous_3_spikes','Spontaneous_4_spikes']
    epoch_names = ['full-exp','pre-inj','post-inj-early','post-inj-late','post-iso','post-iso-recovery']
    sessionID_list = data['sessionID'].unique()

    #Get estimate for window times
    epoch_windows = {}

        
    sData = data[data['sessionID'] == sID]
    nOpto = np.sum(sData['optotagged'])
    print(f'{sID}: {len(sData)} cells, {nOpto} opto-tagged')

    #Get min spike time and largest spike time for each window
    epoch_windows = {}
    for e, c in zip(epoch_names, columns):
        spk_times_all = sData[c].values
        min_st = []; max_st = []
        for spk_times_neuron in spk_times_all:
            try:
                nSpks = len(spk_times_neuron)
            except:
                nSpks = 0

            if nSpks > 0:
                min_st.append(spk_times_neuron.min())
                max_st.append(spk_times_neuron.max())
        if len(min_st) == 0:
            tStart = np.nan; tEnd = np.nan
        else:
            tStart = np.nanmin(min_st); tEnd = np.nanmax(max_st)
            epoch_windows[c] = [tStart, tEnd]
        print(f'\t{e}: [{tStart/60:.1f},{tEnd/60:.1f}] min; {(tEnd-tStart)/60:.2f} min')
        
    ## Calculate CCGs ----------------------------------------

    data_sub = data[data['sessionID'] == sID]

    plot_before = 0
    plot_after = 1
    psueudo_trial_length = plot_before + plot_after
    time_bin = 1/1000
            
    columns = ['Spontaneous_0_spikes', 'Spontaneous_1_spikes','Spontaneous_2_spikes']
    epoch_names = ['pre-inj','post-inj-early','post-inj-late']
    ct_list = ['FSl', 'FSs', 'RS', 'Sim1', 'Tlx3']
    ct_names = ['FS (large footprint)','FS (small footprint)','RS','Sim1 (PT)','Tlx3 (IT)']
    cmap_sal = sns.xkcd_palette(['silver','dark sky blue','cobalt blue'])
    cmap_psi = sns.xkcd_palette(['pale rose','red','dark red'])

    drug_type = data_sub['experiment'].values[0]
    if drug_type == 'saline':
        cmap = cmap_sal
    elif drug_type == 'psilocybin':
        cmap = cmap_psi
    nN_total = len(data_sub)
    print(f'Processing {sID}, {drug_type}, {nN_total} neurons')

    #Get spike times for all neurons in session
    spike_times_all = data_sub['Spontaneous_0_spikes'].values
    tW_epoch = epoch_windows[sID]['Spontaneous_0_spikes']

    #Calculate average firing rate per neuron
    meanFR = np.full((nN_total),np.nan)
    for iN,  spk_times_neuron in enumerate(spike_times_all):
        meanFR[iN] = len(spk_times_neuron)/(tW_epoch[1]-tW_epoch[0])
    
    #Select neurons with FR > 0.5 Hz
    neural_indices = np.where(meanFR >= fr_thresh)[0]
    nN = len(neural_indices)
    print(f'Processing {sID}, {drug_type}, {nN} neurons out of {nN_total} with FR > {fr_thresh} Hz')

    areas = data_sub['area'].values
    cell_types = data_sub['ct'].values
    areas = areas[neural_indices]
    cell_types = cell_types[neural_indices]
    boundaries, ticks, labels, layers, areas, groups, mesogroups, supergroups, order_by_group = util.get_group_plotting_info2(areas)
    nConnFrac_list = []
    for e, c in zip(epoch_names,columns):
        #Get spike times for all neurons in session
        spike_times_all = data_sub[c].values
        # areas = data_sub['area'].values
        # cell_types = data_sub['ct'].values
        # boundaries, ticks, labels, layers, areas, groups, mesogroups, supergroups, order_by_group = util.get_group_plotting_info2(areas)

        #Get time window of spontaneous activity
        tW_epoch = epoch_windows[c]

        # #Break up 20min window into 5 min windows
        # tWindow_width = 10*60
        # tW_starts = np.arange(tW_epoch[0],tW_epoch[1],tWindow_width)[:-1]
        # tW_ends = tW_starts + tWindow_width
        # time_window_array = np.array([tW_starts,tW_ends]).T
        # time_window_list = time_window_array.tolist()
        # time_window_centers = time_window_array[:,0] + tWindow_width/2
        # nWindows = len(time_window_list)   

        # for ii, tW in enumerate(time_window_list):
        #     tW_center = tW[0] + (tW[1]-tW[0])
        #     e_sub = f'{e}_{ii}'
            
        ii = 0
        e_sub = f'{e}'
        tW = tW_epoch
        
        #Create pseudo-trials of 1s duration
        trial_times = np.arange(tW[0],tW[1],1)
        bins = np.arange(-plot_before, plot_after+time_bin, time_bin)
        
        #Bin spike times around "trial" times
        spk_counts = np.full((nN_total,len(trial_times),len(bins)-1),np.nan)
        
        #Loop through neurons
        for iN, spk_times_neuron in enumerate(spike_times_all):
            for iTrial, E in enumerate(trial_times):
                window_spikes = spk_times_neuron[np.squeeze(np.argwhere((spk_times_neuron >= E - plot_before) & (spk_times_neuron <= E + plot_after)))]
                window_spikes = window_spikes - E
                spk_counts[iN,iTrial], _ = np.histogram(window_spikes, bins)

        # data_list.append(spk_counts)
        spk_counts = spk_counts[neural_indices]
        print(f'\t\t{ii}: {spk_counts.shape}')

        #Calculate CCG
        ccg = ccg_lib.CCG(num_jitter=25, L=25, window=50, memory=False, use_parallel=True, num_cores=30)
        ccg_jitter_corrected, ccgs_uncorrected = ccg.calculate_mean_ccg_corrected(spk_counts, disable=False)
        
        #Detect significant connections
        connection_detection = ccg_lib.SharpPeakIntervalDetection(max_duration=10, maxlag=15, n=4)
        significant_ccg,significant_confidence,significant_offset,significant_duration = connection_detection.get_significant_ccg(ccg_jitter_corrected)
        full_ccg, full_confidence, full_offset, full_duration = connection_detection.get_full_ccg(ccg_jitter_corrected)
        adjmat = ~np.isnan(significant_ccg)
        nConnFrac = np.sum(~np.isnan(significant_ccg))/np.prod(significant_ccg.shape)
        
        #Save
        np.savez(join(SaveDir,f'ccg_{drug_type}_{e_sub}_{sID}.npz'),neural_indices=neural_indices,ccg_jitter_corrected=ccg_jitter_corrected,significant_ccg=significant_ccg,significant_confidence=significant_confidence,significant_offset=significant_offset,significant_duration=significant_duration,full_ccg=full_ccg,full_confidence=full_confidence,full_offset=full_offset,full_duration=full_duration,nConnFrac=nConnFrac)

        #Plot
        fig, axes = plt.subplots(1,2,figsize=(10,5))
        plt.suptitle(f'Cross correlograms: {sID}, {e_sub} {drug_type}',y=.9)
        nConnFrac = np.sum(~np.isnan(significant_ccg))/np.prod(significant_ccg.shape)
        nConnFrac_list.append(nConnFrac)
        vmax = np.nanpercentile(np.abs(significant_ccg),98)
        axes[0].set_title('Fraction connected: {:.2f}'.format(nConnFrac))
        tmp = significant_ccg.copy()
        tmp = tmp[order_by_group][:,order_by_group]
        sns.heatmap(tmp,cmap='RdBu_r',center=0,vmin=-1*vmax,vmax=vmax,square=True,cbar_kws={'label':'CCG','shrink':.5},ax=axes[0])

        tmp = significant_offset.copy()
        axes[1].set_title(f'Offset of connections: mean = {np.nanmean(significant_offset):.2f} ms')
        tmp = significant_offset.copy()
        tmp = tmp[order_by_group][:,order_by_group]
        sns.heatmap(tmp,cmap='viridis',square=True,cbar_kws={'label':'Lag (ms)','shrink':.5},ax=axes[1])

        for ax in axes:
            ax.set_xticks(ticks)
            ax.set_xticklabels(labels)
            ax.vlines(boundaries,0,nN, color='k', lw=1,alpha=0.5)
            ax.hlines(boundaries,0,nN, color='k', lw=1,alpha=0.5)
            ax.set_yticks(ticks)
            ax.set_yticklabels(labels)

        plt.savefig(join(PlotDir,f'ccg_{drug_type}_{e_sub}_{sID}_2.png'),dpi=300,bbox_inches='tight',facecolor='w')
        plt.close()
        # pdb.set_trace()
    fig, ax = plt.subplots(figsize=(6,4))
    plt.suptitle(f'{sID}: {drug_type}',y=.95)
    ax.plot(nConnFrac_list)
    ylim = ax.get_ylim()
    ax.vlines(4,*ylim,color=cmap[1],ls='--')
    ax.vlines(8,*ylim,color=cmap[2],ls='--')
    ax.set_xticks(np.arange(0,13))
    ax.set_xticklabels(np.arange(0,13)*5)
    usrplt.adjust_spines(ax)
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Fraction of connections')
    plt.savefig(join(PlotDir,f'frac-connected_{drug_type}_{sID}_2.png'),dpi=300,bbox_inches='tight',facecolor='w')
    




