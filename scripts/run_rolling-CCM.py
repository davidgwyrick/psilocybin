base_dir_server = '/allen/programs/mindscope/workgroups/templeton-psychedelics/'
base_dir_local = '/data/tiny-blue-dot/zap-n-zip/EEG_exp'; base_dir = base_dir_server
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
import pingouin as pg
import itertools as it
#Scipy
import scipy.signal as sig
import scipy.stats as st
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression, PoissonRegressor

#Plot
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from pptx import Presentation

#Project
from tbd_eeg.data_analysis.eegutils import EEGexp
import tbd_eeg.data_analysis.Utilities.utilities as tbd_util

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

#CCM
from delay_embedding import ccm
from delay_embedding import evaluation as E
from delay_embedding import helpers as H
from delay_embedding import surrogate as S
import ray
import kedm

#Network 
import networkx as nx
import networkx.algorithms.community as nx_comm
from networkx.algorithms.community import greedy_modularity_communities, modularity
from networkx.algorithms.efficiency_measures import global_efficiency, local_efficiency

#Templeton-log_exp
gc = gspread.service_account() # need a key file to access the account (step 2) 
sh = gc.open('Templeton-log_exp') # open the spreadsheet 

df = pd.DataFrame(sh.sheet1.get()) 
exp_table = df.T.set_index(0).T # put it in a nicely formatted dataframeexp_table.iloc[10:]
exp_table = exp_table.set_index('mouse_name')
exp_table.head()

##===== ============================ =====##
##===== Parse Command Line Arguments =====##
parser = argparse.ArgumentParser(description='Rolling FCF')

##===== Data Options =====##
parser.add_argument('--mID',type=str, default='mouse735052',
                    help='mouse to perform analysis on')

parser.add_argument('--rec_name',type=str, default='aw_psi_2024-06-20_10-40-56',
                    help='experiment to perform analysis on')

parser.add_argument('--time_bin_ms',type=int, default=50,
                    help='Time bin width')

parser.add_argument('--tWindow_width_s',type=int, default=300,
                    help='Time window over which to calculate FC')

parser.add_argument('--tWindow_shift_s',type=float, default=150,
                    help='Amount of time to shift rolling window by')

parser.add_argument('--isi_interval_ms',type=int, default=5,
                    help='Max ISI for burst definition')

parser.add_argument('--min_spks_per_burst',type=int, default=3,
                    help='minimum number of spikes per burst')

parser.add_argument('--quiescence_ms',type=int, default=10,
                    help='Amount of time before burst')

parser.add_argument('--delay',type=int, default=1,
                    help='tau')

parser.add_argument('--zscore',type=int, default=0,
                    help='zscore spike counts?')

parser.add_argument('--fr_thresh',type=float, default=0,
                    help='Firing rate threshold for neurons to include in analysis')

parser.add_argument('--FCF_thresh',type=float, default=0.1,
                    help='Functional connectivity threshold to assess significance')


##===== ======= Argument End ======= =====##
##===== ============================ =====##

if __name__ == '__main__':

    ## Parse the arguments ----------------------------------------
    args = parser.parse_args()

    #Which experiment?
    mID = args.mID
    rec_name = args.rec_name
    print(f'{mID}, {rec_name}')

    #Parameters
    time_bin_ms = args.time_bin_ms
    time_bin = time_bin_ms/1000
    tWindow_width_s = args.tWindow_width_s
    tWindow_width = tWindow_width_s#*60
    tWindow_shift = args.tWindow_shift_s
    delay = args.delay
    zscore = bool(args.zscore)
    fr_thresh = args.fr_thresh
    FCF_thresh = args.FCF_thresh
    isi_interval = args.isi_interval_ms/1000
    min_spks_per_burst = args.min_spks_per_burst
    quiescence = args.quiescence_ms/1000

    ## FOLDERS ----------------------------------------
    #Create directory for saving to
    TempDir = os.path.join(ServDir,'results','FC_neuron',mID,rec_name)
    if not os.path.exists(TempDir):
        os.makedirs(TempDir)

    tmp_list = sorted(glob(join(TempDir,f'kedm_run_*')))
    if len(tmp_list) == 0:
        curr_run = 0
    else:
        last_run = int(tmp_list[-1][-1])
        curr_run = last_run+1

    folder = f'kedm_run_{curr_run:02d}'
    SaveDir = os.path.join(TempDir,folder)
    if not os.path.exists(SaveDir):
        os.makedirs(SaveDir)
    
    PlotDir = join(SaveDir,'plots')
    if not os.path.exists(PlotDir):
        os.makedirs(PlotDir)

    #Save model parameters
    args_dict = args.__dict__
    args_dict['SaveDir'] = SaveDir
    args_dict['kEDM'] = 'xmap'
    with open(join(SaveDir,f"ccm_parameters_run_{curr_run}.json"), "w") as outfile:
        json.dump(args_dict, outfile)

    #Extract row from templeton dataframe
    exp_df = exp_table.loc[exp_table.exp_name == rec_name]
    if type(exp_df) == pd.core.series.Series:
        exp_df = exp_df.to_frame().T

    drug_type = exp_df['drug'].values[0]
    stim_type = exp_df['stimulation'].values[0]
    print(f'Experiment type: {stim_type}, {drug_type}')

    ## Read in experiment ----------------------------------------
    #Upload the whole experiment and generate the global clock
    exp = EEGexp(os.path.join(base_dir,mID,rec_name,'experiment1','recording1'), preprocess=False, make_stim_csv=False)

    #Load probe data
    probe_unit_data, probe_info, nN, metric_list = tbd_util.get_neuropixel_data(exp)
    probe_list = [x.replace('_sorted', '') for x in exp.experiment_data if 'probe' in x]

    #Get recording start time
    probei = probe_list[0]
    open_ephys_start = np.round(np.min(probe_unit_data[probei]['spike_times']))
    open_ephys_end = np.round(np.max(probe_unit_data[probei]['spike_times']))
    recording_length = open_ephys_end - open_ephys_start
    mm, ss = divmod(recording_length,60)
    hh, mm = divmod(mm, 60)
    print(f'{hh} hrs, {mm} minutes, {ss} seconds')

    #Read in behavior
    run_ts, run_signal, run_signal_s, acceleration, pupil_ts, pupil_radius, pupil_dxdt, plot_pupil = util.get_behavioral_data(exp, mID, rec_name)
    f_run = interp1d(run_ts,run_signal); run_signal_p = f_run(pupil_ts)
    f_run_s = interp1d(run_ts,run_signal_s); run_signal_p_s = f_run_s(pupil_ts)
    f_pupil = interp1d(pupil_ts,pupil_radius)

    injection_times = None; injection_time_windows = None; inj_tuple = None
    #For saline & psilocybin experiments, get injection times and types of injection
    if drug_type in ['saline', 'psilocybin','ketanserin+psilocybin','urethane+psilocybin']:
        injection_time_windows = np.array([np.array(exp_df['First injection window'].values[0].split(','),dtype=float),
                                np.array(exp_df['Second injection window'].values[0].split(','),dtype=float)])
        
        #Take second time in each window as "injection time"
        injection_times = np.array([exp_df['First injection time (s)'].values[0],exp_df['Second injection time (s)'].values[0]],dtype=float)

        if drug_type in ['psilocybin','urethane+psilocybin']:
            injection_types = ['sal1','psi']
            injection_colors = sns.xkcd_palette(['dark sky blue','darkish red'])
            cmap = sns.xkcd_palette(['silver','dark sky blue','cobalt blue'])
        elif drug_type == 'saline':
            injection_types = ['sal1', 'sal2']
            injection_colors = sns.xkcd_palette(['dark sky blue','cobalt blue'])
        elif drug_type == 'ketanserin+psilocybin':
            injection_types = ['ket','psi']
            injection_colors = sns.xkcd_palette(['magenta','goldenrod'])
        inj_tuple = (injection_times, injection_types, injection_colors)

    #For isoflurane experiments, get iso level
    iso_induction_times = None; iso_maintenance_times = None; iso_tuple = None
    if drug_type == 'isoflurane':
        iso_level, iso_times = exp.load_analog_iso()
        iso_induction_times, iso_maintenance_times = exp.load_iso_times()

    # extract the timestamps of the selected stimuli
    evoked_time_window_list = []; evoked_type_list = []; evoked_tuple = None
    try:
        stim_log = pd.read_csv(exp.stimulus_log_file)
        stim_exists = True
        for s in np.unique(stim_log['sweep']):
            for t in np.unique(stim_log.loc[stim_log.sweep == s]['stim_type']):
                sub_df = stim_log.loc[(stim_log.sweep == s) & (stim_log.stim_type == t)]
                tS = sub_df['onset'].min(); tE = sub_df['offset'].max()
                
                evoked_time_window_list.append([tS,tE])
                evoked_type_list.append(t)
        evoked_tuple = (evoked_time_window_list,evoked_type_list)

    except:
        stim_exists = False
    
    #Determine rolling windows
    tW_starts = np.arange(open_ephys_start,open_ephys_end-tWindow_width,tWindow_shift)
    tW_ends = tW_starts + tWindow_width
    time_window_array = np.array((tW_starts,tW_ends)).T
    time_window_list = time_window_array.tolist()
    time_window_centers = time_window_array[:,0] + tWindow_width/2
    nWindows = len(time_window_list)
    print(f'{nWindows} windows, {tWindow_width} seconds long, separeted by {tWindow_shift} sec')

    #Create epoch names
    epoch_list = []
    for ii, tW in enumerate(time_window_list):
        tW_center = tW[0] + (tW[1]-tW[0])
        window_type = 'spont'
        for epoch_type, epoch_window in zip(evoked_type_list,evoked_time_window_list):
            if (tW_center >= epoch_window[0]) & (tW_center < epoch_window[1]):
                window_type = epoch_type
                break
        if drug_type in ['saline', 'psilocybin','ketanserin+psilocybin','urethane+psilocybin']: 
            if tW_center < injection_times[0]:
                epoch_list.append(f'{window_type}-{ii:03d}_pre-inj')
            elif (tW_center >= injection_times[0]) & (tW_center < injection_times[1]):
                epoch_list.append(f'{window_type}-{ii:03d}_post-{injection_types[0]}-inj')
            else:
                epoch_list.append(f'{window_type}-{ii:03d}_post-{injection_types[1]}-inj')
        elif drug_type == 'isoflurane':
            if tW_center < iso_induction_times[0]:
                epoch_list.append(f'{window_type}-{ii:03d}_pre-iso')
            elif (tW_center >= iso_induction_times[0]) & (tW_center < iso_induction_times[1]):
                epoch_list.append(f'{window_type}-{ii:03d}_iso-ind')
            else:
                epoch_list.append(f'{window_type}-{ii:03d}_post-iso')
        else:
            t1 = int(tW[0]/60)
            t2 = int(tW[1]/60)
            epoch_list.append(f'{window_type}-{ii:03d}_{t1}-{t2}')
            
    # #Get overall firing rates
    # FR = np.concatenate([np.array(probe_unit_data[probei]['firing_rate']) for probei in probe_list])
    # FR_mask = FR >= fr_thresh
    # neuron_indices = np.where(FR_mask)[0]
    # N = len(neuron_indices)
    # print(f'{N} neurons > {fr_thresh} Hz overall')
    
    ## Bin spiking data ----------------------------------------
    tW_full = [open_ephys_start,open_ephys_end]
    data_list, ts_list, neuron_indices, plot_tuple, _ = util.bin_spiking_data(probe_unit_data, [tW_full], time_bin=time_bin,fr_thresh=0)
    data = data_list[0]; ts = ts_list[0] #[:,neuron_indices]
    T, N = data.shape

    if len(plot_tuple) == 11:
        boundaries, ticks, labels, celltypes, durations, layers, areas, groups, mesogroups, supergroups, order_by_group = plot_tuple
        print(f'{N} neurons, {len(np.unique(areas))} areas, {len(np.unique(groups))} groups, {len(np.unique(supergroups))} supergroups')
        area_info = True
    else:
        boundaries, ticks, labels = plot_tuple
        area_info = False
        print('No area information')
        exit()

    #Calculate behavioral measures
    running_moments = []
    pupil_moments = []
    for tW in time_window_list:
        indy = np.where((run_ts > tW[0]) & (run_ts <= tW[1]))
        rs_m = np.nanmean(run_signal[indy])
        rs_std = st.iqr(run_signal[indy])
        running_moments.append((rs_m,rs_std))
        if plot_pupil:
            indy = np.where((pupil_ts > tW[0]) & (pupil_ts <= tW[1]))
            pd_m = np.nanmean(pupil_radius[indy])
            pd_std = st.iqr(pupil_radius[indy])
            pupil_moments.append((pd_m,pd_std))
    running_moments = np.array(running_moments)
    pupil_moments = np.array(pupil_moments)

    #Plot behavior
    fig = usrplt.plot_behavior((run_ts,run_signal),(pupil_ts,pupil_radius),f'{mID} {rec_name}',evoked_tuple,inj_tuple,iso_tuple)
    plt.savefig(join(PlotDir,f'behavior_{rec_name}.png'),dpi=300,bbox_inches='tight',facecolor='w')
    
    #Save time windows
    np.savez(os.path.join(SaveDir,f'time_windows.npz'),time_window_list=time_window_list,epoch_list=epoch_list,
             running_moments=running_moments,pupil_moments=pupil_moments,areas=areas,groups=groups,supergroups=supergroups,celltypes=celltypes,order_by_group=order_by_group)
    
    ## Identify bursts ------------------------------------------
    tW_full = [open_ephys_start,open_ephys_end]  
    spike_time_dict = util.get_spike_time_dict(probe_unit_data, tWindow=tW_full)

    burst_time_dict = {}
    max_bl = np.full((nN,nWindows),np.nan)
    mean_bl = np.full((nN,nWindows),np.nan)
    burst_prop = np.full((nN,nWindows),np.nan)
    burst_rate = np.full((nN,nWindows),np.nan)

    for iN in range(nN):
        spk_times_i = spike_time_dict[iN]
        burst_spk_times, nSpikes_per_burst, mISI_per_burst, burst_indices = util.get_burst_events(spk_times_i, isi_interval=isi_interval,min_spikes=min_spks_per_burst,quiescence=quiescence)
        # burst_time_dict[iN] = burst_spk_times
        if np.all(np.isnan(burst_spk_times)):
            continue

        burst_lengths = np.zeros((len(burst_spk_times)))
        burst_spk_times_m = np.zeros((len(burst_spk_times)))
        # burst_spk_times_e = np.zeros((len(burst_spk_times)))
        for ii, iS in enumerate(burst_indices):
            iE = iS + nSpikes_per_burst[ii] - 1
            bl = spk_times_i[iE] - spk_times_i[iS]
            burst_lengths[ii] = bl
            burst_spk_times_m[ii] = spk_times_i[iS] + bl/2  #Get mid time of burst
            # burst_spk_times_e[ii] = spk_times_i[iE]       #Get end time of burst
        
        burst_time_dict[iN] = burst_spk_times_m
        for iW, tW in enumerate(time_window_list):
            b_indy = np.where((burst_spk_times >= tW[0]) & (burst_spk_times < tW[1]))[0]
            s_indy = np.where((spk_times_i >= tW[0]) & (spk_times_i < tW[1]))[0]
            if len(b_indy) == 0:
                continue
            burst_prop[iN,iW] = len(b_indy)/len(s_indy)
            burst_rate[iN,iW] = len(b_indy)/(tW[1]-tW[0])
            max_bl[iN,iW] = np.max(burst_lengths[b_indy])
            mean_bl[iN,iW] = np.mean(burst_lengths[b_indy])

    np.savez(join(SaveDir,f'burst_data_{rec_name}.npz'),burst_prop=burst_prop,burst_rate=burst_rate,max_bl=max_bl,mean_bl=mean_bl)

    # ## Bin burst events ------------------------------------------
    # bins = np.arange(tW_full[0], tW_full[1]+time_bin, time_bin)
    # burst_rate = np.full((nN,len(bins)-1),np.nan)
    # for iN in range(nN):
    #     if iN not in burst_time_dict.keys():
    #         continue
    #     burst_spk_times = burst_time_dict[iN]
    #     burst_rate[iN], edges = np.histogram(burst_spk_times, bins)

    ## Calculate rolling FCF ----------------------------------------
    t0_outer = time.perf_counter()
    takens_range = np.arange(1,40) 
    X_all = data
    T, nN = X_all.shape

    mod_FCF = np.zeros((nWindows,2))
    mod_corr = np.zeros((nWindows,2))

    #Create community structure based on group definition
    group_comm_list = []
    for g in np.unique(groups):
        indy = np.where(groups == g)[0]
        group_comm_list.append(frozenset(indy))

    FCF_all = []
    pdfdoc = PdfPages(join(PlotDir,f'rolling_FCF_{mID}_{rec_name}.pdf'))
    #Loop over different time blocks
    for ii, tW in enumerate(time_window_list):
        t0 = time.perf_counter()
        print(epoch_list[ii])

        #Subselect window
        indy = np.where((ts >= tW[0]) & (ts < tW[1]))[0]
        X = X_all[indy]

        #Remove neurons with no spikes
        neural_indices = np.where(np.nansum(X,axis=0) > 0)[0]
        X = X[:,neural_indices]
        n = len(neural_indices)
        print(f'\t{n} neurons with spikes in epoch')
        # if zscore:
        #     X = util.usr_zscore(X)

        print('\tCalculating Correlation')
        #Calculate correlation
        correlation = E.correlation_FC(X)
        COR_plot = np.zeros((nN,nN))
        COR_plot[np.ix_(neural_indices,neural_indices)] = correlation
        
        print('\tCalculating FCF')
        FCF_takens = np.zeros((len(takens_range),n,n))  

        #Calculate cross-mapping skill with set takens dimension
        for kk, dim in enumerate(tqdm(takens_range)):
            edims = np.repeat(dim,n)
            FCF_takens[kk] = kedm.xmap(X,edims,tau=1)
        FCF_optimal, complexity = util.determine_optimal_FCF(FCF_takens,takens_range)
        FCF_optimal[np.diag_indices(n)] = np.nan
        directionality = FCF_optimal - FCF_optimal.T

        #Save 
        np.savez(join(SaveDir,f'rolling-FC_{epoch_list[ii]}.npz'),neural_indices=neural_indices,FCF_takens=FCF_takens,correlation=correlation,FCF_optimal=FCF_optimal,complexity=complexity,mID=mID,rec_name=rec_name,tWindow=tW,epoch=epoch_list[ii])
        
        #Reshape
        FCF_plot = np.zeros((nN,nN))
        FCF_plot[np.ix_(neural_indices,neural_indices)] = FCF_optimal
        FCF_all.append(FCF_optimal.copy())

        DIR_plot = np.zeros((nN,nN))
        DIR_plot[np.ix_(neural_indices,neural_indices)] = directionality

        CPX_plot = np.zeros((nN,nN))
        CPX_plot[np.ix_(neural_indices,neural_indices)] = complexity
        
        ## Calculate modularity ----------------------------------------
        #Define adjacency matrix from correlation
        tmp_corr = COR_plot.copy()
        tmp_corr[np.diag_indices(nN)] = 0
        mask = tmp_corr > FCF_thresh
        tmp_corr[~mask] = 0
        UG_weight = nx.Graph(tmp_corr)

        #Find communities and calculate modularity
        comm_list = greedy_modularity_communities(UG_weight,weight='weight')
        mod_corr[ii,0] = modularity(UG_weight,comm_list)

        #Define adjacency matrix 
        mask = FCF_plot > FCF_thresh
 
        #Define directed graph
        weighted_FC = FCF_plot.copy()
        weighted_FC[~mask] = 0
        DG_weight = nx.DiGraph(weighted_FC)

        #Find communities and calculate modularity
        comm_list = greedy_modularity_communities(DG_weight,weight='weight')
        mod_FCF[ii,0] = modularity(DG_weight,comm_list)

        #Calculate modulatity of group definitions of community
        if area_info:
            mod_corr[ii,1] = modularity(UG_weight,group_comm_list)
            mod_FCF[ii,1] = modularity(DG_weight,group_comm_list)
        
        #Plot FCF results
        COR_plot = COR_plot[order_by_group,:][:,order_by_group]
        FCF_plot = FCF_plot[order_by_group,:][:,order_by_group]
        DIR_plot = DIR_plot[order_by_group,:][:,order_by_group]
        CPX_plot = CPX_plot[order_by_group,:][:,order_by_group]
        fig = usrplt.plot_CCM_results(COR_plot,FCF_plot,DIR_plot,CPX_plot,vmax_fcf=None,dir_clims=None,vmax_takens=None,title=epoch_list[ii],ticks=ticks,labels=labels,boundaries=boundaries)
        pdfdoc.savefig(fig)
        plt.savefig(join(PlotDir,f'rolling_FCF_{epoch_list[ii]}.png'),facecolor='white',dpi=300,bbox_inches='tight')
        plt.close(fig)
        # pdb.set_trace()
        # FCF_all.append(FCF_optimal)
        tE = (time.perf_counter() - t0)/60
        print('\tCompleted in {:.2f} mins'.format(tE))

    comp_length = time.perf_counter() - t0_outer
    mm, ss = divmod(comp_length,60)
    hh, mm = divmod(mm, 60)
    print(f'\n All windows calculated in {hh} hrs, {mm} minutes, {ss} seconds')
    np.savez(join(SaveDir,f'rolling-modularity_{mID}_{rec_name}.npz'),mod_FCF=mod_FCF,mod_corr=mod_corr)

    ## Plot modularity ----------------------------------------
    fig, axes = plt.subplots(3,1,figsize=(10,8),gridspec_kw = {'height_ratios': [4,2,2],'hspace':0.4})
    plt.suptitle(f'Modularity on rolling correlation matrices; {mID}, {rec_name}')

    cc0 = np.corrcoef(running_moments[:,0],mod_corr[:,0])[0,1]
    cc1 = np.corrcoef(running_moments[:,0],mod_corr[:,1])[0,1]
    ax = axes[0]
    ax.plot(time_window_centers/60,mod_corr[:,0],lw=2,zorder=2,color=usrplt.cc[1],label=f'Modularity_algorithm: {cc0:.3f}')
    ax.plot(time_window_centers/60,mod_corr[:,1],lw=2,zorder=2,color=usrplt.cc[2],label=f'Modularity_areas: {cc1:.3f}')
    ax.set_ylabel('Modularity')
    # ax.set_xlabel('Center of moving window (min)')

    usrplt.adjust_spines(ax)
    ylim = ax.get_ylim()

    if injection_times is not None:
        ax.vlines(np.array(injection_times[0])/60,*ylim,color=injection_colors[0],label=f'{injection_types[0]} injection')
        ax.vlines(np.array(injection_times[1])/60,*ylim,color=injection_colors[1],label=f'{injection_types[1]} injection')
    ax.legend(loc=1)

    for tW in evoked_time_window_list:
        # ax.fill_between(np.array(tW)/60,0,6,color=c,lw=1.5,zorder=0,alpha=0.5)
        ax.fill_between(np.array(tW)/60,*ylim,color=usrplt.cc[8],alpha=0.5,zorder=0)

    ax = axes[1]
    ax.set_title('Locomotion')
    ax.errorbar(time_window_centers/60,running_moments[:,0],yerr=running_moments[:,1],color='k')
    ax.set_ylabel('Mean running speed')
    usrplt.adjust_spines(ax)

    ax = axes[2]
    ax.set_title('Pupil')
    if len(pupil_moments) > 2:
        ax.errorbar(time_window_centers/60,pupil_moments[:,0],yerr=pupil_moments[:,1],color=usrplt.cc[8])
    ax.set_ylabel('Mean pupil diameter')
    usrplt.adjust_spines(ax)
    ax.set_xlabel('Center of moving window (min)')

    pdfdoc.savefig(fig)
    plt.savefig(join(PlotDir,f'modularity_corr-mat_{mID}_{rec_name}.png'),facecolor='white',dpi=300,bbox_inches='tight')
    plt.close(fig)

    ## Plot modularity ----------------------------------------
    fig, axes = plt.subplots(3,1,figsize=(10,8),gridspec_kw = {'height_ratios': [4,2,2],'hspace':0.4})
    plt.suptitle(f'Modularity on rolling FCF matrices; {mID}, {rec_name}')

    cc0 = np.corrcoef(running_moments[:,0],mod_FCF[:,0])[0,1]
    cc1 = np.corrcoef(running_moments[:,0],mod_FCF[:,1])[0,1]
    ax = axes[0]
    ax.plot(time_window_centers/60,mod_FCF[:,0],lw=2,zorder=2,color=usrplt.cc[1],label=f'Modularity_algorithm: {cc0:.3f}')
    ax.plot(time_window_centers/60,mod_FCF[:,1],lw=2,zorder=2,color=usrplt.cc[2],label=f'Modularity_areas: {cc1:.3f}')
    ax.set_ylabel('Modularity')
    # ax.set_xlabel('Center of moving window (min)')

    usrplt.adjust_spines(ax)
    ylim = ax.get_ylim()

    if injection_times is not None:
        ax.vlines(np.array(injection_times[0])/60,*ylim,color=injection_colors[0],label=f'{injection_types[0]} injection')
        ax.vlines(np.array(injection_times[1])/60,*ylim,color=injection_colors[1],label=f'{injection_types[1]} injection')
    ax.legend(loc=1)

    for tW in evoked_time_window_list:
        # ax.fill_between(np.array(tW)/60,0,6,color=c,lw=1.5,zorder=0,alpha=0.5)
        ax.fill_between(np.array(tW)/60,*ylim,color=usrplt.cc[8],alpha=0.5,zorder=0)

    ax = axes[1]
    ax.set_title('Locomotion')
    ax.errorbar(time_window_centers/60,running_moments[:,0],yerr=running_moments[:,1],color='k')
    ax.set_ylabel('Mean running speed')
    usrplt.adjust_spines(ax)

    ax = axes[2]
    ax.set_title('Pupil')
    if len(pupil_moments) > 2:
        ax.errorbar(time_window_centers/60,pupil_moments[:,0],yerr=pupil_moments[:,1],color=usrplt.cc[8])
    ax.set_ylabel('Mean pupil diameter')
    usrplt.adjust_spines(ax)
    ax.set_xlabel('Center of moving window (min)')

    pdfdoc.savefig(fig)
    plt.savefig(join(PlotDir,f'modularity_FCF-mat_{mID}_{rec_name}.png'),facecolor='white',dpi=300,bbox_inches='tight')
    plt.close(fig)

 
    FCF_all = np.array(FCF_all)
    import itertools as it
    
    supergroups_of_interest = ['CTX','TH']
    sg_pairs = list(it.product(supergroups_of_interest,repeat=2))
    tmp_list = []
    num_significant = []

    FC_traces = np.zeros((nWindows,len(sg_pairs),4))
    sig_frac = np.zeros((nWindows,len(sg_pairs)))
    FC_thresh = 0.1
    for iS, epoch in enumerate(epoch_list):
        tmp_mat = FCF_all[iS].copy()

        conn_type = []
        for jj, (g1, g2) in enumerate(sg_pairs):
            indy1 = np.where(supergroups == g1)[0]
            indy2 = np.where(supergroups == g2)[0]

            #Get FC data
            FC = tmp_mat[indy2][:,indy1].ravel()
            mask = FC > FC_thresh
            n = len(FC)

            #Calculate mean and iqr
            FC_traces[iS,jj,0] = np.nanmean(FC)
            FC_traces[iS,jj,1] = st.sem(FC)

            FC_traces[iS,jj,2] = np.nanmean(FC[mask])
            FC_traces[iS,jj,3] = st.sem(FC[mask])

            gg = f'{g1}/{g2}'
            conn_type.append(gg)
            tmp_list.append((np.repeat(epoch,n),np.repeat(gg,n),FC))

            
            nFrac = np.sum(mask)/len(mask)
            sig_frac[iS,jj] = nFrac
            num_significant.append((epoch,gg,nFrac))

    FC_df = pd.DataFrame(np.hstack(tmp_list).T,columns = ['epoch','type','FC'])
    FC_df = FC_df.astype({'epoch':str,'type':str,'FC':float})
    FC_df['sig'] = FC_df['FC'] > 0.1

    sig_df = pd.DataFrame(np.stack(num_significant,axis=1).T,columns = ['epoch','type','frac'])
    sig_df = sig_df.astype({'epoch':str,'type':str,'frac':float})


    sig_df.to_csv(join(SaveDir,f'sig_df_{mID}_{rec_name}.csv'),index=False)
    FC_df.to_csv(join(SaveDir,f'FC_df_{mID}_{rec_name}.csv'),index=False)

    ## Plot FC over time ----------------------------------------
    fig, axes = plt.subplots(5,1,figsize=(12,12),gridspec_kw = {'height_ratios': [4,4,4,1.5,1.5],'hspace':0.4})
    plt.suptitle(f'Supergroup FC over time; {mID}, {rec_name}',y=0.925)

    for ii in range(2):
        ax = axes[2*ii]
        for jj, ct in enumerate(conn_type):
            cc0 = np.corrcoef(running_moments[:,0],FC_traces[:,jj,2*ii])[0,1]; cc01 = np.corrcoef(running_moments[:,1],FC_traces[:,jj,2*ii])[0,1]
            ax.errorbar(time_window_centers/60,FC_traces[:,jj,2*ii],yerr=FC_traces[:,jj,2*ii+1],lw=2,zorder=2,color=usrplt.cc[jj],label=f'{ct}: {cc0:.2f}')

    ax = axes[1]
    for jj, ct in enumerate(conn_type):
        cc0 = np.corrcoef(running_moments[:,0],sig_frac[:,jj])[0,1]; cc01 = np.corrcoef(running_moments[:,1],sig_frac[:,jj])[0,1]
        ax.plot(time_window_centers/60,sig_frac[:,jj],lw=2,zorder=2,color=usrplt.cc[jj],label=f'{ct}: {cc0:.2f}')

    for ii in range(3):
        ax = axes[ii]
        # ax.autoscale(tight=True)
        usrplt.adjust_spines(ax)
        ylim = ax.get_ylim()
        if injection_times is not None:
            ax.vlines(np.array(injection_times[0])/60,*ylim,color=injection_colors[0])#,label=f'{injection_types[0]} injection')
            ax.vlines(np.array(injection_times[1])/60,*ylim,color=injection_colors[1])#,label=f'{injection_types[1]} injection')

        for tW in evoked_time_window_list:
            ax.fill_between(np.array(tW)/60,*ylim,color=usrplt.cc[8],alpha=0.5,zorder=0)

        ax.legend(loc=2)
    # axes[0].legend()
    axes[0].set_title('FC of all connections');axes[0].set_ylabel('FC')
    axes[1].set_title('Fraction of significant connections (FC > 0.1)');axes[1].set_ylabel('Fraction')
    axes[2].set_title('FC of significant connections (FC > 0.1)');axes[2].set_ylabel('FC')


    ax = axes[3]
    # ax.set_title('Behavior')
    ax.set_title('Running speed')
    ax.errorbar(time_window_centers/60,running_moments[:,0],yerr=running_moments[:,1],color='k')
    ax.set_ylabel('Speed')
    usrplt.adjust_spines(ax)
    # ax.set_xlabel('Center of moving window (min)')
    usrplt.adjust_spines(ax)

    ax2 = axes[4]
    # ax2 = ax.twinx()
    ax2.set_title('Pupil size')
    if len(pupil_moments) > 2:
        ax2.errorbar(time_window_centers/60,pupil_moments[:,0],yerr=pupil_moments[:,1],color=usrplt.cc[8])
    ax2.set_ylabel('Radius',color=usrplt.cc[8])
    usrplt.adjust_spines(ax2)
    ax2.set_xlabel('Center of moving window (min)')
    
    pdfdoc.savefig(fig)
    plt.savefig(join(PlotDir,f'rolling-FC_{mID}_{rec_name}.pdf'),facecolor='white',dpi=300,bbox_inches='tight')
    plt.close(fig)

    ## CLOSE files ----------------------------------------
    pdfdoc.close()

    




