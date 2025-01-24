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
parser.add_argument('--mID',type=str, default='mouse678912',
                    help='mouse to perform analysis on')

parser.add_argument('--rec_name',type=str, default='spont_aw_psi_2023-06-22_11-42-00',
                    help='experiment to perform analysis on')

parser.add_argument('--time_bin_ms',type=int, default=25,
                    help='Time bin width')

parser.add_argument('--tWindow_width_s',type=int, default=120,
                    help='Time window over which to calculate FC')

parser.add_argument('--tWindow_shift_s',type=float, default=20,
                    help='Amount of time to shift rolling window by')

parser.add_argument('--fr_thresh',type=float, default=2,
                    help='Firing rate threshold for neurons to include in analysis')


##===== ======= Argument End ======= =====##
##===== ============================ =====##

if __name__ == '__main__':

    ## Parse the arguments ----------------------------------------
    args = parser.parse_args()

    #Which experiment?
    mID = args.mID
    rec_name = args.rec_name
    print(f'{mID}, {rec_name}')

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
    TempDir = os.path.join(ServDir,'results','FC_ccg',mID,rec_name)
    if not os.path.exists(TempDir):
        os.makedirs(TempDir)

    tmp_list = sorted(glob(join(TempDir,f'ccg_run_*')))
    if len(tmp_list) == 0:
        curr_run = 0
    else:
        last_run = int(tmp_list[-1][-1])
        curr_run = last_run+1
    # curr_run = 2
    folder = f'ccg_run_{curr_run:02d}'
    SaveDir = os.path.join(TempDir,folder)
    if not os.path.exists(SaveDir):
        os.makedirs(SaveDir)
    
    PlotDir = join(SaveDir,'plots')
    if not os.path.exists(PlotDir):
        os.makedirs(PlotDir)

    #Save model parameters
    args_dict = args.__dict__
    args_dict['SaveDir'] = SaveDir

    with open(join(SaveDir,f"ccm_parameters_run_{curr_run}.json"), "w") as outfile:
        json.dump(args_dict, outfile)
    # exit()
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
    probe_unit_data, probe_info, total_units, metric_list = tbd_util.get_neuropixel_data(exp)
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

    injection_times = None
    injection_time_windows = None
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

        print(f'Injection times: {injection_times}')

    else:
        injection_times = None

    #For isoflurane experiments, get iso level
    if drug_type == 'isoflurane':
        iso_level, iso_times = exp.load_analog_iso()
        iso_induction_times, iso_maintenance_times = exp.load_iso_times()

    # extract the timestamps of the selected stimuli
    try:
        stim_log = pd.read_csv(exp.stimulus_log_file)
        stim_exists = True
    except:
        stim_exists = False

    ## Determine windows ----------------------------------------
    evoked_time_window_list = []
    evoked_type_list = []
    if stim_exists:
        for s in np.unique(stim_log['sweep']):
            for t in np.unique(stim_log.loc[stim_log.sweep == s]['stim_type']):
                sub_df = stim_log.loc[(stim_log.sweep == s) & (stim_log.stim_type == t)]
                tS = sub_df['onset'].min(); tE = sub_df['offset'].max()
                
                evoked_time_window_list.append([tS,tE])
                evoked_type_list.append(t)
    
    #Determine rolling windows
    epoch_size = tWindow_width; trial_length = 1
    tW_starts = np.arange(open_ephys_start,open_ephys_end,epoch_size)[:-1]
    tW_ends = tW_starts + epoch_size
    time_window_array = np.array([tW_starts,tW_ends]).T
    time_window_list = time_window_array.tolist()
    time_window_centers = time_window_array[:,0] + tWindow_width/2
    
    nWindows = len(time_window_list)
    print(f'{nWindows} windows, {tWindow_width} seconds long, separeted by {tWindow_shift} sec')

    # exit()
    #Create epoch names
    epoch_list = []
    for ii, tW in enumerate(time_window_list):
        tW_center = tW[0] + (tW[1]-tW[0])
 
        #Determine whether most of the data is in evoked or spontaneous periods
        window_type = 'spont'
        for epoch_type, epoch_window in zip(evoked_type_list,evoked_time_window_list):
            if (tW_center >= epoch_window[0]) & (tW_center < epoch_window[1]):
                window_type = epoch_type
                break
        
        # epoch_list.append(f'{window_type}-{ii:03d}_no-inj')
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
            
    ## Get neurons & order ----------------------------------------
    #Get overall firing rates
    # import pdb; pdb.set_trace()
    FR = np.concatenate([np.array(probe_unit_data[probei]['firing_rate']) for probei in probe_list])
    FR_mask = FR > fr_thresh
    neuron_indices = np.where(FR_mask)[0]
    N = len(neuron_indices)
    print(f'{N} neurons > {fr_thresh} Hz overall')
    
    #Read in first window for area definitions
    tW = time_window_list[0]
    data_list, ts_list, _, plot_tuple, _ = util.bin_spiking_data(probe_unit_data, [tW], time_bin=time_bin,fr_thresh=fr_thresh)
    if len(plot_tuple) == 11:
        boundaries, ticks, labels, celltypes, durations, layers, areas, groups, mesogroups, supergroups, order_by_group = plot_tuple
        areas_sub = areas[neuron_indices]
        groups_sub = groups[neuron_indices]
        supergroups_sub = supergroups[neuron_indices]
        celltypes_sub = celltypes[neuron_indices]

        areas_ro = areas_sub[order_by_group]
        groups_ro = groups_sub[order_by_group]
        supergroups_ro = supergroups_sub[order_by_group]
        celltypes_ro = celltypes_sub[order_by_group]
        area_info = True

        #Calculate FCF in this order
        unique_sorted, uniq_indices = np.unique(areas_ro, return_index=True)
        uniq_areas_order = unique_sorted[np.argsort(uniq_indices)]

        #Create community structure based on group definition
        group_comm_list = []
        for grp in labels:
            indy = np.where(groups_ro == grp)[0]
            group_comm_list.append(frozenset(indy))
        
    else:
        boundaries, ticks, labels = plot_tuple
        area_info = False
        print('No area information')

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

    #Save time windows areas=areas,groups=groups,areas_ro=areas_ro,groups_ro=groups_ro,supergroups_ro=supergroups_ro,order_by_group=order_by_group,
    np.savez(os.path.join(SaveDir,f'time_windows.npz'),time_window_list=time_window_list,epoch_list=epoch_list,neuron_indices=neuron_indices,
             running_moments=running_moments,pupil_moments=pupil_moments,areas=areas,groups=groups,supergroups=supergroups,celltypes=celltypes,order_by_group=order_by_group)
    

    # #Read in preprocessed data
    # tmp = np.load(join(ProjDir,'data','control_mouse_topPCsremoved',f'data_{rec_name}.npz'))
    # data_all = tmp['data']
    # ts_all = tmp['ts']
    sys.path.append('/home/david.wyrick/Git/functional-network')
    import ccg_library as ccg_lib

    ## Calculate rolling FCF ----------------------------------------
    t0_outer = time.perf_counter()
    takens_range = np.arange(1,31) 

    mod_corr = np.zeros((nWindows,2))
    mod_FCF = np.zeros((nWindows,2))
    mod_ccg = np.zeros((nWindows,2))

    FCF_all = []# np.zeros((nWindows,N,N))
    pdfdoc = PdfPages(join(PlotDir,f'rolling_CCG_{mID}_{rec_name}.pdf'))
    
    frac_conn_list = []
    FCF_thresh = 0.1
    #Loop over different time blocks
    for ii, tW in enumerate(time_window_list):
        t0 = time.perf_counter()
        print(epoch_list[ii])

        ## Calculate CCG ----------------------------------------
        trial_times = np.arange(tW[0],tW[1],1)
        
        #Get evoked spike counts centered around "trial" times
        spk_list = [];time_bin = 1/1000
        for probei in probe_list:
            evoked_spks, bins = util.get_evoked_spike_counts(probe_unit_data[probei]['spike_times'], probe_unit_data[probei]['spike_clusters'], probe_unit_data[probei]['units'], trial_times, 0, trial_length, time_bin)
            spk_list.append(evoked_spks)
            spk_ts_trial = bins[:-1] + time_bin/2
        evoked_activity = np.concatenate(spk_list)  #NxnTrialsxT
        data_sub = evoked_activity[neuron_indices]
        data_ro = np.ascontiguousarray(data_sub[order_by_group])
        
        #Calculate CCG
        ccg = ccg_lib.CCG(num_jitter=10, L=25, window=50, memory=False, use_parallel=True, num_cores=30)
        ccg_jitter_corrected, ccgs_uncorrected = ccg.calculate_mean_ccg_corrected(data_ro, disable=False)
        
        connection_detection = ccg_lib.SharpPeakIntervalDetection(max_duration=20, maxlag=25, n=4)
        significant_ccg,significant_confidence,significant_offset,significant_duration = connection_detection.get_significant_ccg(ccg_jitter_corrected)
        full_ccg, full_confidence, full_offset, full_duration = connection_detection.get_full_ccg(ccg_jitter_corrected)
        adjmat = ~np.isnan(significant_ccg)

        fig, axes = plt.subplots(1,2,figsize=(10,5))
        nConnFrac = np.sum(~np.isnan(significant_ccg))/np.prod(significant_ccg.shape)
        frac_conn_list.append(nConnFrac)
        plt.suptitle(f'{mID}, {rec_name}, 50ms window\nCCG, {epoch_list[ii]}, {tW[0]}-{tW[1]} {nConnFrac:.3f} connection fraction',y=0.96)
        tmp = significant_ccg.copy()
        vmax = np.nanpercentile(np.abs(tmp),98) #vmin=-4E-2,vmax=4E-2
        tmp[~adjmat] = full_ccg[~adjmat]
        axes[0].set_title('Significant cross-correlograms')
        sns.heatmap(tmp,cmap='RdBu_r',center=0,vmin=-1*vmax,vmax=vmax,square=True,cbar_kws={'label':'CCG','shrink':.5},ax=axes[0])

        tmp = significant_offset.copy()
        tmp[~adjmat] = full_offset[~adjmat]
        axes[1].set_title('Offset of connections')
        sns.heatmap(tmp,cmap='viridis',square=True,cbar_kws={'label':'Lag (ms)','shrink':.5},ax=axes[1])

        for ax in axes:
            ax.set_yticks(ticks,minor=True)
            ax.set_yticks(boundaries,minor=False)
            ax.hlines(boundaries,*ax.get_xlim(),color='k',lw=1)
            ax.set_yticklabels(labels,minor=True)
            ax.set_yticklabels([],minor=False)

            ax.set_xticks(ticks,minor=True)
            ax.set_xticks(boundaries,minor=False)
            ax.vlines(boundaries,*ax.get_ylim(),color='k',lw=1)
            ax.set_xticklabels(labels,minor=True,rotation=30)
            ax.set_xticklabels([],minor=False)

        
        plt.savefig(join(PlotDir,f'rolling_CCG_{epoch_list[ii]}.png'),facecolor='white',dpi=300,bbox_inches='tight')
        pdfdoc.savefig(fig)
        plt.close(fig)
        mask = ~np.isnan(significant_ccg)
        adjacency_mat = np.array(mask,dtype=int)

        #Define directed graph
        weighted_FC = significant_ccg.copy()
        weighted_FC[~mask] = 0
        DG_weight = nx.DiGraph(weighted_FC)

        #Find communities and calculate modularity
        comm_list = greedy_modularity_communities(DG_weight,weight='weight')
        mod_ccg[ii,0] = modularity(DG_weight,comm_list)
        mod_ccg[ii,1] = modularity(DG_weight,group_comm_list)

        np.savez(join(SaveDir,f'rolling-CCG_{epoch_list[ii]}.npz'),ccg_jitter_corrected=ccg_jitter_corrected,ccgs_uncorrected=ccgs_uncorrected,significant_ccg=significant_ccg,significant_confidence=significant_confidence,
                 significant_offset=significant_offset,significant_duration=significant_duration,mID=mID,rec_name=rec_name,tWindow=tW,epoch=epoch_list[ii],
                 full_ccg=full_ccg,full_confidence=full_confidence,full_offset=full_offset,full_duration=full_duration)
        tE = (time.perf_counter() - t0)/60
        print('\tCompleted in {:.2f} mins'.format(tE))
        # ## Calculate FCF ----------------------------------------
        # #Read in neuropixel data for spontaneous periods
        # time_bin = 25/1000
        # data_list, ts_list, _, plot_tuple, _ = util.bin_spiking_data(probe_unit_data, [tW], time_bin=time_bin,fr_thresh=fr_thresh)
        # data_sub = data_list[0][:,neuron_indices]
        # data_ro = data_sub[:,order_by_group] #Critical! I'm reordering the data here, before the calculation of FC
        # T, N = data_ro.shape
        # X = data_ro

        # T,N = X.shape
        # print('\tCalculating Correlation')
        # #Calculate correlation
        # correlation = E.correlation_FC(X)

        # #Define mask for correlation as well
        # tmp_corr = correlation.copy()
        # tmp_corr[np.diag_indices(N)] = 0
        # mask = tmp_corr > FCF_thresh
        # tmp_corr[~mask] = 0
        # UG_weight = nx.Graph(tmp_corr)

        # #Find communities and calculate modularity
        # comm_list = greedy_modularity_communities(UG_weight,weight='weight')
        # mod_corr[ii,0] = modularity(UG_weight,comm_list)
        
        # print('\tCalculating FCF')
        # FCF_takens = np.zeros((len(takens_range),N,N))  
        # #Calculate cross-mapping skill with set takens dimension
        # for kk, dim in enumerate(tqdm(takens_range)):
        #     edims = np.repeat(dim,N)
        #     FCF_takens[kk] = kedm.xmap(X,edims,tau=1)
        # FCF_optimal, complexity = util.determine_optimal_FCF(FCF_takens,takens_range)
        # FCF_optimal[np.diag_indices(N)] = np.nan
        # directionality = FCF_optimal - FCF_optimal.T

        # #Define adjacency matrix 
        # mask = FCF_optimal > FCF_thresh
        # adjacency_mat = np.array(mask,dtype=int)

        # #Define directed graph
        # weighted_FC = FCF_optimal.copy()
        # weighted_FC[~mask] = 0
        # DG_weight = nx.DiGraph(weighted_FC)

        # #Find communities and calculate modularity
        # comm_list = greedy_modularity_communities(DG_weight,weight='weight')
        # mod_FCF[ii,0] = modularity(DG_weight,comm_list)

        # #Calculate modulatity of group definitions of community
        # if area_info:
        #     mod_corr[ii,1] = modularity(UG_weight,group_comm_list)
        #     mod_FCF[ii,1] = modularity(DG_weight,group_comm_list)

        # #Plot FCF results
        # fig = usrplt.plot_CCM_results(correlation,FCF_optimal,directionality,complexity,vmax_fcf=None,dir_clims=None,vmax_takens=None,title=epoch_list[ii],ticks=ticks,labels=labels,boundaries=boundaries)
        
        # plt.savefig(join(PlotDir,f'rolling_FCF_{epoch_list[ii]}.png'),facecolor='white',dpi=300,bbox_inches='tight')
        # pdfdoc.savefig(fig)
        # plt.close(fig)
        # np.savez(join(SaveDir,f'rolling-FC_{epoch_list[ii]}.npz'),FCF_takens=FCF_takens,correlation=correlation,FCF_optimal=FCF_optimal,complexity=complexity,mID=mID,rec_name=rec_name,tWindow=tW,epoch=epoch_list[ii])

        # #Save FC for this epoch
        # # FCF_all.append(FCF_optimal)
        # tE = (time.perf_counter() - t0)/60
        # print('\tCompleted in {:.2f} mins'.format(tE))

        # fig, axes = plt.subplots(1,3,figsize=(15,5))
        # plt.suptitle(f'{mID}, {rec_name}, {epoch_list[ii]}')
        # ax = axes[0]
        # tmp_ccg = significant_ccg.copy()
        # vmax = np.nanpercentile(np.abs(tmp_ccg),98) #vmin=-4E-2,vmax=4E-2
        # mask = np.isnan(tmp_ccg)
        # tmp_ccg[mask] = 0 #ccg25[mask]
        # axes[0].set_title('Significant cross-correlograms')
        # sns.heatmap(tmp_ccg,cmap='RdBu_r',center=0,vmin=-1*vmax,vmax=vmax,square=True,cbar_kws={'label':'CCG','shrink':.5},ax=axes[0],rasterized=True)

        # ax = axes[1]
        # fcf = FCF_optimal.copy()
        # vmax_fcf = np.nanpercentile(np.abs(fcf),98)
        # sns.heatmap(fcf,square=True,annot=False,cmap='viridis',vmin=0, vmax=vmax_fcf,ax=ax,cbar_kws={'shrink':0.5,'ticks':[0,vmax_fcf]},rasterized=True) #,'label': 'FCF'

        # ax = axes[2]
        # conn_mask = ~np.isnan(significant_ccg)
        # x = significant_ccg[conn_mask].ravel()
        # y = fcf[conn_mask].ravel()
        # # ax.plot(x,y,'.',alpha=0.5)
        # usrplt.using_datashader(ax,x,y)
        # ax.set_xlabel('CCG')
        # ax.set_ylabel('FCF')
        
        # plt.savefig(join(PlotDir,f'FCFvsCCG_{epoch_list[ii]}.png'),facecolor='white',dpi=300,bbox_inches='tight')
        # pdfdoc.savefig(fig)
        # plt.close(fig)



    comp_length = time.perf_counter() - t0_outer
    mm, ss = divmod(comp_length,60)
    hh, mm = divmod(mm, 60)

    print(f'\n All windows calculated in {hh} hrs, {mm} minutes, {ss} seconds')
    np.savez(join(SaveDir,f'rolling-modularity_{mID}_{rec_name}.npz'),mod_FCF=mod_FCF,mod_corr=mod_corr,mod_ccg=mod_ccg,frac_conn_list=frac_conn_list)

    ## Plot modularity ----------------------------------------
    fig, axes = plt.subplots(3,1,figsize=(10,8),gridspec_kw = {'height_ratios': [4,2,2],'hspace':0.4})
    plt.suptitle(f'Fraction connected; {mID}, {rec_name}')

    cc0 = np.corrcoef(running_moments[:,0],frac_conn_list)[0,1]

    ax = axes[0]
    ax.plot(time_window_centers/60,frac_conn_list,lw=2,zorder=2,color=usrplt.cc[1],label=f'{cc0:.3f}')
    ax.set_ylabel('Fraction connected')
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
    plt.savefig(join(PlotDir,f'CCG_frac-connected_{mID}_{rec_name}.png'),facecolor='white',dpi=300,bbox_inches='tight')
    plt.close(fig)

    # ## Plot modularity ----------------------------------------
    # fig, axes = plt.subplots(3,1,figsize=(10,8),gridspec_kw = {'height_ratios': [4,2,2],'hspace':0.4})
    # plt.suptitle(f'Modularity on rolling FCF matrices; {mID}, {rec_name}')

    # cc0 = np.corrcoef(running_moments[:,0],mod_FCF[:,0])[0,1]
    # cc1 = np.corrcoef(running_moments[:,0],mod_FCF[:,1])[0,1]
    # ax = axes[0]
    # ax.plot(time_window_centers/60,mod_FCF[:,0],lw=2,zorder=2,color=usrplt.cc[1],label=f'Modularity_algorithm: {cc0:.3f}')
    # ax.plot(time_window_centers/60,mod_FCF[:,1],lw=2,zorder=2,color=usrplt.cc[2],label=f'Modularity_areas: {cc1:.3f}')
    # ax.set_ylabel('Modularity')
    # # ax.set_xlabel('Center of moving window (min)')

    # usrplt.adjust_spines(ax)
    # ylim = ax.get_ylim()

    # if injection_times is not None:
    #     ax.vlines(np.array(injection_times[0])/60,*ylim,color=injection_colors[0],label=f'{injection_types[0]} injection')
    #     ax.vlines(np.array(injection_times[1])/60,*ylim,color=injection_colors[1],label=f'{injection_types[1]} injection')
    # ax.legend(loc=1)

    # for tW in evoked_time_window_list:
    #     # ax.fill_between(np.array(tW)/60,0,6,color=c,lw=1.5,zorder=0,alpha=0.5)
    #     ax.fill_between(np.array(tW)/60,*ylim,color=usrplt.cc[8],alpha=0.5,zorder=0)

    # ax = axes[1]
    # ax.set_title('Locomotion')
    # ax.errorbar(time_window_centers/60,running_moments[:,0],yerr=running_moments[:,1],color='k')
    # ax.set_ylabel('Mean running speed')
    # usrplt.adjust_spines(ax)

    # ax = axes[2]
    # ax.set_title('Pupil')
    # if len(pupil_moments) > 2:
    #     ax.errorbar(time_window_centers/60,pupil_moments[:,0],yerr=pupil_moments[:,1],color=usrplt.cc[8])
    # ax.set_ylabel('Mean pupil diameter')
    # usrplt.adjust_spines(ax)
    # ax.set_xlabel('Center of moving window (min)')

    # pdfdoc.savefig(fig)
    # plt.savefig(join(PlotDir,f'modularity_FCF-mat_{mID}_{rec_name}.png'),facecolor='white',dpi=300,bbox_inches='tight')
    # plt.close(fig)

 
    # FCF_all = np.array(FCF_all)
    # import itertools as it
    
    # supergroups_of_interest = ['CTX','TH']
    # sg_pairs = list(it.product(supergroups_of_interest,repeat=2))
    # tmp_list = []
    # num_significant = []

    # FC_traces = np.zeros((nWindows,len(sg_pairs),4))
    # sig_frac = np.zeros((nWindows,len(sg_pairs)))
    # FC_thresh = 0.1
    # for iS, epoch in enumerate(epoch_list):
    #     tmp_mat = FCF_all[iS].copy()

    #     conn_type = []
    #     for jj, (g1, g2) in enumerate(sg_pairs):
    #         indy1 = np.where(supergroups_pop == g1)[0]
    #         indy2 = np.where(supergroups_pop == g2)[0]

    #         #Get FC data
    #         FC = tmp_mat[indy2][:,indy1].ravel()
    #         mask = FC > FC_thresh
    #         n = len(FC)

    #         #Calculate mean and iqr
    #         FC_traces[iS,jj,0] = np.nanmean(FC)
    #         FC_traces[iS,jj,1] = st.sem(FC)

    #         FC_traces[iS,jj,2] = np.nanmean(FC[mask])
    #         FC_traces[iS,jj,3] = st.sem(FC[mask])

    #         gg = f'{g1}/{g2}'
    #         conn_type.append(gg)
    #         tmp_list.append((np.repeat(epoch,n),np.repeat(gg,n),FC))

            
    #         nFrac = np.sum(mask)/len(mask)
    #         sig_frac[iS,jj] = nFrac
    #         num_significant.append((epoch,gg,nFrac))

    # FC_df = pd.DataFrame(np.hstack(tmp_list).T,columns = ['epoch','type','FC'])
    # FC_df = FC_df.astype({'epoch':str,'type':str,'FC':float})
    # FC_df['sig'] = FC_df['FC'] > 0.1

    # sig_df = pd.DataFrame(np.stack(num_significant,axis=1).T,columns = ['epoch','type','frac'])
    # sig_df = sig_df.astype({'epoch':str,'type':str,'frac':float})


    # sig_df.to_csv(join(SaveDir,f'sig_df_{mID}_{rec_name}.csv'),index=False)
    # FC_df.to_csv(join(SaveDir,f'FC_df_{mID}_{rec_name}.csv'),index=False)

    # ## Plot FC over time ----------------------------------------
    # fig, axes = plt.subplots(5,1,figsize=(12,12),gridspec_kw = {'height_ratios': [4,4,4,1.5,1.5],'hspace':0.4})
    # plt.suptitle(f'Supergroup FC over time; {mID}, {rec_name}',y=0.925)

    # for ii in range(2):
    #     ax = axes[2*ii]
    #     for jj, ct in enumerate(conn_type):
    #         cc0 = np.corrcoef(running_moments[:,0],FC_traces[:,jj,2*ii])[0,1]; cc01 = np.corrcoef(running_moments[:,1],FC_traces[:,jj,2*ii])[0,1]
    #         ax.errorbar(time_window_centers/60,FC_traces[:,jj,2*ii],yerr=FC_traces[:,jj,2*ii+1],lw=2,zorder=2,color=usrplt.cc[jj],label=f'{ct}: {cc0:.2f}')

    # ax = axes[1]
    # for jj, ct in enumerate(conn_type):
    #     cc0 = np.corrcoef(running_moments[:,0],sig_frac[:,jj])[0,1]; cc01 = np.corrcoef(running_moments[:,1],sig_frac[:,jj])[0,1]
    #     ax.plot(time_window_centers/60,sig_frac[:,jj],lw=2,zorder=2,color=usrplt.cc[jj],label=f'{ct}: {cc0:.2f}')

    # for ii in range(3):
    #     ax = axes[ii]
    #     # ax.autoscale(tight=True)
    #     usrplt.adjust_spines(ax)
    #     ylim = ax.get_ylim()
    #     if injection_times is not None:
    #         ax.vlines(np.array(injection_times[0])/60,*ylim,color=injection_colors[0])#,label=f'{injection_types[0]} injection')
    #         ax.vlines(np.array(injection_times[1])/60,*ylim,color=injection_colors[1])#,label=f'{injection_types[1]} injection')

    #     for tW in evoked_time_window_list:
    #         ax.fill_between(np.array(tW)/60,*ylim,color=usrplt.cc[8],alpha=0.5,zorder=0)

    #     ax.legend(loc=2)
    # # axes[0].legend()
    # axes[0].set_title('FC of all connections');axes[0].set_ylabel('FC')
    # axes[1].set_title('Fraction of significant connections (FC > 0.1)');axes[1].set_ylabel('Fraction')
    # axes[2].set_title('FC of significant connections (FC > 0.1)');axes[2].set_ylabel('FC')


    # ax = axes[3]
    # # ax.set_title('Behavior')
    # ax.set_title('Running speed')
    # ax.errorbar(time_window_centers/60,running_moments[:,0],yerr=running_moments[:,1],color='k')
    # ax.set_ylabel('Speed')
    # usrplt.adjust_spines(ax)
    # # ax.set_xlabel('Center of moving window (min)')
    # usrplt.adjust_spines(ax)

    # ax2 = axes[4]
    # # ax2 = ax.twinx()
    # ax2.set_title('Pupil size')
    # if len(pupil_moments) > 2:
    #     ax2.errorbar(time_window_centers/60,pupil_moments[:,0],yerr=pupil_moments[:,1],color=usrplt.cc[8])
    # ax2.set_ylabel('Radius',color=usrplt.cc[8])
    # usrplt.adjust_spines(ax2)
    # ax2.set_xlabel('Center of moving window (min)')
    
    # pdfdoc.savefig(fig)
    # plt.savefig(join(PlotDir,f'rolling-FC_{mID}_{rec_name}.pdf'),facecolor='white',dpi=300,bbox_inches='tight')
    # plt.close(fig)

    ## CLOSE files ----------------------------------------
    pdfdoc.close()

    




