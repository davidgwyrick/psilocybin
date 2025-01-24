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

#Plot
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from pptx import Presentation

#Project
from tbd_eeg.data_analysis.eegutils import EEGexp
import tbd_eeg.data_analysis.Utilities.utilities as tbd_util

#User
sys.path.append(ProjDir)
import util
import plotting as usrplt


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
parser.add_argument('--mID',type=str, default='mouse728449',
                    help='mouse to perform analysis on')

parser.add_argument('--rec_name',type=str, default='aw_sal_2024-04-10_10-04-50',
                    help='experiment to perform analysis on')

parser.add_argument('--time_bin_ms',type=int, default=100,
                    help='Time bin width')

parser.add_argument('--tWindow_width_min',type=int, default=2,
                    help='Time window over which to calculate FC')

parser.add_argument('--tWindow_shift_min',type=float, default=2,
                    help='Amount of time to shift rolling window by')

parser.add_argument('--fr_thresh',type=float, default=0,
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
    tWindow_width_min = args.tWindow_width_min
    tWindow_width = tWindow_width_min*60
    tWindow_shift_min = args.tWindow_shift_min

    #Data preprocessin
    fr_thresh = args.fr_thresh
    
    ## FOLDERS ----------------------------------------
    #Create directory for saving to
    TempDir = os.path.join(ServDir,'results','NGSC',mID,rec_name)
    if not os.path.exists(TempDir):
        os.makedirs(TempDir)

    tmp_list = sorted(glob(join(TempDir,f'corr_run_*')))
    if len(tmp_list) == 0:
        curr_run = 0
    else:
        last_run = int(tmp_list[-1][-1])
        curr_run = last_run+1
    folder = f'corr_run_{curr_run:02d}'
    
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
    with open(join(SaveDir,f"parameters_run_{curr_run}.json"), "w") as outfile:
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
    probe_unit_data, probe_info, total_units = tbd_util.get_neuropixel_data(exp)
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
    run_ts, raw_run_signal, run_signal, run_signal_s, pupil_ts, pupil_radius, run_signal_p, run_signal_p_s, plot_pupil = util.get_behavioral_data(exp,mID,rec_name)

    #For saline & psilocybin experiments, get injection times and types of injection
    injection_times = None
    injection_time_windows = None
    if (drug_type == 'ketanserin+psilocybin') | (mID == 'mouse728449') | (mID == 'mouse724057'):
        injection_time_windows = np.array([np.array(exp_df['First injection time (s)'].values[0].split(','),dtype=float),
                                np.array(exp_df['Second injection time (s)'].values[0].split(','),dtype=float)])
        
        #Take second time in each window as "injection time"
        injection_times = [injection_time_windows[0,1],injection_time_windows[1,1]]
    elif drug_type in ['saline', 'psilocybin']:
        injection_times = [float(exp_df['First injection time (s)'].values[0]),
                        float(exp_df['Second injection time (s)'].values[0])]

    if drug_type == 'psilocybin':
        injection_types = ['sal1','psi']
        injection_colors = ['g','r']
        cmap = np.concatenate((sns.color_palette('Blues',2),sns.color_palette('Reds',1)))
    elif drug_type == 'saline':
        injection_types = ['sal1', 'sal2']
        injection_colors = sns.color_palette('Greens',2)
        cmap = sns.color_palette('Greens',3)
    elif drug_type == 'ketanserin+psilocybin':
        injection_types = ['ket','psi']
        injection_colors = sns.xkcd_palette(['goldenrod','darkish red'])
        cmap = sns.xkcd_palette(['silver','goldenrod','darkish red'])

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
    tW_starts = np.arange(open_ephys_start,open_ephys_end-tWindow_width,tWindow_shift_min*60)
    tW_ends = tW_starts + tWindow_width
    time_window_array = np.array((tW_starts,tW_ends)).T
    time_window_list = time_window_array.tolist()
    time_window_centers = time_window_array[:,0] + tWindow_width/2
    nWindows = len(time_window_list)
    print(f'{nWindows} windows, {tWindow_width_min} mins long, separeted by {tWindow_shift_min} min')

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

        if drug_type in ['saline', 'psilocybin','ketanserin+psilocybin']: 
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
    FR = np.concatenate([probe_unit_data[probei]['firing_rate'] for probei in probe_list])
    FR_mask = FR > fr_thresh
    neuron_indices = np.where(FR_mask)[0]
    N = len(neuron_indices)
    print(f'{N} neurons > {fr_thresh} Hz overall')
    

    #Read in first window for area definitions
    tW = time_window_list[0]
    data_list, ts_list, _, plot_tuple, _ = util.bin_spiking_data(probe_unit_data, [[open_ephys_start,open_ephys_end]], time_bin=time_bin,fr_thresh=fr_thresh)
    if len(plot_tuple) == 11:
        boundaries_group, ticks_group, labels_group, celltypes, durations, layers, areas, groups,mesogroups, supergroups, order_by_group = plot_tuple

        #Ordered by area
        areas_sub = areas[neuron_indices]
        groups_sub = groups[neuron_indices]
        supergroups_sub = supergroups[neuron_indices]
        celltypes_sub = celltypes[neuron_indices]

        areas_ro = areas_sub[order_by_group]
        groups_ro = groups_sub[order_by_group]
        supergroups_ro = supergroups_sub[order_by_group]
        celltypes_ro = celltypes_sub[order_by_group]
        area_info = True
        sort_by_area = True

        #Calculate FCF in this order
        unique_sorted, uniq_indices = np.unique(areas_ro, return_index=True)
        uniq_areas_order = unique_sorted[np.argsort(uniq_indices)]

    else:
        #Ordered by probe
        boundaries, ticks, labels = plot_tuple
        sort_by_area = False

    X = data_list[0][:,neuron_indices]; ts = ts_list[0]

    if sort_by_area:
        X = X[:,order_by_group] #Critical! I'm reordering the data here, before the calculation of FC # T x N
    
    #Calculate firing rate per block
    FR_per_block = np.zeros((nWindows,N))
    for ii, tW in enumerate(time_window_list):
        # Select time window
        tslice = np.where((ts >= tW[0]) & (ts < tW[1]))[0]
        X_sub = X[tslice]
        FR_per_block[ii] = np.sum(X_sub,axis=0)/(tW[1]-tW[0])

    #Calculate population mean of RS cells
    pop_traces = []
    area_list = []

    for a in uniq_areas_order:
        if a in ['nan','root','null']:
            continue
        # for ct in ['RS','FS']:
        ct = 'RS'
        indy = np.where((areas_ro == a) & (celltypes_ro == ct))[0]
        if len(indy) < 5:
            continue
        else:
            pop_spks = np.sum(X[:,indy],axis=1)
        pop_traces.append(pop_spks)
        area_list.append(a)

    areas_pop = np.array(area_list)
    groups_pop, _, _, _, _, supergroups_pop = util.determine_groups(areas_pop) 

    
    np.savez(join(os.path.join(TempDir,f'corr_run_01'),f'area_information_{mID}_{rec_name}.npz'),FR_per_block=FR_per_block,FR=FR,areas=areas_ro, groups=groups_ro, supergroups=supergroups_ro, celltypes=celltypes_ro, areas_pop = areas_pop, groups_pop = groups_pop, supergroups_pop = supergroups_pop)
    # np.savez(join(os.path.join(TempDir,f'corr_run_01'),f'area_information_{mID}_{rec_name}.npz'),areas=areas, groups=groups, supergroups=supergroups, celltypes=celltypes, neuron_indices=neuron_indices,order_by_group=order_by_group, areas_pop = areas_pop, groups_pop = groups_pop, supergroups_pop = supergroups_pop)
    # np.savez(join(SaveDir,f'area_information_{mID}_{rec_name}.npz'),areas=areas, groups=groups, supergroups=supergroups, celltypes=celltypes, neuron_indices=neuron_indices,order_by_group=order_by_group, areas_pop = areas_pop, groups_pop = groups_pop, supergroups_pop = supergroups_pop)
    # exit()
    #population traces
    X_pop = np.array(pop_traces).T
    N_pop = len(areas_pop)

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
        else:
            pupil_moments.append((np.nan,np.nan))
    running_moments = np.array(running_moments)
    pupil_moments = np.array(pupil_moments)

    t0_outer = time.perf_counter()
    correlation_all = np.zeros((nWindows,N,N))
    covariance_all = np.zeros((nWindows,N,N))
    participation_ratio = np.zeros((nWindows,4))
    correlation_pop = np.zeros((nWindows,N_pop,N_pop))

    #Loop over different time blocks
    for ii, tW in enumerate(time_window_list):

        # Select time window
        tslice = np.where((ts >= tW[0]) & (ts < tW[1]))[0]
        X_sub = X[tslice]
        T, N = X_sub.shape

        #Calculate covariance for all neurons
        cov_mat = np.cov(X_sub.T)
        covariance_all[ii] = cov_mat
        eigenvalues = np.linalg.eigvals(cov_mat)
        participation_ratio[ii,0] = (np.sum(eigenvalues)**2/np.sum(eigenvalues**2))/N

        #Calculate correlation for all pairs of neurons
        corr = np.corrcoef(X_sub.T)
        correlation_all[ii] = corr

        #Calculate correlation for all pairs of population vectors
        correlation_pop[ii] = np.corrcoef(X_pop[tslice].T)

        if sort_by_area:
            #Calculate covariance for cortical connections
            indy1 = np.where((supergroups_ro == 'CTX'))[0]
            cov_mat_sub = cov_mat[indy1][:,indy1]
            eigenvalues = np.linalg.eigvals(cov_mat_sub)
            participation_ratio[ii,1] = (np.sum(eigenvalues)**2/np.sum(eigenvalues**2))/len(indy1)

            #Calculate covariance for thalamic connections
            indy2 = np.where((supergroups_ro == 'TH'))[0]
            cov_mat_sub = cov_mat[indy2][:,indy2]
            eigenvalues = np.linalg.eigvals(cov_mat_sub)
            participation_ratio[ii,2] = (np.sum(eigenvalues)**2/np.sum(eigenvalues**2))/len(indy2)

    if sort_by_area:
        n1 = np.sum(supergroups_ro == 'CTX');n2 = np.sum(supergroups_ro == 'TH')
        normalization_factors = np.array([N,n1,n2])
    else:
        normalization_factors = np.array([N,N,N])

    correlation_pop2 = np.zeros((nWindows,N_pop,N_pop))
    for ii, tW in enumerate(time_window_list):
        corr_neuron = correlation_all[ii]
        
        for i, a_i in enumerate(areas_pop):
            indy_i = np.where((areas_ro == a_i) & (celltypes_ro == 'RS'))[0]
            for j, a_j in enumerate(areas_pop):
                indy_j = np.where((areas_ro == a_j) & (celltypes_ro == 'RS'))[0]

                corr_sub = corr_neuron[indy_i][:,indy_j].ravel()
                correlation_pop2[ii,i,j] = np.nanmean(corr_sub)

    correlation_all[np.isnan(correlation_all)] = 0

    pdfdoc = PdfPages(join(PlotDir,f'rolling_correlation_{mID}_{rec_name}.pdf'))
    clims1 = [0,np.round(np.nanpercentile(correlation_all,99),2)]
    # clims2 = [np.round(np.nanpercentile(np.triu(np.abs(correlation_pop2),2),2)),np.round(np.nanpercentile(np.triu(np.abs(correlation_pop2),2),98),2)]
    clims3 = [0,np.round(np.nanpercentile(np.triu(np.abs(correlation_pop),2),99),2)]
    for ii, tW in enumerate(time_window_list):
        
        fig, axes = plt.subplots(1,2,figsize=(10,5))
        plt.suptitle(f'{epoch_list[ii]}')
        #center=0,cmap='RdBu_r'
        usrplt.visualize_matrix(np.abs(correlation_all[ii]),ax=axes[0],clims=clims1,cmap='viridis',plot_ylabel=True,title='Neuron pairwise',cbar_label='| Correlation |',cbar=True,ticks=ticks,labels=labels,boundaries=None)
        # usrplt.visualize_matrix(np.abs(correlation_pop2[ii]),ax=axes[1],clims=clims2,cmap='viridis',plot_ylabel=True,title='Population average of matrix 1',cbar_label='| Correlation |',cbar=True,ticks=np.arange(len(areas_pop)),labels=areas_pop)
        usrplt.visualize_matrix(np.abs(correlation_pop[ii]),ax=axes[1],clims=clims3,cmap='viridis',plot_ylabel=True,title='Population pairwise',cbar_label='| Correlation |',cbar=True,ticks=np.arange(len(areas_pop)),labels=areas_pop)
        pdfdoc.savefig(fig)
        plt.close(fig)
    

    pdfdoc2 = PdfPages(join(PlotDir,f'rolling_covariance_{mID}_{rec_name}.pdf'))
    clims = [np.round(np.nanpercentile(covariance_all,2),2),np.round(np.nanpercentile(covariance_all,98),2)]
    clims2 = [np.round(np.nanpercentile(np.abs(correlation_all),2),2),np.round(np.nanpercentile(np.abs(correlation_all),98),2)]

    for ii, tW in enumerate(time_window_list):
        
        fig, axes = plt.subplots(1,2,figsize=(10,5))
        plt.suptitle(f'{epoch_list[ii]}')

        usrplt.visualize_matrix(covariance_all[ii],ax=axes[0],center=0,clims=clims,cmap='RdBu_r',plot_ylabel=True,title='Covariance',cbar_label='Covariance',cbar=True,ticks=ticks,labels=labels,boundaries=boundaries)
        usrplt.visualize_matrix(np.abs(correlation_all[ii]),ax=axes[1],clims=clims2,cmap='viridis',plot_ylabel=True,title='|Correlation |',cbar_label='| Correlation |',cbar=True,ticks=ticks,labels=labels,boundaries=boundaries)

        pdfdoc2.savefig(fig)
        plt.close(fig)
    #Save FC for this epoch
    np.savez(join(SaveDir,f'rolling-covariance_{mID}_{rec_name}.npz'),normalization_factors=normalization_factors,participation_ratio=participation_ratio,covariance=covariance_all,correlation_pop=correlation_pop,correlation=correlation_all,running_moments=running_moments,pupil_moments=pupil_moments,time_window_centers=time_window_centers,epoch_list=epoch_list,time_window_array=time_window_array)

    ## Plot participation ratio ----------------------------------------
    fig, axes = plt.subplots(3,1,figsize=(10,8),gridspec_kw = {'height_ratios': [4,2,2],'hspace':0.4})
    plt.suptitle(f'Participation ratio; {mID}, {rec_name}\n {tWindow_width_min} min window, {len(neuron_indices)} neurons')

    corr_with_running = np.zeros((3,2))
    for i in range(3):
        for j in range(2):
            corr_with_running[i,j] = np.corrcoef(running_moments[:,j],participation_ratio[:,i])[0,1]

    ax = axes[0]
    ax.plot(time_window_centers/60,participation_ratio[:,0],lw=2,zorder=2,color=usrplt.cc[1],label=f'All neurons: {corr_with_running[0,0]:.3f}, {corr_with_running[0,1]:.3f}')
    if sort_by_area:
        ax.plot(time_window_centers/60,participation_ratio[:,1],lw=2,zorder=2,color=usrplt.cc[2],label=f'CTX neurons: {corr_with_running[1,0]:.3f}, {corr_with_running[1,1]:.3f}')
        ax.plot(time_window_centers/60,participation_ratio[:,2],lw=2,zorder=2,color=usrplt.cc[3],label=f'TH neurons: {corr_with_running[2,0]:.3f}, {corr_with_running[2,1]:.3f}')
    ax.set_ylabel('Participation ratio')
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

    pdfdoc2.savefig(fig)
    plt.savefig(join(PlotDir,f'participation_ratio_{mID}_{rec_name}.png'),facecolor='white',dpi=300,bbox_inches='tight')
    plt.close(fig)
    pdfdoc2.close()
    
    # N = len()
    window_indices = np.arange(nWindows)
    results = []
    for ii in window_indices[::5]:
        e = epoch_list[ii]
        if drug_type == 'urethane':
            e = 'urethane'
        else:
            e = e.split('_')[1]
        results.append((mID,rec_name,e,participation_ratio[ii,0],participation_ratio[ii,1],participation_ratio[ii,2],running_moments[ii,0],running_moments[ii,1],pupil_moments[ii,0],pupil_moments[ii,1]))
    
    tmp_df = pd.DataFrame(np.stack(results),columns=['mID','rec_name','epoch','participation_ratio_all','participation_ratio_ctx','participation_ratio_th','running_speed_mean','running_speed_std','pupil_diameter_mean','pupil_diameter_std'])
    tmp_df = tmp_df.astype({'mID':str,'rec_name':str,'epoch':str,'participation_ratio_all':float,'participation_ratio_ctx':float,'participation_ratio_th':float,'running_speed_mean':float,'running_speed_std':float,'pupil_diameter_mean':float,'pupil_diameter_std':float})
    tmp_df.to_csv(join(SaveDir,f'participation_ratio_{mID}_{rec_name}.csv'),index=False)


    import itertools as it
    
    supergroups_of_interest = ['CTX','TH']
    sg_pairs = list(it.product(supergroups_of_interest,repeat=2))
    tmp_list = []
    num_significant = []

    FC_traces = np.zeros((nWindows,len(sg_pairs),4))
    sig_frac = np.zeros((nWindows,len(sg_pairs)))
    FC_thresh = 0.1
    for iS, epoch in enumerate(epoch_list):
        tmp_mat = correlation_all[iS].copy()

        conn_type = []
        for jj, (g1, g2) in enumerate(sg_pairs):
            indy1 = np.where(supergroups_ro == g1)[0]
            indy2 = np.where(supergroups_ro == g2)[0]

            #Get FC data
            FC = tmp_mat[indy2][:,indy1].ravel()
            mask = FC > FC_thresh
            n = len(FC)

            #Calculate mean and iqr
            FC_traces[iS,jj,0] = np.nanmean(FC)
            FC_traces[iS,jj,1] = st.sem(FC,nan_policy='omit')

            FC_traces[iS,jj,2] = np.nanmean(FC[mask])
            FC_traces[iS,jj,3] = st.sem(FC[mask],nan_policy='omit')

            nFrac = np.sum(mask)/len(mask)
            sig_frac[iS,jj] = nFrac
            gg = f'{g1}/{g2}'
            conn_type.append(gg)

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



    supergroups_of_interest = ['CTX','TH']
    sg_pairs = list(it.product(supergroups_of_interest,repeat=2))
    tmp_list = []
    num_significant = []

    FC_traces = np.zeros((nWindows,len(sg_pairs),4))
    sig_frac = np.zeros((nWindows,len(sg_pairs)))
    FC_thresh = 0.1
    for iS, epoch in enumerate(epoch_list):
        tmp_mat = correlation_pop[iS].copy()

        conn_type = []
        for jj, (g1, g2) in enumerate(sg_pairs):
            indy1 = np.where(supergroups_pop == g1)[0]
            indy2 = np.where(supergroups_pop == g2)[0]

            #Get FC data
            FC = tmp_mat[indy2][:,indy1].ravel()
            mask = FC > FC_thresh
            n = len(FC)

            #Calculate mean and iqr
            FC_traces[iS,jj,0] = np.nanmean(FC)
            FC_traces[iS,jj,1] = st.sem(FC,nan_policy='omit')

            FC_traces[iS,jj,2] = np.nanmean(FC[mask])
            FC_traces[iS,jj,3] = st.sem(FC[mask],nan_policy='omit')

            nFrac = np.sum(mask)/len(mask)
            sig_frac[iS,jj] = nFrac
            gg = f'{g1}/{g2}'
            conn_type.append(gg)

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
    plt.savefig(join(PlotDir,f'rolling-FC2_{mID}_{rec_name}.pdf'),facecolor='white',dpi=300,bbox_inches='tight')

    pdfdoc.close()




