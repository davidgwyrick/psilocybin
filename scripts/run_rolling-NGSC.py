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
import numpy as np
import pandas as pd
from tqdm import tqdm

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

parser.add_argument('--tWindow_shift_min',type=float, default=0.5,
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

    tmp_list = sorted(glob(join(TempDir,f'NGSC_run_*')))
    if len(tmp_list) == 0:
        curr_run = 0
    else:
        last_run = int(tmp_list[-1][-1])
        curr_run = last_run+1
    folder = f'NGSC_run_{curr_run:02d}'
    
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
    run_ts, run_signal, run_signal_s, acceleration, pupil_ts, pupil_radius, pupil_dxdt, plot_pupil = util.get_behavioral_data(exp,mID,rec_name)
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
            macro_names = ['pre-inj','post-sal1-inj','post-psi-inj']
            cmap = sns.xkcd_palette(['silver','dark sky blue','darkish red'])
        elif drug_type == 'saline':
            injection_types = ['sal1', 'sal2']
            injection_colors = sns.xkcd_palette(['dark sky blue','cobalt blue'])
            macro_names = ['pre-inj','post-sal1-inj','post-sal2-inj']
            cmap = sns.xkcd_palette(['silver','dark sky blue','cobalt blue'])
        elif drug_type == 'ketanserin+psilocybin':
            injection_types = ['ket','psi']
            injection_colors = sns.xkcd_palette(['magenta','goldenrod'])
            macro_names = ['pre-inj','post-ket-inj','post-psi-inj']
            cmap = sns.xkcd_palette(['silver','magenta','goldenrod'])
        inj_tuple = (injection_times, injection_types, injection_colors)
    else:
        injection_times = None

    #For isoflurane experiments, get iso level
    iso_induction_times = None; iso_maintenance_times = None; iso_tuple = None
    if drug_type == 'isoflurane':
        iso_level, iso_times = exp.load_analog_iso()
        iso_induction_times, iso_maintenance_times = exp.load_iso_times()
        induction_colors = sns.xkcd_palette(['light teal','teal'])
        iso_tuple = (iso_induction_times, induction_colors)
        macro_names = ['pre-iso','iso-ind','post-iso']
        cmap = sns.xkcd_palette(['silver','light teal','teal'])

    # extract the timestamps of the selected stimuli
    try:
        stim_log = pd.read_csv(exp.stimulus_log_file)
        stim_exists = True
    except:
        stim_exists = False

    ## Determine windows ----------------------------------------
    evoked_time_window_list = []; evoked_type_list = []; evoked_tuple = None
    if stim_exists:
        for s in np.unique(stim_log['sweep']):
            for t in np.unique(stim_log.loc[stim_log.sweep == s]['stim_type']):
                sub_df = stim_log.loc[(stim_log.sweep == s) & (stim_log.stim_type == t)]
                tS = sub_df['onset'].min(); tE = sub_df['offset'].max()
                
                evoked_time_window_list.append([tS,tE])
                evoked_type_list.append(t)
        evoked_tuple = (evoked_time_window_list,evoked_type_list)
    
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

    #Preallocate
    t0_outer = time.perf_counter()
    covariance_all = np.zeros((nWindows,N,N))
    participation_ratio = np.zeros((nWindows))
    NGSC = np.zeros((nWindows))

    #Loop over different time blocks
    for ii, tW in enumerate(tqdm(time_window_list)):

        # Select time window
        tslice = np.where((ts >= tW[0]) & (ts < tW[1]))[0]
        X_sub = X[tslice]
        T, N = X_sub.shape

        #Calculate covariance for all neurons
        data_cov = np.cov(X_sub.T)
        covariance_all[ii] = data_cov
        #Calculate eigenvalues of covariance matrix
        eigvals, eigvecs = np.linalg.eig(data_cov)
        eigvals_sort = np.sort(eigvals)[::-1]

        #Normalize eigenvalues
        norm_eigvals = eigvals_sort / np.sum(eigvals_sort)

        #Calculate participation ratio
        participation_ratio[ii] = (np.sum(eigvals)**2/np.sum(eigvals**2))/N

        #Calculate normalized global spatial complexity
        NGSC[ii] = -1*np.nansum(norm_eigvals*np.log(norm_eigvals)/np.log(N))

    comp_length = time.perf_counter() - t0_outer
    mm, ss = divmod(comp_length,60)
    hh, mm = divmod(mm, 60)
    print(f'Completed in {hh:.0f} hrs, {mm:.0f} minutes, {ss:.0f} seconds')

    #Save results
    np.savez(join(SaveDir,f'NGSC_results_{curr_run:02d}.npz'),covariance_all=covariance_all,participation_ratio=participation_ratio,NGSC=NGSC)

    ## ----------------------------------------
    #NGSC plot
    fig, axes = plt.subplots(3,1,figsize=(10,8),gridspec_kw = {'height_ratios': [4,2,2],'hspace':0.4})
    plt.suptitle(f'Normalized global spatial complexity; {mID}, {rec_name}')

    ax = axes[0]
    ax.plot(time_window_centers/60,NGSC,lw=2,zorder=2,color=usrplt.cc[1])
    ax.set_ylabel('NGSC')
    ax.set_xlabel('Center of moving window (min)')

    usrplt.adjust_spines(ax)
    ylim = ax.get_ylim()
    if injection_times is not None:
        ax.vlines(np.array(injection_times[0])/60,*ylim,color=injection_colors[0],label=f'{injection_types[0]} injection')
        ax.vlines(np.array(injection_times[1])/60,*ylim,color=injection_colors[1],label=f'{injection_types[1]} injection')

    for tW in evoked_time_window_list:
        # ax.fill_between(np.array(tW)/60,0,6,color=c,lw=1.5,zorder=0,alpha=0.5)
        ax.fill_between(np.array(tW)/60,*ylim,color=usrplt.cc[8],alpha=0.5,zorder=0)

    ax = axes[1]
    ax.set_title('Locomotion')
    cc = np.corrcoef(running_moments[:,0],NGSC)[0,1]
    ax.errorbar(time_window_centers/60,running_moments[:,0],yerr=running_moments[:,1],color='k',label=f'\u03C1 (PC, pupil) = {cc:.2f}')
    ax.set_ylabel('Mean running speed')
    usrplt.adjust_spines(ax)
    ax.legend(loc=2,framealpha=1)

    ax = axes[2]
    ax.set_title('Pupil')
    cc = np.corrcoef(pupil_moments[:,0],NGSC)[0,1]
    if len(pupil_moments) > 2:
        ax.errorbar(time_window_centers/60,pupil_moments[:,0],yerr=pupil_moments[:,1],color=usrplt.cc[8],label=f'\u03C1 (PC, pupil) = {cc:.2f}')
    ax.set_ylabel('Mean pupil diameter')
    usrplt.adjust_spines(ax)
    ax.set_xlabel('Center of moving window (min)')
    ax.legend(loc=2,framealpha=1)

    plt.savefig(join(SaveDir,f'NGSC_{mID}_{rec_name}.png'),facecolor='white',dpi=300,bbox_inches='tight')
    plt.savefig(join(ProjDir,'plots','NGSC',f'NGSC_{mID}_{rec_name}_{time_bin_ms}.png'),facecolor='white',dpi=300,bbox_inches='tight')
    plt.close(fig)


    ## ----------------------------------------
    #Participation_ratio plot
    fig, axes = plt.subplots(3,1,figsize=(10,8),gridspec_kw = {'height_ratios': [4,2,2],'hspace':0.4})
    plt.suptitle(f'Participation ratio; {mID}, {rec_name}')

    ax = axes[0]
    ax.plot(time_window_centers/60,participation_ratio,lw=2,zorder=2,color=usrplt.cc[1])
    ax.set_ylabel('Participation ratio')
    ax.set_xlabel('Center of moving window (min)')

    usrplt.adjust_spines(ax)
    ylim = ax.get_ylim()
    if injection_times is not None:
        ax.vlines(np.array(injection_times[0])/60,*ylim,color=injection_colors[0],label=f'{injection_types[0]} injection')
        ax.vlines(np.array(injection_times[1])/60,*ylim,color=injection_colors[1],label=f'{injection_types[1]} injection')

    for tW in evoked_time_window_list:
        # ax.fill_between(np.array(tW)/60,0,6,color=c,lw=1.5,zorder=0,alpha=0.5)
        ax.fill_between(np.array(tW)/60,*ylim,color=usrplt.cc[8],alpha=0.5,zorder=0)

    ax = axes[1]
    ax.set_title('Locomotion')
    cc = np.corrcoef(running_moments[:,0],participation_ratio)[0,1]
    ax.errorbar(time_window_centers/60,running_moments[:,0],yerr=running_moments[:,1],color='k',label=f'\u03C1 (PC, pupil) = {cc:.2f}')
    ax.set_ylabel('Mean running speed')
    usrplt.adjust_spines(ax)
    ax.legend(loc=2,framealpha=1)

    ax = axes[2]
    ax.set_title('Pupil')
    cc = np.corrcoef(pupil_moments[:,0],participation_ratio)[0,1]
    if len(pupil_moments) > 2:
        ax.errorbar(time_window_centers/60,pupil_moments[:,0],yerr=pupil_moments[:,1],color=usrplt.cc[8],label=f'\u03C1 (PC, pupil) = {cc:.2f}')
    ax.set_ylabel('Mean pupil diameter')
    usrplt.adjust_spines(ax)
    ax.set_xlabel('Center of moving window (min)')
    ax.legend(loc=2,framealpha=1)

    plt.savefig(join(SaveDir,f'PR_{mID}_{rec_name}.png'),facecolor='white',dpi=300,bbox_inches='tight')
    plt.savefig(join(ProjDir,'plots','NGSC',f'PR_{mID}_{rec_name}_{time_bin_ms}.png'),facecolor='white',dpi=300,bbox_inches='tight')
    plt.close(fig)

