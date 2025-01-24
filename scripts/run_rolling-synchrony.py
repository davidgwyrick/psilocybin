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
import ray

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

parser.add_argument('--dt_ms',type=int, default=10,
                    help='time bin width for phase calculation')

parser.add_argument('--tWindow_width_min',type=int, default=2,
                    help='Time window over which to calculate FC')

parser.add_argument('--tWindow_shift_min',type=float, default=2,
                    help='Amount of time to shift rolling window by')

parser.add_argument('--fr_thresh',type=float, default=0,
                    help='Firing rate threshold for neurons to include in analysis')



##===== ======= Argument End ======= =====##
##===== ============================ =====##

def get_phase_spikes(spk_times, tWindow, dt, method = 'continuous'):

    num_timesteps = int(np.ceil((tWindow[1] - tWindow[0])/dt) + 1)
    phase = np.zeros((num_timesteps)) 

    spikes = spk_times - tWindow[0]

    for k in range(len(spikes)):   
        index = int(np.ceil(spikes[k]/dt) )

        if k == 0 and len(phase) != 1:
            phase[index] = 0 
        else:
            index_prev = int(np.ceil(spikes[k-1]/dt))
            if index == index_prev:
                phase[index] += 2 * np.pi
            else:
                phase[index_prev:index+1] = phase[index_prev] + (np.arange(1, 1 + (index - index_prev + 1)) / (index - index_prev + 1)) * (2 * np.pi)

    if method == 'continuous':   
        phase[index+1:] = phase[index]
    elif method == 'nan':
        phase[index+1:] = np.NaN

    return phase

@ray.remote
def calculate_circular_variance(phase_i, phase_j, pba = None):

    if len(phase_i) == 0 or len(phase_j) == 0:
        S_ij = 0
        return S_ij
    
    delta_phi = np.mod(phase_i - phase_j, 2*np.pi)
    S_ij = np.sqrt(np.square(np.mean(np.cos(delta_phi))) + 
                                np.square(np.mean(np.sin(delta_phi))))
    
    #Update ray progress bar
    if pba is not None:
        pba.update.remote(1)

    return S_ij

if __name__ == '__main__':

    ## Parse the arguments ----------------------------------------
    args = parser.parse_args()

    #Which experiment?
    mID = args.mID
    rec_name = args.rec_name
    print(f'{mID}, {rec_name}')

    #How to segment data
    dt_ms = args.dt_ms
    dt = dt_ms/1000
    time_bin_ms = args.time_bin_ms
    time_bin = time_bin_ms/1000
    tWindow_width_min = args.tWindow_width_min
    tWindow_width = tWindow_width_min*60
    tWindow_shift_min = args.tWindow_shift_min

    #Data preprocessin
    fr_thresh = args.fr_thresh
    
    ## FOLDERS ----------------------------------------
    #Create directory for saving to
    TempDir = os.path.join(ServDir,'results','synchrony',mID,rec_name)
    if not os.path.exists(TempDir):
        os.makedirs(TempDir)

    tmp_list = sorted(glob(join(TempDir,f'synch_run_*')))
    if len(tmp_list) == 0:
        curr_run = 0
    else:
        last_run = int(tmp_list[-1][-1])
        curr_run = last_run+1
    folder = f'synch_run_{curr_run:02d}'
    
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
    data_list, ts_list, _, plot_tuple, _ = util.bin_spiking_data(probe_unit_data, [[open_ephys_start,open_ephys_end]], time_bin=time_bin,fr_thresh=fr_thresh)

    if len(plot_tuple) == 10:
        #Ordered by area
        boundaries, ticks, labels, celltypes, durations, layers, areas, groups, supergroups, order_by_group = plot_tuple
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

    np.savez(join(SaveDir,f'area_information_{mID}_{rec_name}.npz'),areas=areas, groups=groups, supergroups=supergroups, celltypes=celltypes, neuron_indices=neuron_indices,order_by_group=order_by_group)

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
    correlation = np.zeros((nWindows,N,N))
    synchrony = np.zeros((nWindows,N,N))

    global_synchrony = np.zeros((nWindows,3))
    #Loop over different time blocks
    for ii, tW in enumerate(time_window_list):
        print(f'{epoch_list[ii]}')
        # # Select time window
        # tslice = np.where((ts >= tW[0]) & (ts < tW[1]))[0]
        # X_sub = X[tslice]

        # #Calculate correlation for all pairs of neurons
        # corr = np.corrcoef(X_sub.T)
        # correlation[ii] = corr

        #Get spike times
        spike_time_dict = util.get_spike_time_dict(probe_unit_data, tWindow=tW)

        #Calculate instantaneous phase for each neuron
        phase_list = []
        for i, spk_times in spike_time_dict.items():
            if len(spk_times) == 0:
                phase = []
            else:
                phase = get_phase_spikes(spk_times, tW, dt)
            phase_list.append(phase)

        tmp_synchrony = np.zeros((N,N))
        for i, j in tqdm(it.combinations(range(N),2)):
            phase_i = phase_list[i]
            phase_j = phase_list[j]
            if len(phase_i) == 0 or len(phase_j) == 0:
                continue
            delta_phi = np.mod(phase_i - phase_j, 2*np.pi)

            tmp_synchrony[i,j] = np.sqrt(np.square(np.mean(np.cos(delta_phi))) + 
                                        np.square(np.mean(np.sin(delta_phi))))
            tmp_synchrony[j,i] = tmp_synchrony[i,j]
        synchrony[ii] = tmp_synchrony

        # #Calculate global synchrony for all connections
        # (D, V) = np.linalg.eig(tmp_synchrony)
        # order = np.argsort(D)

        # eigenvals = D[order].copy()
        # M = N
        # global_synchrony[ii,0] = (np.max(eigenvals) - 1)/(M - 1)

        # #Calculate global synchrony for cortical connections
        # indy1 = np.where((supergroups == 'CTX'))[0]
        # synchrony_sub = tmp_synchrony[indy1][:,indy1]

        # (D, V) = np.linalg.eig(synchrony_sub)
        # order = np.argsort(D)

        # eigenvals = D[order].copy()
        # M = len(indy1)
        # global_synchrony[ii,1] = (np.max(eigenvals) - 1)/(M - 1)

        # #Calculate global synchrony for thalamic connections
        # indy2 = np.where((supergroups == 'TH'))[0]
        # synchrony_sub = tmp_synchrony[indy2][:,indy2]

        # (D, V) = np.linalg.eig(synchrony_sub)
        # order = np.argsort(D)

        # eigenvals = D[order].copy()
        # M = len(indy2)
        # global_synchrony[ii,2] = (np.max(eigenvals) - 1)/(M - 1)

    np.savez(join(SaveDir,f'synchrony_{mID}_{rec_name}.npz'),correlation=correlation,synchrony=synchrony,global_synchrony=global_synchrony)

    correlation[np.isnan(correlation)] = 0
    pdfdoc = PdfPages(join(PlotDir,f'rolling_synchrony_{mID}_{rec_name}.pdf'))
    clims1 = [0,np.round(np.nanpercentile(correlation,99),2)]
    clims2 = [0,np.round(np.nanpercentile(synchrony,99),2)]
    for ii, tW in enumerate(time_window_list):
        
        s_tmp = synchrony[ii].copy()
        s_tmp = s_tmp[order_by_group][:,order_by_group]

        fig, axes = plt.subplots(1,2,figsize=(10,5))
        plt.suptitle(f'{epoch_list[ii]}')
        #center=0,cmap='RdBu_r'
        usrplt.visualize_matrix(np.abs(correlation[ii]),ax=axes[0],clims=clims1,cmap='viridis',plot_ylabel=True,title='Correlation',cbar_label='| Correlation |',cbar=True,ticks=ticks,labels=labels,boundaries=boundaries)
        usrplt.visualize_matrix(s_tmp,ax=axes[1],clims=clims2,cmap='viridis',plot_ylabel=True,title='Phase synchronization',cbar_label='Synchronization',ticks=ticks,labels=labels,boundaries=boundaries)
        pdfdoc.savefig(fig)
        plt.close(fig)
     

    ## Plot participation ratio ----------------------------------------
    fig, axes = plt.subplots(3,1,figsize=(10,8),gridspec_kw = {'height_ratios': [4,2,2],'hspace':0.4})
    plt.suptitle(f'Global synchrony; {mID}, {rec_name}\n {tWindow_width_min} min window, {len(neuron_indices)} neurons')

    corr_with_running = np.zeros((3,2))
    for i in range(3):
        for j in range(2):
            corr_with_running[i,j] = np.corrcoef(running_moments[:,j],global_synchrony[:,i])[0,1]

    ax = axes[0]
    ax.plot(time_window_centers/60,global_synchrony[:,0],lw=2,zorder=2,color=usrplt.cc[1],label=f'All neurons: {corr_with_running[0,0]:.3f}, {corr_with_running[0,1]:.3f}')
    if sort_by_area:
        ax.plot(time_window_centers/60,global_synchrony[:,1],lw=2,zorder=2,color=usrplt.cc[2],label=f'CTX neurons: {corr_with_running[1,0]:.3f}, {corr_with_running[1,1]:.3f}')
        ax.plot(time_window_centers/60,global_synchrony[:,2],lw=2,zorder=2,color=usrplt.cc[3],label=f'TH neurons: {corr_with_running[2,0]:.3f}, {corr_with_running[2,1]:.3f}')
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


    plt.savefig(join(PlotDir,f'global_synchrony_{mID}_{rec_name}.png'),facecolor='white',dpi=300,bbox_inches='tight')
    pdfdoc.savefig(fig)
    plt.close(fig)
    pdfdoc.close()


