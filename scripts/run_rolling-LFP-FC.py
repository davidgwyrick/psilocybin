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
import itertools as it
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

#User
sys.path.append(ProjDir)
import util
import plotting as usrplt

# #CCM
# from delay_embedding import ccm
# from delay_embedding import evaluation as E
# from delay_embedding import helpers as H
# from delay_embedding import surrogate as S
# import ray
# import kedm

# #Network 
# import networkx as nx
# import networkx.algorithms.community as nx_comm
# from networkx.algorithms.community import greedy_modularity_communities, modularity
# from networkx.algorithms.efficiency_measures import global_efficiency, local_efficiency

#Templeton-log_exp
gc = gspread.service_account() 
sh = gc.open('Templeton-log_exp') 
exp_table = pd.DataFrame(sh.sheet1.get()).T.set_index(0).T 
exp_table = exp_table.set_index('mouse_name')
exp_table.head()

bandpass_freqs = np.array([[1, 4], [4, 8], [8,13], [13, 30], [30, 55], [65, 100], [100,200]])
bandpass_str = ['delta','theta','alpha','beta','low-gamma','high-gamma','HFO']
n_bands = len(bandpass_freqs)

## Parse Command Line Arguments ----------------------------------------
parser = argparse.ArgumentParser(description='Rolling FCF')

parser.add_argument('--mID',type=str, default='mouse703065',
                    help='mouse to perform analysis on')

parser.add_argument('--rec_name',type=str, default='aw_iso_2023-12-07_10-23-39',
                    help='experiment to perform analysis on')

parser.add_argument('--time_bin_ms',type=int, default=50,
                    help='Time bin width')

parser.add_argument('--tWindow_width_s',type=int, default=60,
                    help='Time window over which to calculate FC')

parser.add_argument('--tWindow_shift_s',type=float, default=30,
                    help='Amount of time to shift rolling window by')

parser.add_argument('--delay',type=int, default=1,
                    help='tau')

parser.add_argument('--zscore',type=int, default=1,
                    help='zscore')

def get_lfp_data(exp, tW):

    probe_list = [x.replace('_sorted', '') for x in exp.experiment_data if 'probe' in x]
    lfp_dict = {probei: {} for probei in probe_list}

    #Loop over probes:
    for probei in probe_list:

        #Get CCF area assignment for each channel
        with open(exp.ephys_params[probei]['probe_info']) as data_file:
            data = json.load(data_file)
        area_ch = np.array(data['area_ch'])
        unique_sorted, uniq_indices = np.unique(area_ch, return_index=True)
        uniq_areas_order = unique_sorted[np.argsort(uniq_indices)]

        #Get LFP sample rate and timestamps
        lfp_sr = exp.ephys_params[probei]['lfp_sample_rate']
        ts_lfp = np.load(exp.ephys_params[probei]['lfp_timestamps'])
        t_indy = np.where((ts_lfp >= tW[0]) & (ts_lfp <= tW[1]))[0]
        ts_epoch = ts_lfp[t_indy]

        # Load the LFP & reshape
        lfp_path = exp.ephys_params[probei]['lfp_continuous']
        lfp = np.memmap(lfp_path, dtype='int16', mode='r')
        nChannels = exp.ephys_params[probei]['num_chs']
        lfp = np.reshape(lfp, (int(lfp.size/nChannels), nChannels))
        lfp_epoch = lfp[t_indy]
        
        #Sample the middle channel of LFP for each area
        lfp_area = []; area_list = []
        for a in uniq_areas_order:
            if a in ['root','null']:
                continue
            indy = np.where(area_ch == a)[0]
            area_list.append(a)

            #Select middle channel in CCF area
            middle_ch = int(indy[len(indy)//2])
            lfp_area.append(lfp_epoch[:,middle_ch])
        
        #Save to dictionary
        lfp_area = np.array(lfp_area).T
        lfp_dict[probei]['lfp_raw'] = lfp_area
        lfp_dict[probei]['ts'] = ts_epoch
        lfp_dict[probei]['areas'] = area_list
        # print(f'LFP shape: {lfp_area.shape}')  

        #Calculate bandpass filtered LFP
        lfp_dict[probei]['lfp_hilbert'], lfp_dict[probei]['lfp_bandpass'], lfp_dict[probei]['lfp_phase'] = util.hilbert_transform(lfp_area, lfp_sr)
        
    return lfp_dict

def interpolate_LFP_probes_to_same_time(lfp_dict,new_lfp_sf=500):
    #First get time stamps for each probe
    ts_lfp_minmax = []
    for probei in probe_list:
        ts_probe = lfp_dict[probei]['ts']
        ts_lfp_minmax.append([np.min(ts_probe),np.max(ts_probe)])

    #Create new LFP time series to interpolate to
    lfp_start = np.max(np.array(ts_lfp_minmax)[:,0])
    lfp_end = np.min(np.array(ts_lfp_minmax)[:,1])
    ts_interp = np.arange(lfp_start,lfp_end,1/new_lfp_sf)
    lfp_dict['ts_interp'] = ts_interp

    #Interpolate to common time series
    lfp_dict_new = {}
    for lfp_type in ['lfp_raw','lfp_bandpass','lfp_hilbert','lfp_phase']:
        tmp_list = []
        for probei in probe_list:
            ts_probe = lfp_dict[probei]['ts']

            #Get LFP and timestamps and interpolate
            lfp_tmp = lfp_dict[probei][lfp_type]
            f_lfp = interp1d(ts_probe,lfp_tmp,axis=0)
            lfp_interp = f_lfp(ts_interp)

            #Save to dictionary
            lfp_dict[probei]['lfp_interp'] = lfp_interp
            tmp_list.append(lfp_interp)
        lfp_concat = np.concatenate(tmp_list,axis=1)
        lfp_dict_new[lfp_type] = lfp_concat
    return lfp_dict_new

def calculate_phase_difference(lfp_phase):
    N = lfp_phase.shape[0]
    #Calculate mean absolute phase difference
    phase_diff = np.zeros((N,N))
    phase_coherence = np.zeros((N,N))
    for i, j in it.combinations(range(N),2):
        phase_diff[i,j] = np.mean(np.abs(lfp_phase[i]-lfp_phase[j]))
        phase_diff[j,i] = phase_diff[i,j]

        phase_coherence[i,j] = np.abs(np.mean(np.exp(1j*(lfp_phase[i]-lfp_phase[j]))))
        phase_coherence[j,i] = phase_coherence[i,j]
    return phase_diff, phase_coherence

# lfp_dict = get_lfp_data_for_powerspectra(exp,tW)
def get_lfp_data_for_powerspectra(exp):

    probe_list = [x.replace('_sorted', '') for x in exp.experiment_data if 'probe' in x]
    lfp_dict = {probei: {} for probei in probe_list}

    #Loop over probes:
    for probei in probe_list:
        print(probei)
        #Get CCF area assignment for each channel
        with open(exp.ephys_params[probei]['probe_info']) as data_file:
            data = json.load(data_file)
        area_ch = np.array(data['area_ch'])
        unique_sorted, uniq_indices = np.unique(area_ch, return_index=True)
        uniq_areas_order = unique_sorted[np.argsort(uniq_indices)]

        #Get LFP sample rate and timestamps
        lfp_sr = exp.ephys_params[probei]['lfp_sample_rate']
        ts_lfp = np.load(exp.ephys_params[probei]['lfp_timestamps'])

        # Load the LFP & reshape
        lfp_path = exp.ephys_params[probei]['lfp_continuous']
        lfp = np.memmap(lfp_path, dtype='int16', mode='r')
        nChannels = exp.ephys_params[probei]['num_chs']
        lfp = np.reshape(lfp, (int(lfp.size/nChannels), nChannels))

        print(f'LFP shape: {lfp.shape}')
        lfp_dict[probei]['lfp_raw'] = lfp
        lfp_dict[probei]['ts'] = ts_lfp
        lfp_dict[probei]['areas'] = area_ch
        print(f'Saved')

    return lfp_dict
  
## Main ----------------------------------------
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
    zscore = bool(args.zscore)
    
    ## FOLDERS ----------------------------------------
    #Create directory for saving to
    TempDir = os.path.join(ServDir,'results','FC_lfp',mID,rec_name)
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
    probe_unit_data, probe_info, total_units, metric_list = tbd_util.get_neuropixel_data(exp)
    probe_list = [x.replace('_sorted', '') for x in exp.experiment_data if 'probe' in x]

    #Get recording start time
    probei = probe_list[0]
    open_ephys_start = np.round(np.min(probe_unit_data[probei]['spike_times']))+10
    open_ephys_end = np.round(np.max(probe_unit_data[probei]['spike_times']))-10
    recording_length = open_ephys_end - open_ephys_start
    mm, ss = divmod(recording_length,60)
    hh, mm = divmod(mm, 60)
    print(f'{hh} hrs, {mm} minutes, {ss} seconds')

    #Read in behavior
    run_ts, run_signal, run_signal_s, acceleration, pupil_ts, pupil_radius, pupil_dxdt, plot_pupil = util.get_behavioral_data(exp, mID, rec_name)
    f_run = interp1d(run_ts,run_signal); run_signal_p = f_run(pupil_ts)
    f_run_s = interp1d(run_ts,run_signal_s); run_signal_p_s = f_run_s(pupil_ts)
    f_pupil = interp1d(pupil_ts,pupil_radius)
    # open_ephys_start = np.nanmin(run_ts)+10
    # open_ephys_end = np.nanmax(run_ts)-10
    # recording_length = open_ephys_end - open_ephys_start
    # mm, ss = divmod(recording_length,60)
    # hh, mm = divmod(mm, 60)
    # print(f'{hh} hrs, {mm} minutes, {ss} seconds')

    injection_times = None; injection_time_windows = None
    #For saline & psilocybin experiments, get injection times and types of injection
    if drug_type in ['saline', 'psilocybin','ketanserin+psilocybin','urethane+psilocybin']:
        injection_time_windows = np.array([np.array(exp_df['First injection window'].values[0].split(','),dtype=float),
                                np.array(exp_df['Second injection window'].values[0].split(','),dtype=float)])
        
        #Take second time in each window as "injection time"
        injection_times = np.array([exp_df['First injection time (s)'].values[0],exp_df['Second injection time (s)'].values[0]],dtype=float)

        if drug_type in ['psilocybin','urethane+psilocybin']:
            injection_types = ['sal1','psi']
            injection_colors = sns.xkcd_palette(['dark sky blue','darkish red'])
            macro_names = ['pre-inj','post-sal-inj','post-psi-inj']
            cmap = sns.xkcd_palette(['silver','dark sky blue','cobalt blue'])
        elif drug_type == 'saline':
            injection_types = ['sal1', 'sal2']
            injection_colors = sns.xkcd_palette(['dark sky blue','cobalt blue'])
            macro_names = ['pre-inj','post-sal1-inj','post-sal2-inj']
        elif drug_type == 'ketanserin+psilocybin':
            injection_types = ['ket','psi']
            injection_colors = sns.xkcd_palette(['magenta','goldenrod'])
            macro_names = ['pre-inj','post-ket-inj','post-psi-inj']
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
    tW_starts = np.arange(open_ephys_start,open_ephys_end-tWindow_width,tWindow_shift)
    tW_ends = tW_starts + tWindow_width
    time_window_array = np.array((tW_starts,tW_ends)).T
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

    #Save time windows 
    np.savez(os.path.join(SaveDir,f'time_windows.npz'),time_window_list=time_window_list,epoch_list=epoch_list,running_moments=running_moments,pupil_moments=pupil_moments)


    pdfdoc = PdfPages(join(PlotDir,f'STFT_lfp_{mID}_{rec_name}.pdf'))
    lfp_dict = get_lfp_data_for_powerspectra(exp)

    lfp_sr = 2500
    tW = [open_ephys_start+10,open_ephys_end-10]

    for probe_i in probe_list:
        areas = lfp_dict[probe_i]['areas']
        ts = lfp_dict[probe_i]['ts']
        t_indy = np.where((ts >= tW[0]) & (ts <= tW[1]))[0]
        
        unique_sorted, uniq_indices = np.unique(areas, return_index=True)
        uniq_areas_order = unique_sorted[np.argsort(uniq_indices)]

        for a in uniq_areas_order:
            indy = np.where(areas == a)[0]
            if a in ['root','null']:
                continue

            #Select middle channel in CCF area
            middle_ch = int(indy[len(indy)//2])
            lfp = lfp_dict[probe_i]['lfp_raw'][t_indy,middle_ch]

            f, t, zxx = np.abs(sig.stft(lfp, lfp_sr,nperseg=2048))
            print(f.shape,t.shape,zxx.shape)
            x = np.log10(zxx)
            x = np.array([(x_i - np.nanmean(x_i)) / np.std(x_i) for x_i in x])
            x = gaussian_filter(x, sigma=5)
            idx = np.argmin(np.abs(f - 100))
            x_plot = x[:idx,:]

            fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(18, 12), gridspec_kw={'height_ratios': [0.75, 0.125,0.125]})
            plt.suptitle(f'LFP  power modulations in {a}, {probe_i}',y=0.96)
            
            #LFP power spectra 
            ax = axes[0]
            ax.set_title('Deviations from  mean power (dB/Hz)')
            sns.heatmap(np.flipud(x_plot), cmap='RdBu_r', ax=ax, center=0, vmax=0.7,vmin=-0.7, xticklabels=100, cbar=False)

            yticks = np.arange(0,idx,idx//10)
            ax.set_yticks(idx - np.arange(0, 85, 16)-1); ax.set_yticklabels(np.round(f[np.arange(0, 85, 16)] / 10) * 10, rotation=0)
            ax.set_xticks([])
            ax.set_ylabel('Frequency (Hz)')

            #Running speed
            ax = axes[1]
            indy = np.where((run_ts >= tW[0]) & (run_ts <= tW[1]))[0]
            ax.plot(run_ts[indy],run_signal[indy],'-k')
            ax.set_xticklabels([]);ax.set_ylabel('Speed')
            ax.autoscale(tight=True)
            usrplt.adjust_spines(ax) 
            ylim = ax.get_ylim()
            if injection_times is not None:
                for i in range(2):
                    ax.vlines(injection_times[i],*ylim,linestyles='dashed',color=injection_colors[i],label=injection_types[i],lw=3,zorder=4)
            ax.legend()

            #Pupil radius
            ax = axes[2]
            indy = np.where((pupil_ts >= tW[0]) & (pupil_ts <= tW[1]))[0]
            ax.plot(pupil_ts[indy],pupil_radius[indy],'-b')
            ax.autoscale(tight=True)
            usrplt.adjust_spines(ax)  
            ax.set_xlabel('Time (s)');ax.set_ylabel('Speed')
            ylim = ax.get_ylim()
            if injection_times is not None:
                for i in range(2):
                    ax.vlines(injection_times[i],*ylim,linestyles='dashed',color=injection_colors[i],label=injection_types[i],lw=3,zorder=4)
            ax.legend()

            a_str = a.replace('/','-')
            plt.savefig(join(PlotDir,f'STFT-LFP_{probe_i}_{a_str}.png'),facecolor='white',dpi=300,bbox_inches='tight')
            pdfdoc.savefig(fig)
            plt.close(fig)
    pdfdoc.close()
    exit()

    # #Plot STFT for each area
    # for probe_i in probe_list:
    #     areas = lfp_dict[probe_i]['areas']
    #     ts = lfp_dict[probe_i]['ts']
        
    #     unique_sorted, uniq_indices = np.unique(areas, return_index=True)
    #     uniq_areas_order = unique_sorted[np.argsort(uniq_indices)]

    #     for a in uniq_areas_order:
    #         indy = np.where(areas == a)[0]
    #         if a in ['root','null']:
    #             continue
    #         a_str = a.replace('/','-')
    #         pdfdoc = PdfPages(join(PlotDir,f'STFT_lfp_{probe_i}_{a_str}_{mID}_{rec_name}.pdf'))
    #         #Select middle channel in CCF area
    #         middle_ch = int(indy[len(indy)//2])
    #         lfp = lfp_dict[probe_i]['lfp_raw'][:,middle_ch]
    #         tW_starts = np.arange(open_ephys_start+10,open_ephys_end-190,180)
    #         tE_ends = tW_starts + 180

    #         for iW, (tS,tE) in enumerate(zip(tW_starts,tE_ends)):
    #             tM = tS + 90
    #             if tM < injection_time_windows[0]:
    #                 epoch = macro_names[0]
    #             elif (tM > injection_time_windows[0]) & (tM < injection_time_windows[1]):
    #                 epoch = macro_names[1]
    #             else:
    #                 epoch = macro_names[1]
    #             t_indy = np.where((ts >= tS) & (ts <= tE))[0]
    #             lfp_e = lfp[t_indy]
            
    #             f, t, zxx = np.abs(sig.stft(lfp_e, lfp_sr,nperseg=2048))

    #             x = np.log10(zxx)
    #             x = np.array([(x_i - np.nanmean(x_i)) / np.std(x_i) for x_i in x])
    #             x = gaussian_filter(x, sigma=5)
    #             idx = np.argmin(np.abs(f - 100))
    #             x_plot = x[:idx,:]

    #             fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(18, 12), gridspec_kw={'height_ratios': [0.75, 0.125,0.125]})
    #             plt.suptitle(f'LFP  power modulations in {a}, {probe_i}, [{tS:.2f}, {tE:.2f}]s, {epoch} ',y=0.96)
                
    #             #LFP power spectra 
    #             ax = axes[0]
    #             ax.set_title('Deviations from  mean power (dB/Hz)')
    #             sns.heatmap(np.flipud(x_plot), cmap='RdBu_r', ax=ax, center=0, vmax=0.7,vmin=-0.7, xticklabels=100, cbar=False)

    #             yticks = np.arange(0,idx,idx//10)
    #             ax.set_yticks(idx - np.arange(0, 85, 16)-1); ax.set_yticklabels(np.round(f[np.arange(0, 85, 16)] / 10) * 10, rotation=0)
    #             ax.set_xticks([])
    #             ax.set_ylabel('Frequency (Hz)')

    #             #Running speed
    #             ax = axes[1]
    #             indy = np.where((run_ts >= tW[0]) & (run_ts <= tW[1]))[0]
    #             ax.plot(run_ts[indy],run_signal[indy],'-k')
    #             ax.set_xticklabels([]);ax.set_ylabel('Speed')
    #             ax.autoscale(tight=True)
    #             usrplt.adjust_spines(ax) 

    #             #Pupil radius
    #             ax = axes[2]
    #             indy = np.where((pupil_ts >= tW[0]) & (pupil_ts <= tW[1]))[0]
    #             ax.plot(pupil_ts[indy],pupil_radius[indy],'-b')
    #             ax.autoscale(tight=True)
    #             usrplt.adjust_spines(ax)  
    #             ax.set_xlabel('Time (s)');ax.set_ylabel('Speed')
 
    #             # 
    #             # a_str = a.replace('/','-')
    #             # plt.savefig(join(PlotDir,f'STFT-LFP_{probe_i}_{a_str}.png'),facecolor='white',dpi=300,bbox_inches='tight')
    #             pdfdoc.savefig(fig)
    #             plt.close(fig)

    #         pdfdoc.close()

    ## Calculate rolling FCF ----------------------------------------
    t0_outer = time.perf_counter()
    takens_range = np.arange(1,51) 

    mod_corr = np.zeros((nWindows,2))
    mod_FCF = np.zeros((nWindows,2))

    FCF_all = []# np.zeros((nWindows,N,N))
    pdfdoc = PdfPages(join(PlotDir,f'rolling_FCF_lfp_{mID}_{rec_name}.pdf'))
    
    #Loop over different time blocks
    for ii, tW in enumerate(time_window_list):
        t0 = time.perf_counter()
        print(epoch_list[ii])

        #Get LFP data
        lfp_dict = get_lfp_data(exp, tW)
        lfp_dict2 = interpolate_LFP_probes_to_same_time(lfp_dict,new_lfp_sf=500)

        area_all = np.concatenate([lfp_dict[probei]['areas'] for probei in probe_list])
        boundaries = np.cumsum([len(lfp_dict[probei]['areas']) for probei in probe_list])
        ticks = np.arange(len(area_all))
        labels = area_all; N = len(labels)

        #Calculate correlation
        FC_hilbert = np.zeros((n_bands,N,N))
        phase_diff = np.zeros((n_bands,N,N))
        phase_coherence = np.zeros((n_bands,N,N))

        # Calculate cross correlation on bandpassed filtered LFP and envelope LFP for different bands
        fig, axes = plt.subplots(2,4,figsize=(24,12),gridspec_kw={'hspace': 0.25,'wspace':0.25})
        plt.suptitle(f'Cross-correlation across different LFP bands for {epoch_list[ii]} ',y=0.925)
        for i, freq_range in enumerate(bandpass_freqs):
            lfp_hilbert = lfp_dict2['lfp_hilbert'][:,:,i].T
            FC_hilbert[i] = np.corrcoef(lfp_hilbert)

            ax = axes[i//4,i%4]
            usrplt.visualize_matrix(FC_hilbert[i],clims=[0,1],plot_ylabel=True,ticks=ticks,labels=labels,boundaries=boundaries, cbar=False,title=f'{bandpass_str[i]} ({freq_range} Hz) band',cbar_label='Correlation',ax=ax)
        axes[-1,-1].axis('off')

        pdfdoc.savefig(fig)
        plt.savefig(join(PlotDir,f'FC-LFP_{epoch_list[ii]}.png'),facecolor='white',dpi=300,bbox_inches='tight')
        plt.close(fig)

        # Calculate phase difference on bandpassed filtered LFP and envelope LFP for different bands
        fig, axes = plt.subplots(2,4,figsize=(24,12),gridspec_kw={'hspace': 0.25,'wspace':0.25})
        plt.suptitle(f'Phase difference across different LFP bands for {epoch_list[ii]} ',y=0.925)
        for i, freq_range in enumerate(bandpass_freqs):
            lfp_phase = lfp_dict2['lfp_phase'][:,:,i].T
            phase_diff[i], phase_coherence[i] = calculate_phase_difference(lfp_phase)

            ax = axes[i//4,i%4]
            usrplt.visualize_matrix(phase_diff[i],cmap='rocket',clims=[0,np.pi],plot_ylabel=True,ticks=ticks,labels=labels,boundaries=boundaries, cbar=True,title=f'{bandpass_str[i]} ({freq_range} Hz) band',cbar_label='| \u0394-Phase |',ax=ax)
        axes[-1,-1].axis('off')
        # pdfdoc.savefig(fig)
        plt.savefig(join(PlotDir,f'Phase-LFP_{epoch_list[ii]}.png'),facecolor='white',dpi=300,bbox_inches='tight')
        plt.close(fig)

        #Save to file
        np.savez(join(SaveDir,f'rolling-FC_{epoch_list[ii]}.npz'),FC_hilbert=FC_hilbert,phase_diff=phase_diff,phase_coherence=phase_coherence,area_all=area_all,mID=mID,rec_name=rec_name,tWindow=tW,epoch=epoch_list[ii],boundaries=boundaries,ticks=ticks,labels=labels)
        tE = (time.perf_counter() - t0)/60
        print('\tCompleted in {:.2f} mins'.format(tE))

    comp_length = time.perf_counter() - t0_outer
    mm, ss = divmod(comp_length,60)
    hh, mm = divmod(mm, 60)
    print(f'Completed in {hh:.0f} hrs, {mm:.0f} minutes, {ss:.0f} seconds')
    ## CLOSE files ----------------------------------------
    pdfdoc.close()

    




