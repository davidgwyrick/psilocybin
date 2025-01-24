base_dir_server = '/allen/programs/mindscope/workgroups/templeton-psychedelics/'
base_dir_local = '/data/tiny-blue-dot/zap-n-zip/EEG_exp'; base_dir = base_dir_server
ProjDir = '/home/david.wyrick/projects/zap-n-zip/'
ServDir = '/data/projects/zap-n-zip/'

#Basea
import gspread
from os.path import join
from glob import glob
import json, os, time, sys, argparse
import numpy as np
import pandas as pd
import scipy.stats as st
import scipy.signal as sig
import pingouin as pg
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d
from itertools import combinations

#Plot
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pptx import Presentation
from pptx.util import Inches
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D   

#Project
from tbd_eeg.data_analysis.eegutils import EEGexp
import tbd_eeg.data_analysis.Utilities.utilities as tbd_util
from tbd_eeg.data_analysis.Utilities.behavior_movies import Movie

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

#Define behavioral states
# behavior_ranges = {0: [0,1], 1: [1,10], 2: [10,30], 3:[30,500]}
# behavior_dict = {0: 'rest (0-1cm/s)', 1: 'walk (1-10cm/s)', 2: 'shuffle (10-30cm/s)', 3: 'run (>30cm/s)'}
# behavior_strs2 = ['rest','walk','shuffle','run']

behavior_ranges = {0: [0,1], 1: [1,15], 2: [15,500]}#, 3:[30,500]}
behavior_dict = {0: 'rest (0-1cm/s)', 1: 'walk (1-15cm/s)', 2: 'run (>15cm/s)'}
behavior_strs2 = ['rest','walk','run']
behavior_strs = list(behavior_dict.values())
nBehaviors = len(behavior_strs)

#Define windows to calculate firing rate
windows_of_interest = [[.002,.025],[.075,.3],[.3,1],[1,4.5]]
window_strs = ['evoked (2-25ms)','rebound (75-300ms)','post-rebound (0.3-1s)','ISI (1-4.5s)']
nWindows = len(window_strs)

gc = gspread.service_account() # need a key file to access the account (step 2) 
sh = gc.open('Templeton-log_exp') # open the spreadsheet 

df = pd.DataFrame(sh.sheet1.get()) 
exp_table = df.T.set_index(0).T # put it in a nicely formatted dataframeexp_table.iloc[10:]
exp_table = exp_table.set_index('mouse_name')

##===== ============================ =====##
##===== Parse Command Line Arguments =====##
parser = argparse.ArgumentParser(description='single-cell-metrics')

##===== Data Options =====##
parser.add_argument('--mID',type=str, default='mouse709400',
                    help='mouse to perform analysis on')

parser.add_argument('--rec_name',type=str, default='aw_ket_2024-02-01_11-12-34',
                    help='experiment to perform analysis on')

parser.add_argument('--window_t_min',type=int, default=15,
                    help='Window width (mins) to segment data into')

if __name__ == '__main__':

    # Parse the arguments
    args = parser.parse_args()
    mID = args.mID
    rec_name = args.rec_name

    #Segement data into 15 minute windows
    window_t_min = args.window_t_min
    window_t = window_t_min*60
    
    #Extract row from templeton dataframe
    exp_df = exp_table.loc[exp_table.exp_name == rec_name]
    if type(exp_df) == pd.core.series.Series:
        exp_df = exp_df.to_frame().T

    drug_type = exp_df['drug'].values[0]
    stim_type = exp_df['stimulation'].values[0]
    print(f'Experiment type: {stim_type}, {drug_type}')

    #Define directories
    SaveDir = join(ServDir,'results','lfp_spectra',mID,rec_name)
    if not os.path.exists(SaveDir):
        os.makedirs(SaveDir)

    PlotDir = join(SaveDir,'plots')
    if not os.path.exists(PlotDir):
        os.makedirs(PlotDir)

    #%% Upload the whole experiment and generate the global clock
    file_name = os.path.join(base_dir_server,mID,rec_name,'experiment1','recording1')
    exp = EEGexp(file_name, preprocess=False, make_stim_csv=False)

    #Load probe data
    probe_unit_data, probe_info, total_units = tbd_util.get_neuropixel_data(exp)
    probe_list = [x.replace('_sorted', '') for x in exp.experiment_data if 'probe' in x]

    #Get recording start time
    probei = probe_list[0]
    open_ephys_start = np.round(np.min(probe_unit_data[probei]['spike_times']))
    open_ephys_end = np.round(np.max(probe_unit_data[probei]['spike_times']))

    #For saline & psilocybin experiments, get injection times and types of injection
    if drug_type in ['saline', 'psilocybin']:
        injection_times = [float(exp_df['First injection time (s)'].values[0]),
                        float(exp_df['Second injection time (s)'].values[0])]
        injection_time_windows = None
        #Determine injection type
        if 'psi' in rec_name:
            injection_types = ['sal1','psi']
        else:
            injection_types = ['sal1', 'sal2']

    elif drug_type == 'ketanserin+psilocybin':
        injection_time_windows = np.array([np.array(exp_df['First injection time (s)'].values[0].split(','),dtype=float),
                                np.array(exp_df['Second injection time (s)'].values[0].split(','),dtype=float)])
        
        #Take second time in each window as "injection time"
        injection_times = [injection_time_windows[0,1],injection_time_windows[1,1]]

        injection_types = ['ket','psi']
    else:
        injection_times = None
        injection_time_windows = None

    #For isoflurane experiments, get iso level
    if drug_type == 'isoflurane':
        iso_level, iso_times = exp.load_analog_iso()
        iso_induction_times, iso_maintenance_times = exp.load_iso_times()
    else: 
        iso_induction_times = None
        iso_maintenance_times = None
    # extract the timestamps of the selected stimuli
    try:
        stim_log = pd.read_csv(exp.stimulus_log_file)

        if 'circle' in np.unique(stim_log['stim_type']):
            vis_stim_exists = True
        else:
            vis_stim_exists = False
        if 'biphasic' in np.unique(stim_log['stim_type']):
            stim_exists = True
        else:
            stim_exists = False
    except:
        stim_exists = False
        vis_stim_exists = False
        stim_log = None

    #Load behavior
    run_ts, raw_run_signal, run_signal, run_signal_s, pupil_ts, pupil_radius, run_signal_p, run_signal_p_s, plot_pupil = util.get_behavioral_data(exp, mID, rec_name)
    f_run = interp1d(run_ts,run_signal)
    f_run_s = interp1d(run_ts,run_signal_s)

    #Get time windows for each epoch
    epoch_list, time_window_list = util.define_epochs_of_interest([open_ephys_start,open_ephys_end], drug_type, window_t_min=window_t_min, injection_times=injection_times,injection_time_windows=injection_time_windows, iso_induction_times=iso_induction_times, stim_log=stim_log)
    spont_mask = np.array(['spont' in epoch for epoch in epoch_list])
    biphasic_mask = np.array(['biphasic' in epoch for epoch in epoch_list])

    nCond = len(epoch_list)

    if drug_type == 'saline':
        macro_epochs = ['pre-inj',f'post-{injection_types[0]}-inj',f'post-{injection_types[1]}-inj']
        cmap_macro = sns.xkcd_palette(['silver','dark sky blue','cobalt blue'])
        cmap = sns.color_palette('Greens',nCond)

    elif drug_type == 'psilocybin':
        macro_epochs = ['pre-inj',f'post-{injection_types[0]}-inj',f'post-{injection_types[1]}-inj']
        cmap_macro = sns.xkcd_palette(['silver','dark sky blue','darkish red'])
        nPsi = np.sum(['psi' in fname for fname in epoch_list])
        nSal = nCond - nPsi
        if nSal == 0:
            cmap = sns.color_palette('Reds',nCond)
        elif nPsi == 0:
            cmap = sns.color_palette('Blues',nCond)
        else:
            cmap = np.concatenate((sns.color_palette('Blues',nSal),sns.color_palette('Reds',nPsi)))

    elif drug_type == 'ketanserin+psilocybin':
        macro_epochs = ['pre-inj',f'post-{injection_types[0]}-inj',f'post-{injection_types[1]}-inj']
        cmap_macro = sns.xkcd_palette(['silver','goldenrod','darkish red'])
        nPre = np.sum(['pre' in fname for fname in epoch_list])
        nKet = np.sum(['ket' in fname for fname in epoch_list])
        nPsi = np.sum(['psi' in fname for fname in epoch_list])
        cmap = np.concatenate((sns.color_palette('Blues',nPre),sns.color_palette('Oranges',nKet),sns.color_palette('Reds',nPsi)))

    elif drug_type == 'isoflurane':
        macro_epochs = ['pre-iso','iso-ind','post-iso']
        cmap_macro = sns.xkcd_palette(['silver','light teal','teal'])
        nPre = np.sum(['pre' in fname for fname in epoch_list])
        nPost = nCond - nPre
        if nPre == 0:
            cmap = sns.color_palette('Oranges',nCond)
        elif nPost == 0:
            cmap = sns.color_palette('Blues',nCond)
        else:
            cmap = np.concatenate((sns.color_palette('Blues',nPre),sns.color_palette('Oranges',nPost)))

    elif drug_type == 'urethane':
        macro_epochs = ['urethane']
        cmap_macro = sns.xkcd_palette(['orange'])
        cmap = sns.color_palette('Oranges',nCond)


    #%% Calculate LFP spectra
    from neurodsp import spectral
    #Loop over probes:
    for probei in probe_list:

        pdfdoc = PdfPages(join(PlotDir,f'LFP_spectra_averages_{probei}_{mID}_{rec_name}.pdf'))
        print(f'Calculating LFP spectra for {probei}')

        ## Load probe_info.json ##
        with open(exp.ephys_params[probei]['probe_info']) as data_file:
            data = json.load(data_file)
        npx_allch = np.array(data['channel'])       # this is an array from 0 to 384
        surface_ch = int(data['surface_channel'])   # the electrode we said was at the brain surface
        air_ch = int(data['air_channel'])           # the electrode we said was at the saline surface
        allch_z = np.array(data['vertical_pos'])    # vertical posizion of each elec (um), relative to the tip (ch 0 is 20 um from tip)
        ref_mask = np.array(data['mask'])           # contains a False for Npx reference channels and "bad chs"
        ref_mask[surface_ch:] = False
        try:
            area_ch = np.array(data['area_ch'])
            groups_ch, group_dict, graph_order, group_order, group_order_labels, supergroups_ch = util.determine_groups(area_ch)
            area_info = True
        except:
            area_ch = npx_allch
            area_info = False

        #%% Upload the LFP 
        lfp_path = exp.ephys_params[probei]['lfp_continuous']
        lfp = np.memmap(lfp_path, dtype='int16', mode='r')
        timestamps_lfp = np.load(exp.ephys_params[probei]['lfp_timestamps'])

        #Reshape 
        nChannels = exp.ephys_params[probei]['num_chs']
        lfp = np.reshape(lfp, (int(lfp.size/nChannels), nChannels))
        lfp_sr = exp.ephys_params[probei]['lfp_sample_rate']

        fig, ax = plt.subplots(figsize=(9,9))
        ax.set_title(f'Spectral analysis of raw LFP, average spectrum over good channels for {probei}')
        spectrum_per_ch = np.full((nCond,nChannels,int(lfp_sr/2+1)),np.nan)
        for ii, tWindow in enumerate(time_window_list):
            print(f'\t {epoch_list[ii]}')
            indy = np.where((timestamps_lfp > tWindow[0]) & (timestamps_lfp < tWindow[1]))[0]
            lfp_epoch = lfp[indy]
            # lfp_epoch = (lfp_epoch - np.nanmean(lfp_epoch,0))* exp.ephys_params[probei]['bit_volts']
            for iC in range(nChannels):
                if not ref_mask[iC]:
                    continue
                fxx, pxx = spectral.compute_spectrum(lfp_epoch[:,iC], lfp_sr, method='welch', window='hann', nperseg=lfp_sr)
                spectrum_per_ch[ii,iC] = pxx
            pxx_avg = np.nanmean(spectrum_per_ch[ii],axis=0)
            plt.loglog(fxx, pxx_avg,color=cmap[ii],label=epoch_list[ii],lw=1)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power (V^2/Hz)')
        ax.legend()
        usrplt.adjust_spines(ax)
        plt.savefig(join(PlotDir,f'LFP_spectra_avg-over-chs_{probei}_{mID}_{rec_name}.png'))
        pdfdoc.savefig(fig) 
        plt.close(fig)

        #Plot average per area
        if area_info:
            unique_sorted, uniq_indices = np.unique(area_ch, return_index=True)
            uniq_areas = unique_sorted[np.argsort(uniq_indices)]

            for a in uniq_areas:

                #Average over channels in area
                area_indy = np.where(area_ch == a)[0]
                fig, ax = plt.subplots(figsize=(9,9))
                ax.set_title(f'Spectral analysis of raw LFP, average spectrum for area {a}')
                for ii, tWindow in enumerate(time_window_list):
                    pxx_avg = np.nanmean(spectrum_per_ch[ii,area_indy],axis=0)
                    plt.loglog(fxx, pxx_avg,color=cmap[ii],label=epoch_list[ii],lw=1)
                plt.xlabel('Frequency (Hz)')
                plt.ylabel('Power (V^2/Hz)')
                ax.legend()
                usrplt.adjust_spines(ax)
                pdfdoc.savefig(fig) 
                plt.close(fig)

                #Average over channels in area and condition on pre/post injection
                if np.sum(biphasic_mask) > 0:
                    fig, axes = plt.subplots(1,2,figsize=(10,5))
                    bi = True
                else:
                    fig, ax = plt.subplots(figsize=(9,9))
                    bi = False

                plt.suptitle(f'Spectral analysis of raw LFP, average spectrum for area {a}, average over epochs')
                for e, c in zip(macro_epochs,cmap_macro):
                    epoch_mask = np.array([e in epoch for epoch in epoch_list])
                    # import pdb; pdb.set_trace()
                    if bi:
                        epoch_i = np.where(biphasic_mask & epoch_mask)[0]
                        ax = axes[0]
                        ax.set_title('Biphasic')
                        spectrum_sub = spectrum_per_ch[epoch_i][:,area_indy]
                        pxx_avg = np.nanmean(np.nanmean(spectrum_sub,axis=0),axis=0)
                        # import pdb; pdb.set_trace()
                        ax.loglog(fxx, pxx_avg,color=c,label=e,lw=1)
                        ax.set_xlabel('Frequency (Hz)')
                        ax.set_ylabel('Power (V^2/Hz)')
                        ax.legend()
                        usrplt.adjust_spines(ax)
                        ax = axes[1]
                    ax.set_title('Spontaneous')
                    epoch_i = np.where(spont_mask & epoch_mask)[0]
                    spectrum_sub = spectrum_per_ch[epoch_i][:,area_indy]
                    pxx_avg = np.nanmean(np.nanmean(spectrum_sub,axis=0),axis=0)
                    ax.loglog(fxx, pxx_avg,color=c,label=e,lw=1)
                    ax.set_xlabel('Frequency (Hz)')
                    ax.set_ylabel('Power (V^2/Hz)')
                    ax.legend()
                    usrplt.adjust_spines(ax)
                pdfdoc.savefig(fig) 
                plt.close(fig)
        pdfdoc.close()
        
        pdfdoc = PdfPages(join(PlotDir,f'LFP_spectra_per-channel_{probei}_{mID}_{rec_name}.pdf'))
        for iC in range(nChannels):
            if not ref_mask[iC]:
                continue
            fig, ax = plt.subplots(figsize=(9,9))
            ax.set_title(f'Spectral analysis of raw LFP, {probei} channel {iC}, area: {area_ch[iC]}')

            for ii, tWindow in enumerate(time_window_list):
                plt.loglog(fxx, spectrum_per_ch[ii,iC],color=cmap[ii],label=epoch_list[ii],lw=1)
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Power (V^2/Hz)')
            ax.legend()
            usrplt.adjust_spines(ax)
            pdfdoc.savefig(fig) 
            plt.close(fig)
        
        np.savez(join(SaveDir,f'LFP_spectra_per-channel_{probei}_{mID}_{rec_name}.npy'),spectrum_per_ch=spectrum_per_ch,fxx=fxx,npx_allch=npx_allch,area_ch=area_ch,ref_mask=ref_mask)
        pdfdoc.close()

# %%
