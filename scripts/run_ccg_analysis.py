#Base
import argparse
import json, os, time, sys
import numpy as np
import xarray as xr
import pandas as pd
import scipy.stats as st
from tqdm.notebook import trange, tqdm
import ray
from itertools import combinations

#Plot
PlotDir = '/home/david.wyrick/projects/zap-n-zip/plots'
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

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

#CCG
sys.path.append('/home/david.wyrick/Git/modular_network/code')
import ccg
import functional_clustering as fc

ProjDir = '/home/david.wyrick/projects/zap-n-zip/'
PlotDir = os.path.join(ProjDir,'plots','ccg')

##===== ============================ =====##
##===== Parse Command Line Arguments =====##
parser = argparse.ArgumentParser(description='CCG')

##===== Data Options =====##
parser.add_argument('--mID',type=str, default='mouse569064',
                    help='mouse to perform analysis on')

parser.add_argument('--rec_name',type=str, default='estim_vis_2021-04-08_10-28-24',
                    help='experiment ID')
args = parser.parse_args()
if __name__ == '__main__':
        
    # base_dir = '/allen/programs/braintv/workgroups/tiny-blue-dot/zap-n-zip/EEG_exp'
    base_dir = '/data/tiny-blue-dot/zap-n-zip/EEG_exp' #local

    # Parse the arguments
    args = parser.parse_args()
    mID = args.mID 
    print(mID)

    mouseID_folders = os.listdir(base_dir)
    if mID not in mouseID_folders:
        print(f"Error: Source directory '{mID}' does not exist on Allen")
        exit()

    SaveDir = os.path.join(ProjDir,'results','ccg',mID)
    if not os.path.exists(SaveDir):
        os.makedirs(SaveDir)

    #This is the base of the folder tree we're going to copy
    src_dir = os.path.join(base_dir,mID)

    # # Get a list of subfolders in the source directory
    # subfolders = sorted([f for f in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, f))])

    # # Print the list of subfolders and prompt the user to select one
    # print("Select a subfolder to copy:")
    # for i, subfolder in enumerate(subfolders):
    #     print(f"{i+1}. {subfolder}")
    # print('Choose subfolder: ',end=' ')
    # selected_subfolder_index = int(input()) - 1

    # if selected_subfolder_index < 0 or selected_subfolder_index >= len(subfolders):
    #     print("Error: Invalid subfolder selection")
    #     exit()
    rec_name = args.rec_name

    # Create the full path to the selected subfolder in the source directory
    DataDir = os.path.join(src_dir, rec_name)
    file_name = os.path.join(DataDir,'experiment1','recording1')

    ## Upload the whole experiment and generate the global clock
    exp = EEGexp(file_name, preprocess=False, make_stim_csv=False) 
    stim_log = pd.read_csv(exp.stimulus_log_file)

    ## Get probe data
    probe_unit_data, probe_info, total_units = util.get_neuropixel_data(exp)
    probe_list = [x.replace('_sorted', '') for x in exp.experiment_data if 'probe' in x]

    # uniq_areas, uniq_indices, nNeurons_area = np.unique(probe_unit_data['probeB']['areas'],return_index=True, return_counts=True)
    # nNeurons_total = 0
    # for probei in probe_list:
    #     print(probei)
    #     # print(probe_unit_data[probei]['areas'])
    #     uniq_areas, uniq_indices, nNeurons_area = np.unique(probe_unit_data[probei]['areas'],return_index=True, return_counts=True)
    #     nNeurons_total += np.sum(nNeurons_area)
    #     for abbrev, nNeurons in zip(uniq_areas[np.argsort(uniq_indices)],nNeurons_area[np.argsort(uniq_indices)]):
    #         area = ccfsum.loc[ccfsum.abbreviation == abbrev]['name'].values[0]           
    #         print(f'\t{nNeurons:3d} units in {area}, {abbrev}')
    # print(f'{nNeurons_total:3d} units total')

    ## Calculate Firing rate per neuron
    plot_before = 1.; plot_after = 1.; time_bin = 0.001
    bins = np.arange(-plot_before, plot_after+time_bin, time_bin)

    fr_list = []
    for probei in probe_list:
        stimulation_times = stim_log.loc[(stim_log.stim_type == 'biphasic') & (stim_log.sweep == 0)]['onset'].values
        firingrate, bins = get_evoked_spike_counts(probe_unit_data[probei]['spike_times'], probe_unit_data[probei]['spike_clusters'], probe_unit_data[probei]['units'], stimulation_times, plot_before, plot_after, time_bin)
        fr_list.append(firingrate)
    fr_array = np.concatenate(fr_list)
    # areas = np.concatenate([probe_unit_data[probei]['areas'] for probei in probe_list])

    ## Calculate pre-stimulation firing rate for each neuron
    FR_pre_stim_mean = np.mean(np.sum(fr_array[:,:,:1000],axis=2),axis=1)
    highFR_mask = FR_pre_stim_mean > 2
    FR = FR_pre_stim_mean[highFR_mask]
    # areas_sub = areas[highFR_mask]

    # uniq_areas, uniq_indices, num_neurons = np.unique(areas_sub,return_index=True, return_counts=True)
    # area_labels = uniq_areas[np.argsort(uniq_indices)]
    # nNeurons_area = num_neurons[np.argsort(uniq_indices)]

    # boundaries = np.concatenate(([0],np.cumsum(nNeurons_area)))
    # yticks = boundaries[:-1] + np.diff(boundaries)/2
    
    # tmp_df = pd.DataFrame(np.stack((areas,FR_pre_stim_mean)).T,columns=['area','FR'])
    # fig, ax = plt.subplots(figsize=(10,4))
    # ax.set_title('Firing rate pre-stimulation')
    # sns.boxplot(x='area',y='FR',data=tmp_df,ax=ax)
    # ax.set_xticklabels(ax.get_xticklabels(),rotation=45)
    # plt.savefig(os.path.join(PlotDir,f'prestim_FR_{mID}_{rec_name}.pdf'))
    # plt.close(fig)

    #Get various stim parameters
    stim_params = np.unique(stim_log['parameter'])
    stim_params = stim_params[np.where(stim_params != 'white')]
    print(stim_params)
    plot_before = .025; plot_after = 0.725; time_bin = 0.001
    bins = np.arange(-plot_before, plot_after+time_bin, time_bin)

    #Loop over different sweeps. i.e. awake, isofluorene, recovery
    for iSweep in np.unique(stim_log.sweep):
        print(f'Sweep {iSweep}')

        #Get spike counts
        fr_list = []
        for sp in stim_params:
            fr_list2 = []
            for probei in probe_list:
                stimulation_times = stim_log.loc[(stim_log.stim_type == 'biphasic') & (stim_log.sweep == iSweep) & (stim_log.parameter == sp)]['onset'].values
                firingrate, bins = get_evoked_spike_counts(probe_unit_data[probei]['spike_times'], probe_unit_data[probei]['spike_clusters'], probe_unit_data[probei]['units'], stimulation_times, plot_before, plot_after, time_bin)
                fr_list2.append(firingrate)
            fr_list.append(np.concatenate(fr_list2))

        fr_array = np.stack(fr_list)
        fr_array = np.transpose(fr_array,[1,0,2,3])
        ts = bins[:-1] + time_bin/2
        print(fr_array.shape, FR_pre_stim_mean.shape)

        #Look at neurons that fire > 2Hz
        spk_counts = np.ascontiguousarray(fr_array[highFR_mask])
        print(spk_counts.shape, FR.shape)

        #Calculate CCG
        ccgjitter = ccg.get_ccgjitter(spk_counts, FR, jitterwindow=25)
        np.save(os.path.join(SaveDir,f'ccgjitter_sweep-{iSweep}_{mID}_{rec_name}.npy'),ccgjitter)

        #Reformat
        N = np.sum(highFR_mask)
        connectivity = np.zeros((len(stim_params),N,N))
        lag0 = int(ccgjitter.shape[1]/2)
        counter = 0
        for i, j in combinations(np.arange(N),2):
            ccg_tmp = ccgjitter[counter]
            ccg_before = np.mean(ccg_tmp[slice(lag0-13,lag0)],axis=0)
            ccg_after = np.mean(ccg_tmp[slice(lag0,lag0+13)],axis=0)
            connectivity[:,i,j] = ccg_before - ccg_after
            connectivity[:,j,i] = connectivity[:,i,j]
            counter+=1

        np.save(os.path.join(SaveDir,f'ccg_connectivity_sweep-{iSweep}_{mID}_{rec_name}.npy'),connectivity)

        #Plot
        fig, axes = plt.subplots(1,3,figsize=(24,8))
        plt.suptitle(f'Sweep {iSweep}')
        for ii, sp in enumerate(stim_params):
            axes[ii].set_title(f'Stim param {sp}')
            sns.heatmap(connectivity[ii],cmap='bwr',square=True,center=0,vmin=-5e-6,vmax=5e-6,ax=axes[ii],cbar_kws={'shrink':0.5,'label':'\u0394 coinci/spk'})

        # for ax in axes:
        #     ax.set_xticks(yticks); ax.set_xticklabels(area_labels)
        #     ax.set_yticks(yticks); ax.set_yticklabels(area_labels)
        plt.savefig(os.path.join(PlotDir,f'CCG_sweep-{iSweep}_{mID}_{rec_name}.pdf'))
        plt.close(fig)

        #Cluster
        labels_list = []
        for ii, sp in enumerate(stim_params):
            X = connectivity[ii]
            FC = fc.functional_clustering(np.nan_to_num(X))

            # normalize and PCA
            FC.normalize()
            FC.pca()
            # plotted cov of connectivity matrix

            # probiliaty matrix from kmeans
            matrix = FC.probability_matrix(3, data=FC.Z.T)

            # hierarchical clustering
            FC.linkage()

            # plot hierarchical clustering matrix
            FC.plot_matrix()

            # save output cluster ids
            FC.predict_cluster(k=3)
            labels= FC.clusters
            labels_list.append(labels)
        
        fig, axes = plt.subplots(1,3,figsize=(24,8))
        plt.suptitle(f'Sweep {iSweep}')
        for ii, sp in enumerate(stim_params):
            axes[ii].set_title(f'Stim param {sp}')
            X = connectivity[ii]
            labels = labels_list[ii]
            fit_data = X[np.argsort(labels),:][:, np.argsort(labels)]
            sns.heatmap(fit_data,cmap='bwr',square=True,center=0,vmin=-5e-6,vmax=5e-6,ax=axes[ii],cbar_kws={'shrink':0.5,'label':'\u0394 coinci/spk'})
        plt.savefig(os.path.join(PlotDir,f'CCG_clustered_sweep-{iSweep}_{mID}_{rec_name}.pdf'))
        plt.close(fig)

        # jj = 0
        # kwargs = dict(medianprops=dict(linewidth=2, color='k'),boxprops=dict(linewidth=2, color='k'),whiskerprops=dict(linewidth=2, color='k'),capprops=dict(linewidth=2, color='k'))
        # fig, ax = plt.subplots(figsize=(10,4))
        # plt.suptitle(f'Sweep {iSweep}')
        # tmp_list = []; xticks = []; labels = []
        # for a in area_labels:

        #     indy = np.where(areas_sub == a)[0]
        #     if len(indy) < 1:
        #         continue

        #     spos = np.array([0.75,1,1.25]) + 1*jj
        #     jj += 1
        #     xticks.append(spos[1]); labels.append(a)
        #     for ii, sp in enumerate(stim_params):
        #         X = np.mean(connectivity[ii,indy], axis=1)
        #         tmp_list.append(np.stack((np.repeat(a,X.shape[0]),np.repeat(sp,X.shape[0]),X)).T)
        #         bplot = ax.boxplot(X,positions=[spos[ii]],widths=0.2,patch_artist=True,**kwargs)
        #         bplot['boxes'][0].set_facecolor(cc[ii])

        # ax.set_xticks(xticks); ax.set_xticklabels(labels)
        # ax.set_ylim([-5e-7,5e-7])
        # plt.savefig(os.path.join(PlotDir,f'CCG_boxplot_sweep-{iSweep}_{mID}_{rec_name}.pdf'))
        # plt.close(fig)

    print('Done!')

