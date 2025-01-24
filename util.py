base_dir_server = '/allen/programs/mindscope/workgroups/templeton-psychedelics/'
base_dir_local = '/data/tiny-blue-dot/zap-n-zip/EEG_exp'; base_dir = base_dir_server
ProjDir = '/home/david.wyrick/projects/zap-n-zip/'
ServDir = '/data/projects/zap-n-zip/'

#Base
from os.path import join
import json, os, time, sys
import gspread
import pandas as pd
import numpy as np
import networkx as nx

#Scipy
import scipy.signal as sig
import scipy.stats as st
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d
from derivative import dxdt

#Proj
from tbd_eeg.data_analysis.eegutils import EEGexp
import tbd_eeg.data_analysis.Utilities.utilities as tbd_util
from tbd_eeg.data_analysis.Utilities.behavior_movies import Movie

#Allen
from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
mcc = MouseConnectivityCache(resolution=10)
str_tree = mcc.get_structure_tree()

#Misc
from kneed import KneeLocator

#For progress bar
from asyncio import Event
from typing import Tuple
from time import sleep
import ray
from ray.actor import ActorHandle
from tqdm import tqdm

#Define behavioral states
behavior_ranges = {0: [0,1], 1: [1,500]}#, 3:[30,500]}
behavior_dict = { 0: 'rest (<1cm/s)', 1: 'active (>1cm/s)'}
behavior_strs2 = ['rest','active']
behavior_strs = list(behavior_dict.values())
nBehaviors = len(behavior_strs)

#Define windows to calculate firing rate
stim_window_dict = {'spontaneous': [], 'evoked': [],'pre-rebound': [], 'rebound': [], 'post-rebound': [], 'visual': []}
stim_strs = ['spontaneous','evoked','pre-rebound','rebound','post-rebound','visual']
nStimTypes = len(stim_strs)
evoked_windows = [[.002,.025],[0.025,0.075],[.075,.3],[.3,1]]
evoked_strings = ['evoked','pre-rebound','rebound','post-rebound']


from scipy.optimize import curve_fit
#Equations for gain fits
def sigmoid(x,beta,h0,mR):
    y = mR / (1 + np.exp(-beta*(x-h0)))
    return (y)   

def line(x,m,x0):
    y = m*(x+x0)
    return (y)


def relu(x,m,x0):
    #Initialize
    y = np.zeros(x.shape)
    #Fit shifted line
    pos = x < x0
    y[pos] = 0
    y[~pos] = m*(x[~pos]-x0)
    #Ensure positivity; below x0, y = 0
    pos = y < 0
    y[pos] = 0
    return (y)

##------------------------------------------
@ray.remote
class ProgressBarActor:
    counter: int
    delta: int
    event: Event

    def __init__(self) -> None:
        self.counter = 0
        self.delta = 0
        self.event = Event()

    def update(self, num_items_completed: int) -> None:
        """Updates the ProgressBar with the incremental
        number of items that were just completed.
        """
        self.counter += num_items_completed
        self.delta += num_items_completed
        self.event.set()

    async def wait_for_update(self) -> Tuple[int, int]:
        """Blocking call.

        Waits until somebody calls `update`, then returns a tuple of
        the number of updates since the last call to
        `wait_for_update`, and the total number of completed items.
        """
        await self.event.wait()
        self.event.clear()
        saved_delta = self.delta
        self.delta = 0
        return saved_delta, self.counter

    def get_counter(self) -> int:
        """
        Returns the total number of complete items.
        """
        return self.counter

##------------------------------------------   
# Back on the local node, once you launch your remote Ray tasks, call
# `print_until_done`, which will feed everything back into a `tqdm` counter.
class ProgressBar:
    progress_actor: ActorHandle
    total: int
    description: str
    pbar: tqdm

    def __init__(self, total: int, description: str = ""):
        # Ray actors don't seem to play nice with mypy, generating
        # a spurious warning for the following line,
        # which we need to suppress. The code is fine.
        self.progress_actor = ProgressBarActor.remote()  # type: ignore
        self.total = total
        self.description = description

    @property
    def actor(self) -> ActorHandle:
        """Returns a reference to the remote `ProgressBarActor`.

        When you complete tasks, call `update` on the actor.
        """
        return self.progress_actor

    def print_until_done(self) -> None:
        """Blocking call.

        Do this after starting a series of remote Ray tasks, to which you've
        passed the actor handle. Each of them calls `update` on the actor.
        When the progress meter reaches 100%, this method returns.
        """
        pbar = tqdm(desc=self.description, total=self.total)
        while True:
            delta, counter = ray.get(self.actor.wait_for_update.remote())
            pbar.update(delta)
            if counter >= self.total:
                pbar.close()
                return
            
##------------------------------------------
def get_spike_counts_VCNP(spk_dict, unit_ids, tWindow, tBin):

    bins = np.arange(tWindow[0], tWindow[1]+1E-6, tBin)
    spk_counts = np.empty((len(unit_ids), len(bins)-1))*np.nan

    for indi, uniti in enumerate(unit_ids):
        spikesi = spk_dict[uniti]
        spk_counts_i, edges = np.histogram(spikesi, bins)
        spk_counts[indi,:] = spk_counts_i

    return spk_counts, bins

##------------------------------------------
def usr_zscore(X,axis=0):
    mx = np.mean(X,axis=axis); std = np.std(X,axis=axis)
    Xz = np.divide(X-mx,std,out=np.zeros(X.shape),where = std!= 0)
    return Xz

##------------------------------------------
def define_epochs_of_interest(tWindow, drug_type, window_t_min=10, injection_times=None, injection_time_windows=None, iso_induction_times=None, stim_log=None):

    #Segement data into x minute windows
    window_t = window_t_min*60

    ## Create landmark list to create windows relative to 
    landmarks = [tWindow[0],tWindow[-1]]

    #Add periods of evoked activity if it exists
    evoked_time_window_list = []
    evoked_type_list = []
    sweep_list = []
    if stim_log is not None:
        stim_exists = True
        for s in np.unique(stim_log['sweep']):
            for t in np.unique(stim_log.loc[stim_log.sweep == s]['stim_type']):
                sub_df = stim_log.loc[(stim_log.sweep == s) & (stim_log.stim_type == t)]
                tS = sub_df['onset'].min(); tE = sub_df['offset'].max()
                
                evoked_time_window_list.append([tS,tE])
                landmarks.extend([tS,tE])
                evoked_type_list.append(t)
                sweep_list.append(s)
        evoked_time_window_array = np.array(evoked_time_window_list)
                
    #Add injection times to create windows relative to those
    if injection_time_windows is not None:
        if drug_type == 'saline':
            injection_types = ['sal1','sal2']
        elif drug_type == 'psilocybin':
            injection_types = ['sal1','psi']
        elif drug_type == 'ketanserin+psilocybin':
            injection_types = ['ket','psi']
        landmarks.extend(list(np.squeeze(injection_time_windows).ravel()))
        
    if iso_induction_times is not None:
        #Add induction times plus buffer
        landmarks.extend(iso_induction_times)

        #Add 5 minutes to induction time
        t5 = iso_induction_times[1]+5*60
        landmarks.append(t5)
        if stim_exists:
            ##Select first evoked block after iso induction
            ss = np.where(evoked_time_window_array[:,0] > iso_induction_times[-1])[0][0]
            t15 = evoked_time_window_list[ss][0]- window_t
            if t15 > t5:
                landmarks.append(t15) 
    landmarks = sorted(landmarks)

    ## Create windows relative to landmarks 
    time_window_list = []
    for ii, l1 in enumerate(landmarks[:-1]):
        #Get next landmark and define window with 2 points
        l2 = landmarks[ii+1]
        tW = [l1,l2]
        if injection_time_windows is not None:
            #Make sure we're not looking at periods when injection is occuring
            if tW in injection_time_windows:
                # print(tW)
                continue

        #If window is already defined as stimulas evoked window
        #or if the window is less than the desired window length, just add to list as is
        if (tW in evoked_time_window_list) | (np.diff(tW)[0] <  window_t):
            time_window_list.append(tW)

        #Else create more windows between these 2 landmarks at the desired window width
        else:
            # if l2 < injection_times[1]:
            #     window_t2 = 75*60
            # else:
            window_t2 = window_t
            spacing = np.arange(l1,l2,window_t2)
            for jj, t1 in enumerate(spacing[:-1]):
                t2 = spacing[jj+1]
                time_window_list.append([t1,t2])
            if spacing[-1] < l2:
                time_window_list.append([spacing[-1],l2])

    #Elimate windows that are not long enough
    keep = []
    for ll, tW in enumerate(time_window_list):
        if np.diff(tW)[0] > 60:
            keep.append(ll)
    time_window_array = np.array(time_window_list)
    time_window_list = list(time_window_array[keep])

    #Define periods of spontaneous activity
    epoch_list = []; window_type_list = []
    block_labels = []; t2 = injection_time_windows[1,1]
    spont_num = 0
    for s, tW in enumerate(time_window_list):
        tW_c = tW[0] + (tW[1]-tW[0])/2
        if list(tW) in evoked_time_window_list:
            indy = np.where(evoked_time_window_list == tW)[0][0]
            window_type = f'{evoked_type_list[indy]}-sweep-{sweep_list[indy]:02d}'
        else:
            window_type = f'spont-{spont_num:02d}'
            spont_num += 1
        window_type_list.append(window_type)
        if drug_type in ['saline', 'psilocybin','ketanserin+psilocybin']:
            if tW[0] < injection_times[0]:
                epoch_list.append(f'{window_type}_pre-inj')
                block_labels.append('pre-inj')
            elif (tW[0] >= injection_times[0]) & (tW[0] < injection_times[1]):
                epoch_list.append(f'{window_type}_post-{injection_types[0]}-inj')
                block_labels.append(f'post-{injection_types[0]}-inj')
            else:
                epoch_list.append(f'{window_type}_post-{injection_types[1]}-inj')
                iBlock = (tW_c - t2) / 60 // window_t_min
                
                block_labels.append(f'{int(iBlock*window_t_min)}_{int((iBlock+1)*window_t_min)}')
        elif drug_type == 'isoflurane':
            if tW[0] < iso_induction_times[0]:
                epoch_list.append(f'{window_type}_pre-iso')
            elif (tW[0] >= iso_induction_times[0]) & (tW[0] < iso_induction_times[1]):
                epoch_list.append(f'{window_type}_iso-ind')
            else:
                epoch_list.append(f'{window_type}_post-iso')
        elif drug_type == 'urethane':
            t1 = int(tW[0]/60)
            t2 = int(tW[1]/60)
            epoch_list.append(f'{window_type}_{drug_type}_{t1}-{t2}')
        else:
            raise Exception(f"Drug type {drug_type} not recognized.")
        
    
    return epoch_list, block_labels, time_window_list

##------------------------------------------
def bin_spiking_data(probe_unit_data,time_window_list, time_bin=0.1,fr_thresh=2):

    probe_list = list(probe_unit_data.keys())
    probei = probe_list[0]
    if 'areas' in probe_unit_data[probei].keys():
        nNeurons_total = 0
        for probei in probe_list:
            # print(probei)
            # print(probe_unit_data[probei]['areas'])
            uniq_areas, uniq_indices, nNeurons_area = np.unique(np.array(probe_unit_data[probei]['areas'],dtype=str),return_index=True, return_counts=True)
            nNeurons_total += np.sum(nNeurons_area)
            for abbrev, nNeurons in zip(uniq_areas[np.argsort(uniq_indices)],nNeurons_area[np.argsort(uniq_indices)]):
                if (abbrev == 'null') | (abbrev == 'nan'):
                    # print(f'\t{nNeurons:3d} units in {abbrev}')
                    continue
                # area = ccfsum.loc[ccfsum.abbreviation == abbrev]['name'].values[0]
                # print(f'\t{area:13s} -> {nNeurons:3d} neurons')
                # print(f'\t{nNeurons:3d} units in {area}, {abbrev}')
        # print(f'{nNeurons_total:3d} units total')
    
    FR_overall = np.concatenate([probe_unit_data[probei]['firing_rate'] for probei in probe_list])
    FR_mask1 = FR_overall > fr_thresh
    neuron_indices = np.where(FR_mask1)[0]
    N = len(neuron_indices)

    #Get binned spike counts for the spontaneous activity periods between sweeps
    data_list = []; ts_list = []; fr_list = []
    for jj, tWindow in enumerate(time_window_list):
        spk_list = []
        neurons_per_probe = []
        for probei in probe_list:
            spk_counts, bins = get_spike_counts(probe_unit_data[probei]['spike_times'], probe_unit_data[probei]['spike_clusters'], probe_unit_data[probei]['units'], tWindow, time_bin)
            spk_list.append(spk_counts)
            neurons_per_probe.append(spk_counts.shape[0])

        spks = np.concatenate(spk_list)
        X = np.array(spks.T,dtype=np.uint8)
        ts = bins[:-1] + time_bin/2
        ts_list.append(ts)
        fr_list.append(np.sum(X,axis=0)/(X.shape[0]*time_bin))
        data_list.append(X)
    FR_perblock = np.array(fr_list).T

    if 'areas' in probe_unit_data[probei].keys():
        plot_tuple = get_group_plotting_info(probe_unit_data, neuron_indices)
    else:
        plot_tuple = get_probe_plotting_info(probe_unit_data,neuron_indices)

    return data_list, ts_list, neuron_indices, plot_tuple, FR_perblock

##------------------------------------------
def get_spike_counts(spike_times, spike_clusters, unit_ids, tWindow, tBin):

    bins = np.arange(tWindow[0], tWindow[1]+tBin, tBin)
    firingrate = np.empty((len(unit_ids), len(bins)-1))*np.nan

    for indi, uniti in enumerate(unit_ids):
        spikesi = np.squeeze(spike_times[spike_clusters == uniti])
        sp_counts, edges = np.histogram(spikesi, bins)
        firingrate[indi,:] = sp_counts#/tBin

    return firingrate, bins 

##------------------------------------------
def get_group_plotting_info(probe_unit_data, neuron_indices=None):
    probe_list = list(probe_unit_data.keys())

    #Get area assignment
    areas = np.concatenate([np.array(probe_unit_data[probei]['areas'],dtype=str) for probei in probe_list])
    celltypes = np.concatenate([probe_unit_data[probei]['cell_type'] for probei in probe_list])
    durations = np.concatenate([probe_unit_data[probei]['duration'] for probei in probe_list])

    #Get group info
    groups, _, _, _, _, mesogroups, supergroups = determine_groups(areas)
    _, group_dict, graph_order, _, _, _, _ = determine_groups(np.squeeze(areas[neuron_indices]))
    groups_tmp = list(group_dict.keys())
    for g in groups_tmp:
        if len(group_dict[g]) == 0:
            group_dict.pop(g)
            graph_order.pop(g)

    #Get layer info
    layers = []
    for a, g in zip(areas,groups):
        if g in ['HIP','STR','HPF']:
            layers.append('none')
            continue
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

    #Reorder matrix
    group_order = np.array(list(graph_order.keys()))[np.argsort(list(graph_order.values()))]
    group_order_labels = np.array(list(graph_order.keys()))[np.argsort(list(graph_order.values()))]
    order_by_group = []; boundaries = [0]; yticks = []
    areas_sub = np.squeeze(areas[neuron_indices])
    
    counter = 0
    for g in group_order:
        areas_in_group = group_dict[g]
        counter_group = 0
        for a in areas_in_group:
            indy = np.where(areas_sub == a)[0]
            if len(indy) > 0:
                order_by_group.append(indy)
            counter_group += len(indy)
        counter += counter_group
        yticks.append(boundaries[-1]+counter_group/2)
        boundaries.append(counter)
    order_by_group = np.concatenate(order_by_group)
    boundaries.pop(0); boundaries.pop(-1)
    labels = group_order

    plot_tuple = (boundaries, yticks, labels, celltypes, durations, layers, areas, groups, mesogroups, supergroups, order_by_group)
    return plot_tuple
# plot_tuple = get_group_plotting_info(probe_unit_data)
# boundaries, yticks, labels, celltypes, durations, layers, areas, groups, mesogroups, supergroups, order_by_group = plot_tuple
##------------------------------------------
# boundaries, yticks, labels, layers, areas, groups, mesogroups, supergroups, order_by_group = util.get_group_plotting_info2(areas)
def get_group_plotting_info2(areas):
    #Get group info
    groups, group_dict, graph_order, _, _, mesogroups,supergroups = determine_groups(areas)

    groups_tmp = list(group_dict.keys())
    for g in groups_tmp:
        if len(group_dict[g]) == 0:
            group_dict.pop(g)
            graph_order.pop(g)

    #Reorder matrix
    group_order = np.array(list(graph_order.keys()))[np.argsort(list(graph_order.values()))]
    group_order_labels = np.array(list(graph_order.keys()))[np.argsort(list(graph_order.values()))]
    order_by_group = []; boundaries = [0]; yticks = []
    
    counter = 0
    for g in group_order:
        areas_in_group = group_dict[g]
        counter_group = 0
        for a in areas_in_group:
            indy = np.where(areas == a)[0]
            if len(indy) > 0:
                order_by_group.append(indy)
            counter_group += len(indy)
        counter += counter_group
        yticks.append(boundaries[-1]+counter_group/2)
        boundaries.append(counter)
    order_by_group = np.concatenate(order_by_group)
    boundaries.pop(0); boundaries.pop(-1)
    labels = group_order

    #Get layer info
    layers = []
    for a, g in zip(areas,groups):
        if g in ['HIP','STR','HPF']:
            layers.append('none')
            continue
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

    plot_tuple = (boundaries, yticks, labels, layers, areas, groups, mesogroups, supergroups, order_by_group)
    return plot_tuple

##------------------------------------------
def get_probe_plotting_info(probe_unit_data,neuron_indices=None):
    probe_list = list(probe_unit_data.keys())
    neurons_per_probe = [len(np.unique(probe_unit_data[probei]['units'])) for probei in probe_list]
    
    probes = np.concatenate([np.repeat(i,n) for i,n in zip(probe_list,neurons_per_probe)])
    if neuron_indices is not None:
        probes = probes[neuron_indices]

    boundaries = [np.where(probes == i)[0][-1] for i in probe_list]
    boundaries.insert(0,0)
    tmp = np.diff(boundaries)/2
    boundaries.pop(0)
    yticks = boundaries - tmp
    labels = probe_list
    plot_tuple = (boundaries, yticks, labels)

    return plot_tuple

##------------------------------------------
def determine_group_order(groups):

    made_up_groups = ['X','FT','CC','SM-TH','VIS-TH','ANT-TH','RT']
    graph_order_tmp = {'X': 0, 'SM-TH': 639, 'VIS-TH': 640,'ANT-TH':641,'RT':642, 'FT': 1101,'CC': 1198}
    graph_order = {}
    graph_order2  = {}
    for g in groups:
        if g not in made_up_groups:
            tmp_dict = str_tree.get_structures_by_acronym([g])[0]
            go = tmp_dict['graph_order']
            graph_order[g] = go
            graph_order2[tmp_dict['name']] = go
        else:
            go = graph_order_tmp[g]
            graph_order[g] = go
            graph_order2[g] = go
    group_order_labels = np.array(list(graph_order.keys()))[np.argsort(list(graph_order.values()))]
    group_order_names = np.array(list(graph_order2.keys()))[np.argsort(list(graph_order2.values()))]
    return group_order_labels, group_order_names 

##------------------------------------------
def determine_groups(areas):

    th_dict = {'SM-TH': ['PO','VAL','VPL','VPM','VM'], 
                'VIS-TH': ['LP','LGN','LGd','LGd-co','LGd-sh','LGd-ip'],
                'ANT-TH': ['AV','AMd','AMv','AD','MD','MDm','MDc','MDl','RE','RH','CM','LD', 'CL'],
                'TH': ['Eth', 'IAD', 'IGL', 'IntG', 'LGv','MGd', 'MGm', 'MGv', 'PCN', 'PF', 'PIL', 'PoT', 'SGN','SPFp', 'TH'],'RT': ['RT']}
    
    group_dict_tmp = {'X': ['null','root','nan'],'FT':['cing','scwm','fiber tracts','ar', 'em','int', 'or','alv', 'fi','ml','lfbs'],'CC': ['ccb', 'fp'],
    'SM-TH': ['PO','VAL','VPL','VPM','VM'], 'VIS-TH': ['LP','LGN','LGd','LGd-co','LGd-sh','LGd-ip'],'ANT-TH': ['AV','AMd','AMv','AD','MD','MDm','MDc','MDl','RE','RH','CM','LD', 'CL'],'RT': ['RT']} #'root'
    made_up_groups = ['X','FT','CC','SM-TH','VIS-TH','ANT-TH','RT']
    group_dict = {}
    graph_order_tmp = {'X': 0, 'SM-TH': 639, 'VIS-TH': 640,'ANT-TH':641,'RT':642, 'FT': 1101,'CC': 1198}
    graph_order = {}
    graph_order2 = {}
    skip_area = False
    uniq_areas, uniq_indices, num_neurons = np.unique(areas,return_index=True, return_counts=True)
    for a in uniq_areas:
        for k in group_dict_tmp.keys():
            if (a in group_dict_tmp[k]):# or (a in group_dict_tmp['fiber tracts']):
                if k in group_dict.keys():
                    group_dict[k].append(a)
                else:
                    group_dict[k] = [a]
                skip_area = True
                if k not in graph_order.keys():
                    graph_order[k] = graph_order_tmp[k]
                    graph_order2[k] = graph_order_tmp[k]
                continue
        if skip_area:
            skip_area = False
            continue

        area_dict = str_tree.get_structures_by_acronym([a])[0]
        structure_id_path = area_dict['structure_id_path']
        if 1009 in structure_id_path:
            if 'FT' not in group_dict.keys():
                group_dict['FT'] = [a]
            else:
                group_dict['FT'].append(a)
            continue

        # #If area is within thalamus group, just group that
        if 549 in structure_id_path:
            tmp_dict = str_tree.get_structures_by_id([549])[0]
        #If area is within olfactory group, just group that
        elif 698 in structure_id_path:
            tmp_dict = str_tree.get_structures_by_id([698])[0]
        elif 1097 in structure_id_path:
            tmp_dict = str_tree.get_structures_by_id([1097])[0]
        elif 477 in structure_id_path:
            tmp_dict = str_tree.get_structures_by_id([477])[0]
        #Otherwise, take the group 3 levels above
        else:
            if len(structure_id_path) < 3:
                tmp_dict = str_tree.get_structures_by_id([structure_id_path[-1]])[0]
            else:
                tmp_dict = str_tree.get_structures_by_id([structure_id_path[-3]])[0]
            
        #Take structure 1 level below if
        if tmp_dict['acronym'] == 'Isocortex':
            tmp_dict = str_tree.get_structures_by_id([structure_id_path[-2]])[0]

        hierarchical_group = tmp_dict['acronym']
        go = tmp_dict['graph_order']
        
        #Save to dictionary
        if hierarchical_group in group_dict.keys():
            group_dict[hierarchical_group].append(a)
        else:
            group_dict[hierarchical_group] = [a]
            graph_order[hierarchical_group] = go
            graph_order2[tmp_dict['name']] = go

    #Assign each neuron to a group
    groups = []; group_labels = []
    for a in areas:
        in_g = 0
        for g in group_dict.keys():
            if a in group_dict[g]:
                groups.append(g)
                in_g = 1

    groups = np.array(groups)
    group_order = np.array(list(graph_order.keys()))[np.argsort(list(graph_order.values()))]
    group_order_labels = np.array(list(graph_order2.keys()))[np.argsort(list(graph_order2.values()))]

    supergroup_dict = {688: 'CTX',623: 'CNU', 549: 'TH', 1097: 'HY', 313: 'MB', 1065: 'HB', 512: 'CB', 73: 'VS', 1009: 'fiber tracts'}
    supergroups = []
    for a, g in zip(areas,groups):
        if (a == 'nan') | (g == 'X') | (g == 'grey'):
            supergroups.append(g)
            continue
        area_dict = str_tree.get_structures_by_acronym([a])[0]
        structure_id_path = np.array(area_dict['structure_id_path'])

        #Determine supergroup
        for sg_id, sg_name in supergroup_dict.items():
            if sg_id in structure_id_path:
                supergroups.append(sg_name)
                break
    supergroups = np.array(supergroups)

    #Determine "mesogroup"
    mesogroup_dict_tmp = {'TH_core': ['VPL','VPM','LGN','LGd','LGd-co','LGd-sh','LGd-ip'],
                          'TH_matrix':['MG','MGd','MGv','MGm','AV','LD','VAL','PT','MD','PO','PCN','VM','RE','SMT','PIL','POL','AM','AMd','AMv','LP'],
                          'TH_intralaminar': ['PF','IMD','CL','CM','IAD','PVT'],
                          'TH_none': ['LGv','SPFm','LH','MH'],
                          'TH_RT': ['RT'],
                          'TH_unknown': ['AD','PoT','SGN','PF','IntG','Eth','TH','SPFp'],
                          'HY': ['HY','PVZ','PVR','MEZ','LZ','ME'],
                          'CTX_frontal': ['FRP','ORB','MO','PL','ILA','ACA'],
                          
                          'CTX_sensory': ['VIS','SS','SSp','SSs','GU','OLF','AUD','RSP'],
                          'HIP': ['HIP','CA','DG'],
                          'STR': ['STR','STRd','STRv','LSX','sAMY']} #'CTX_associative': ['ACA','RSP','PTLp','TEa'],
        # th_dict = {'SM-TH': ['PO','VAL','VPL','VPM','VM'], 
        #         'VIS-TH': ['LP','LGN','LGd','LGd-co','LGd-sh','LGd-ip'],
        #         'ANT-TH': ['AV','AMd','AMv','AD','MD','MDm','MDc','MDl','RE','RH','CM','LD', 'CL'],
        #         'TH': ['Eth', 'IAD', 'IGL', 'IntG', 'LGv','MGd', 'MGm', 'MGv', 'PCN', 'PF', 'PIL', 'PoT', 'SGN','SPFp', 'TH'],'RT': ['RT']}
    mesogroups = []
    for a, g, sg in zip(areas,groups,supergroups):
        if (a == 'nan') | (g == 'X') | (g == 'grey'):
            mesogroups.append('none')
            continue
        
        mg2 = 'UNKNOWN'
        for mg, areas in mesogroup_dict_tmp.items():
            if a in areas:
                mg2 = mg
                break
            elif g in areas:
                mg2 = mg
                break
        mesogroups.append(mg2)
    mesogroups = np.array(mesogroups)
    return groups, group_dict, graph_order, group_order, group_order_labels, mesogroups, supergroups

# def determine_supergroups(groups):
#     #Determine "mesogroup"
#     mesogroup_dict_tmp = {'TH_core': ['VPL','VPM','LGN','LGd','LGd-co','LGd-sh','LGd-ip'],
#                           'TH_matrix':['MG','MGd','MGv','MGm','AV','LD','VAL','PT','MD','PO','PCN','VM','RE','SMT','PIL','POL','AM','AMd','AMv','LP'],
#                           'TH_intralaminar': ['PF','IMD','CL','CM','IAD','PVT'],
#                           'TH_none': ['LGv','SPFm','LH','MH'],
#                           'TH_RT': ['RT'],
#                           'TH_unknown': ['AD','PoT','SGN','PF','IntG','Eth','TH','SPFp'],
#                           'HY': ['HY','PVZ','PVR','MEZ','LZ','ME'],
#                           'CTX_frontal': ['FRP','ORB','MO','PL','ILA','ACA'],
                          
#                           'CTX_sensory': ['VIS','SS','SSp','SSs','GU','OLF','AUD'],
#                           'HIP': ['HIP','CA','DG'],
#                           'STR': ['STR','STRd','STRv','LSX','sAMY']} #'CTX_associative': ['ACA','RSP','PTLp','TEa'],
#         # th_dict = {'SM-TH': ['PO','VAL','VPL','VPM','VM'], 

#     mesogroups = []
#     for g, sg in zip(groups,supergroups):
#         if (g == 'X') | (g == 'grey'):
#             mesogroups.append('none')
#             continue
        
#         mg2 = 'UNKNOWN'
#         for mg, areas in mesogroup_dict_tmp.items():
#             if a in areas:
#                 mg2 = mg
#                 break
#             elif g in areas:
#                 mg2 = mg
#                 break
#         mesogroups.append(mg2)
#     mesogroups = np.array(mesogroups)
##------------------------------------------
def get_preprocessed_eeg(exp, bad_channels=None):
    try:
        #%% load the EEG data
        eeg_data,eeg_ts = exp.load_eegdata()
        eeg_fs = exp.ephys_params['EEG']['sample_rate']

        fs = 2500
        nyq = 0.5 * fs
        #Band-Pass Filter
        cutoff = [0.1, 100]
        normal_cutoff = np.array(cutoff) / nyq
        b_bpf, a_bpf = sig.butter(3, normal_cutoff, 'bandpass')
        # w_bpf, h_bpf = signal.freqz(b_bpf, a_bpf)

        #Notch filter
        b_notch, a_notch = sig.iirnotch(60, 60, fs)
        # w_notch, h_notch = signal.freqz(b_notch, a_notch, fs=fs)

        #Bandpass filter from 0.1 to 100Hz (Butterworth filter, 3rd order)
        eeg_data_bandpass = sig.filtfilt(b_bpf, a_bpf, eeg_data,axis=0)

        #Remove 60Hz noise
        eeg_data_notch = sig.filtfilt(b_notch, a_notch, eeg_data_bandpass,axis=0)

        #Re-reference to the common average across electrodes or preform laplacan differentiation
        mask = np.ones((30,),dtype=bool)
        if bad_channels is not None:
            mask[bad_channels] = False
        eeg_data_notch[:,~mask] = np.nan
        mean_eeg_signal = np.mean(eeg_data_notch[:,mask],axis=1)
        eeg_data_rereferenced = eeg_data_notch - mean_eeg_signal.reshape(-1,1)

        eeg_ts_sub = eeg_ts[::25]
        eeg_data_sub = eeg_data_rereferenced[::25] #gaussian_filter1d(eeg_data,12,axis=0)[::25]
        mean_signal_sub = mean_eeg_signal[::25]
        plot_eeg = True
    except:
        plot_eeg = False
        eeg_ts_sub = np.array([np.nan])
        eeg_data_sub = np.array([np.nan])

    return eeg_ts_sub, eeg_data_sub, plot_eeg 

##------------------------------------------
def get_behavioral_data(exp, mID, rec_name,normalize=True):

    run_file = os.path.join(base_dir,mID,rec_name,'experiment1','recording1','running_signal.npy')
    run_file2 = os.path.join(base_dir,mID,rec_name,'experiment1','recording1','raw_running_signal.npy')
    if (os.path.exists(run_file)) & (os.path.exists(run_file2)):
        run_signal = np.load(run_file)
        raw_run_signal = np.load(os.path.join(base_dir,mID,rec_name,'experiment1','recording1','raw_running_signal.npy'))

        ts_file = os.path.join(base_dir,mID,rec_name,'experiment1','recording1','running_timestamps.npy')
        if os.path.exists(ts_file):
            run_ts = np.load(ts_file)
        else:
            ts_file = os.path.join(base_dir,mID,rec_name,'experiment1','recording1','running_timestamps_master_clock.npy')
            run_ts = np.load(ts_file)

    else:
        print('\tRunning file does not exist')
        run_signal, raw_run_signal, run_ts = exp.load_running()
        np.save(os.path.join(base_dir,mID,rec_name,'experiment1','recording1','raw_running_signal.npy'),raw_run_signal)
        np.save(os.path.join(base_dir,mID,rec_name,'experiment1','recording1','running_signal.npy'),run_signal)
        np.save(os.path.join(base_dir,mID,rec_name,'experiment1','recording1','running_timestamps.npy'),run_ts)

    # #Load acceleration signal
    # acc_file = os.path.join(base_dir,mID,rec_name,'experiment1','recording1','acceleration_signal.npy')
    # if os.path.exists(acc_file):
    #     acceleration = np.load(acc_file)
    # else:
    #     acceleration = dxdt(run_signal, run_ts, kind="savitzky_golay", left=1, right=1, order=3)
    #     np.save(acc_file,acceleration)


    #Apply savitzky-golay filter to running trace to make it more differentiable
    fw = 500; run_time_bin = 0.01
    run_signal = sig.savgol_filter(run_signal,int(fw/run_time_bin/1000),3)
    f_run = interp1d(run_ts,run_signal)
    acceleration = np.gradient(run_signal)
    f_acc = interp1d(run_ts,acceleration)
    
    #Create gaussian smoothed running signal to condition on
    fw = 750
    run_signal_s = gaussian_filter(run_signal,int(fw/run_time_bin/1000))
    f_run_s = interp1d(run_ts,run_signal_s)

    try:

        pupil_csv = os.path.join(base_dir,mID,rec_name,'experiment1','recording1',f'Pupileye_{rec_name}.csv')
        if os.path.exists(pupil_csv):
            table = pd.read_csv(pupil_csv)
            normalize = True
        else:
            table = pd.read_csv(os.path.join(base_dir,mID,rec_name,'experiment1','recording1',f'Pupil_{rec_name}.csv'))
            normalize = False
        # table = pd.read_csv(os.path.join(base_dir,mID,rec_name,'experiment1','recording1',f'Pupil_{rec_name}.csv'))
        # normalize = False
        pupil_radius = table['Largest_Radius'].values

        #Pupil master clock
        pupil_ts = Movie(filepath=exp.pupilmovie_file,
                        sync_filepath=exp.sync_file,
                        sync_channel='eyetracking'
                        ).sync_timestamps
        plot_pupil = True

        #Ensure timeseries are same length
        t = np.min([len(pupil_ts),len(pupil_radius)])
        pupil_ts = pupil_ts[:t]
        pupil_radius = pupil_radius[:t]

        pupil_radius_z = st.zscore(pupil_radius,nan_policy='omit')
        #Interpolate to equal time bins & remove outliers
        indy = np.where(~np.isnan(pupil_radius) & ~np.isinf(pupil_radius) & (np.abs(pupil_radius_z) < 5))[0]
        f_pupil = interp1d(pupil_ts[indy],pupil_radius[indy])
        pupil_time_bin = 1/30
        pupil_ts_orig = pupil_ts.copy()
        pupil_ts = np.arange(np.nanmin(pupil_ts[indy]),np.nanmax(pupil_ts[indy]),pupil_time_bin)
        pupil_radius = f_pupil(pupil_ts)

        if normalize:
            #Get eye length
            eye_length = table['Eye_Diameter'].values
            eye_length_z = st.zscore(eye_length,nan_policy='omit')
            outliers = np.where(np.abs(eye_length_z) > 3)[0]
            eye_length[outliers] = np.nan

            #Eliminate outliers
            indy = np.where(~np.isnan(eye_length) & ~np.isinf(eye_length) & (np.abs(eye_length_z) < 3))[0]

            #Normalize by median length
            median_length = np.nanmedian(eye_length[indy])
            pupil_radius = pupil_radius/median_length

        #Apply savitzky-golay filter to pupil trace to make it more differentiable
        fw = 500
        pupil_radius_s = sig.savgol_filter(pupil_radius,int(fw/pupil_time_bin/1000),3)
        
        #Interpolate running signal to pupil_ts
        run_signal_p = f_run(pupil_ts)
        run_signal_p_s = f_run_s(pupil_ts)
        # acceleration_p = f_acc(pupil_ts)

        # 2. Savitzky-Golay using cubic polynomials to fit in a centered window of length 1
        dpdt_file = os.path.join(base_dir,mID,rec_name,'experiment1','recording1',f'pupil_velocity_{rec_name}.npy')
        if os.path.exists(dpdt_file):
            pupil_dxdt = np.load(dpdt_file)
        else:
            pupil_dxdt = dxdt(pupil_radius, pupil_ts, kind="savitzky_golay", left=0.5, right=0.5, order=3)
            np.save(dpdt_file,pupil_dxdt)


    except:
        print('\t No Pupil ?!')
        pupil_ts = np.array([np.nan])
        pupil_radius =np.array([np.nan])
        pupil_dxdt = np.array([np.nan])
        run_signal_p = np.array([np.nan])
        run_signal_p_s = np.array([np.nan])
        plot_pupil = False

    return (run_ts, run_signal, run_signal_s, acceleration, pupil_ts, pupil_radius, pupil_dxdt, plot_pupil)

##------------------------------------------
def get_spike_time_dict(probe_unit_data, tWindow=None):

    probe_list = list(probe_unit_data.keys())
    if np.all(tWindow == None):
        max_spike_time = []
        for probei in probe_list:
            max_spike_time.append(np.max(probe_unit_data[probei]['spike_times']))
        tWindow = [0,np.max(max_spike_time)]
    new_index = 0

    spike_time_dict = {}
    for probei in probe_list:
        spike_times = probe_unit_data[probei]['spike_times']
        spike_clusters = probe_unit_data[probei]['spike_clusters']
        unit_ids = probe_unit_data[probei]['units']

        for i, uniti in enumerate(unit_ids):
            spikesi = np.squeeze(spike_times[spike_clusters == uniti])
            indy = np.where((spikesi > tWindow[0]) & (spikesi < tWindow[1]))[0]
            if len(indy) > 0:
                if spikesi.ndim == 0:
                    spike_time_dict[new_index] = spikesi
                else:
                    spike_time_dict[new_index] = spikesi[indy]
            else:
                spike_time_dict[new_index] = np.array([])
            new_index += 1
    return spike_time_dict

##------------------------------------------
# spike_counts, bins = util.get_evoked_spike_counts(spike_times, spike_clusters, unit_ids, events, plot_before, plot_after, time_bin)
def get_evoked_spike_counts(spike_times, spike_clusters, unit_ids, events, plot_before, plot_after, time_bin):

    bins = np.arange(-plot_before, plot_after+time_bin, time_bin)
    spike_counts = np.empty((len(unit_ids), len(events), len(bins)-1))*np.nan

    for indi, uniti in enumerate(unit_ids):
        spikesi = np.squeeze(spike_times[spike_clusters == uniti])

        for iTrial, E in enumerate(events):
            window_spikes = spikesi[np.squeeze(np.argwhere((spikesi >= E-plot_before) & (spikesi <= E+plot_after)))]
            window_spikes = window_spikes - E
            sp_counts, edges = np.histogram(window_spikes, bins)
            spike_counts[indi,iTrial] = sp_counts

    return spike_counts, bins

##------------------------------------------
def get_running_during_evoked_period(run_signal, run_timestamps, events, plot_before, plot_after):
    
    time_bin = 0.01
    bins = np.arange(-plot_before, plot_after+time_bin, time_bin)
    ts = bins[:-1] + time_bin/2

    f = interp1d(run_timestamps,run_signal)
    running_per_trial = np.empty((len(events), len(ts)))*np.nan
    run_ts_per_trial = np.empty((len(events), len(ts)))*np.nan
    for iTrial, E in enumerate(events):
        running_per_trial[iTrial] = f(E+ts)
        run_ts_per_trial[iTrial] = E+ts
        
    return running_per_trial, run_ts_per_trial, ts

##------------------------------------------
def determine_optimal_FCF(FCF_takens,takens_range):
    dFC_threshold = 0.02
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
            peak_indices, _ = sig.find_peaks(fcf_ij)
            
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

##------------------------------------------
def cross_correlation(signal1, signal2,normalize=True):
    """Calculates the cross-correlation between two signals.
    
    Args:
        signal1 (array-like): The first signal.
        signal2 (array-like): The second signal.

    Returns:
        array: The cross-correlation between the two signals.
    """
    signal1 = np.asarray(signal1)
    signal2 = np.asarray(signal2)
    
    # Calculate the lengths of the signals
    len1 = signal1.shape[0]
    len2 = signal2.shape[0]
    
    #Subtract mean
    x = signal1# - np.mean(signal1)
    y = signal2# - np.mean(signal2)

    # Pad the signals to the same length
    max_len = max(len1, len2)
    x = np.pad(x, (0, max_len - len1))
    y = np.pad(y, (0, max_len - len2))

    # Perform the cross-correlation using the Fourier Transform
    xcorr = np.fft.ifft(np.fft.fft(x) * np.fft.fft(y).conj()).real
    
    # Calculate the time lags corresponding to the result
    mid = max_len//2
    time_lags = np.arange(-mid,max_len-mid)

    # Optionally normalize the cross-correlation
    if normalize:
        norm_factor = np.sqrt(np.sum(signal1**2) * np.sum(signal2**2))
        xcorr /= norm_factor

    return time_lags, xcorr

##------------------------------------------
def targeted_attack(G):

    steps = 0
    centrality = nx.betweenness_centrality(G)
    centrality_dist = list(centrality.values())
    indices = np.argsort(centrality_dist)[::-1]
    if (nx.is_empty(G)) | (len(G.nodes()) < 1):
            return steps
    while nx.is_connected(G):
        G.remove_node(indices[steps])
        steps += 1
        if (nx.is_empty(G)) | (len(G.nodes()) < 1):
            return steps
    else:
        return steps
##------------------------------------------
def random_attack(G):
    steps = 0
    if (nx.is_empty(G)) | (len(G.nodes()) < 1):
            return steps
    while nx.is_connected(G):

        node = np.random.choice(G.nodes())
        G.remove_node(node)
        steps += 1
        if (nx.is_empty(G)) | (len(G.nodes()) < 1):
            return steps
    else:
        return steps
    
##------------------------------------------
def get_burst_events(spk_times, isi_interval=0.004, min_spikes = 2, quiescence=0.1):
    #Get ISIs
    isi = np.diff(spk_times)

    #Get where bursts occur (e.g. ISI < 4ms)
    burst_indices_all = np.where(isi <= isi_interval)[0]

    if len(burst_indices_all) == 0:
        return np.array([np.nan]),np.array([np.nan]),np.array([np.nan]),np.array([np.nan])#,np.array([np.nan]),np.array([np.nan])

    #Subselect indices that begin each burst & get number of spikes per burst
    tmp = np.diff(np.concatenate(([-1],burst_indices_all)))
    indy = np.where(tmp > 1)[0]
    burst_indices_s = burst_indices_all[indy]
    nSpikes_per_burst = np.diff(np.concatenate((indy,[len(tmp)]))) + 1
    burst_indices_e = burst_indices_s + nSpikes_per_burst - 1

    #Calculate mean ISI for each burst
    meanISI_per_burst = np.array([np.mean(isi[i:j]) for i,j in zip(burst_indices_s,burst_indices_e)])
    minISI_per_burst = np.array([np.min(isi[i:j]) for i,j in zip(burst_indices_s,burst_indices_e)])
    maxISI_per_burst = np.array([np.max(isi[i:j]) for i,j in zip(burst_indices_s,burst_indices_e)])

    #Apply queiscence criterion if necessary
    indy = np.where(spk_times[burst_indices_s] - spk_times[burst_indices_s-1] > quiescence)[0]
    burst_indices = burst_indices_s[indy]
    nSpikes_per_burst = nSpikes_per_burst[indy]
    meanISI_per_burst = meanISI_per_burst[indy]
    minISI_per_burst = minISI_per_burst[indy]
    maxISI_per_burst = maxISI_per_burst[indy]

    #Apply minimum number of spikes criterion
    indy = np.where(nSpikes_per_burst >= min_spikes)[0]
    burst_indices = burst_indices[indy]
    nSpikes_per_burst = nSpikes_per_burst[indy]
    meanISI_per_burst = meanISI_per_burst[indy]
    minISI_per_burst = minISI_per_burst[indy]
    maxISI_per_burst = maxISI_per_burst[indy]

    #Get burst times
    burst_spk_times = spk_times[burst_indices]
    return burst_spk_times, nSpikes_per_burst, meanISI_per_burst, burst_indices

##------------------------------------------
def get_behavior_during_spike(spk_times, window_dict):
    behav_per_spk = []
    for t in spk_times:
        b = -1
        for key, behavioral_bouts in window_dict.items():
            if len(behavioral_bouts) == 0:
                continue
            tmp = np.array(behavioral_bouts)
            mask = (t >= tmp[:,0]) & (t <= tmp[:,1])
            if mask.any():
                b = key
                break
                
        behav_per_spk.append(b)
    behav_per_spk = np.array(behav_per_spk)
    assert len(behav_per_spk) == len(spk_times)
    return behav_per_spk

##------------------------------------------
def get_epoch_during_spike(spk_times, epoch_dict):
    epoch_per_spk = []
    for t in spk_times:
        which_e = -1
        for key, (tW, e) in epoch_dict.items():
            in_epoch = (t >= tW[0]) & (t <= tW[1])
            if in_epoch:
                which_e = key
                break
                
        epoch_per_spk.append(which_e)
    epoch_per_spk = np.array(epoch_per_spk)
    assert len(epoch_per_spk) == len(spk_times)
    return epoch_per_spk

##------------------------------------------
def get_stim_during_spike(spk_times, stim_window_list):
    window_type = np.array([s[0] for s in stim_window_list])
    window_sweep = np.array([s[1] for s in stim_window_list])
    window_param = np.array([s[2] for s in stim_window_list])
    window_time = np.array([[s[3],s[4]] for s in stim_window_list])

    stim_per_spk = []; sweep_per_spk = []; param_per_spk = []
    for t in spk_times:
        which_type = 'out'
        which_sweep = np.nan
        which_param = np.nan
        mask = (t >= window_time[:,0]) & (t < window_time[:,1])

        if mask.any():
            which_type = window_type[mask][0]
            which_sweep = window_sweep[mask][0]
            which_param = window_param[mask][0]
        stim_per_spk.append(which_type)
        sweep_per_spk.append(which_sweep)
        param_per_spk.append(which_param)

    stim_per_spk = np.array(stim_per_spk)
    sweep_per_spk = np.array(sweep_per_spk)
    param_per_spk = np.array(param_per_spk)
    assert len(stim_per_spk) == len(spk_times)
    assert len(sweep_per_spk) == len(spk_times)
    return stim_per_spk, sweep_per_spk, param_per_spk


# def get_stim_during_spike_old(spk_times, stim_window_dict):
    # stim_per_spk = []; sweep_per_spk = []; num_elements = []
    # for t in spk_times:
    #     which_stim = 'out'
    #     which_sweep = np.nan
    #     n = 0
    #     for key, tmp in stim_window_dict.items():
    #         if len(tmp) == 0:
    #             continue
    #         sweep = np.array(tmp)[:,0]
    #         time_windows = np.array(tmp)[:,1:]
  
    #         mask = (t >= time_windows[:,0]) & (t < time_windows[:,1])
    #         if mask.any():
    #             n = np.sum(mask)
    #             which_stim = key
    #             which_sweep = sweep[mask][0]
    #             break
    #     num_elements.append(n)
    #     stim_per_spk.append(which_stim)
    #     sweep_per_spk.append(which_sweep)
    # stim_per_spk = np.array(stim_per_spk)
    # sweep_per_spk = np.array(sweep_per_spk)
    # assert len(stim_per_spk) == len(spk_times)
    # assert len(sweep_per_spk) == len(spk_times)
    # return stim_per_spk, sweep_per_spk

##------------------------------------------
def perm_test_statistic(x, y):
    return np.mean(x) - np.mean(y)

# from scipy.signal import butter, sosfiltfilt, hilbert, stft
##------------------------------------------
def hilbert_transform(lfp_matrix, fs):
    T, N = lfp_matrix.shape

    def butter_bandpass(lowcut, highcut, fs, order):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        filter = sig.butter(order, [low, high], btype='band', output='sos')
        return filter

    # filters
    bandpass_freqs = np.array([[1, 4], [4, 8], [8,13], [13, 30], [30, 55], [65, 100], [100,200]])
    n_bands = len(bandpass_freqs)

    lfp_bandpass = np.zeros((T,N,n_bands),dtype='float32')
    lfp_hilbert = np.zeros((T,N,n_bands),dtype='float32')
    lfp_phase = np.zeros((T,N,n_bands),dtype='float32')

    for i, freq_range in enumerate(bandpass_freqs):
        filter = butter_bandpass(freq_range[0], freq_range[1], fs, 11)
        for j in range(N):
            lfp_filtered = sig.sosfiltfilt(filter, lfp_matrix[:,j])
            lfp_bandpass[:,j,i] = lfp_filtered
            lfp_hilbert[:,j,i] = np.abs(sig.hilbert(lfp_filtered))  # get envelope
            lfp_phase[:,j,i] = np.angle(sig.hilbert(lfp_filtered))
            # lfp_bandpass[i,j] = lfp_bandpass[i,j] / np.nanmax(lfp_bandpass[i,j])
    

    return lfp_hilbert, lfp_bandpass, lfp_phase