base_dir_server = '/allen/programs/mindscope/workgroups/templeton-psychedelics/'
base_dir_local = '/data/tiny-blue-dot/zap-n-zip/EEG_exp'; base_dir = base_dir_local
ProjDir = '/home/david.wyrick/projects/zap-n-zip/'
ServDir = '/data/projects/zap-n-zip/'

#Base
import json, os, time, sys
import argparse
from glob import glob
from os.path import join
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

#Proj
from tbd_eeg.data_analysis.eegutils import EEGexp
import tbd_eeg.data_analysis.Utilities.utilities as tbd_util
from tbd_eeg.data_analysis.Utilities.behavior_movies import Movie

#User
sys.path.append(ProjDir)
import util
import plotting as usrplt

#CCM
from delay_embedding import ccm
from delay_embedding import helpers as H
from delay_embedding import surrogate as S
import ray

##===== ============================ =====##
##===== Parse Command Line Arguments =====##
parser = argparse.ArgumentParser(description='FCF')

##===== Data Options =====##
parser.add_argument('--mouseID',type=str, default='mouse689241',
                    help='mouse to perform analysis on')

parser.add_argument('--time_bin_ms',type=int, default=100,
                    help='time_bin_ms')

parser.add_argument('--delay',type=int, default=1,
                    help='tau')

parser.add_argument('--rand_proj',type=int, default=0,
                    help='random projection of delay vectors?')

parser.add_argument('--xval',type=int, default=0,
                    help='cross validation?')

parser.add_argument('--nKfold',type=int, default=10,
                    help='# of folds')

parser.add_argument('--zscore',type=int, default=1,
                    help='zscore spike counts?')

parser.add_argument('--fr_thresh',type=float, default=2.5,
                    help='Firing rate threshold for neurons to include in analysis')

parser.add_argument('--rand_proj_dist',type=str, default='normal',
                    help='Distribution to draw random matrix')

if __name__ == '__main__':

    # Parse the arguments
    args = parser.parse_args()
    mouse_name = args.mouseID 
    mID = args.mouseID
    time_bin_ms = args.time_bin_ms
    time_bin = time_bin_ms/1000
    delay = args.delay
    
    rand_proj = bool(args.rand_proj)
    xval = bool(args.xval)
    zscore = bool(args.zscore)

    nKfold = args.nKfold
    fr_thresh = args.fr_thresh
    rand_proj_dist = args.rand_proj_dist
    
    base_dir = '/data/tiny-blue-dot/zap-n-zip/EEG_exp' #local

    #%% define the file name to analyze
    if mID == 'mouse678912':
        rec_name = 'spont_aw_psi_2023-06-22_11-42-00'
    elif  mID == 'mouse678913':
        rec_name = 'spont_aw_psi_2023-06-29_12-49-40'
    elif  mID == 'mouse689241':
        rec_name = 'spont_aw_psi_2023-07-27_11-05-05'
    elif  mID == 'mouse666194':
        rec_name = 'pilot_aw_psi_2023-02-23_10-40-34'
    elif  mID == 'mouse689239':
        rec_name = 'aw_psi_2023-08-10_11-26-36'
    elif mID == 'mouse692643':
        rec_name = 'spont_aw_psi_2023-08-31_12-41-54'

    # ('mouse689241', 'spont_aw_psi_2023-07-27_11-05-05'),('mouse689239','aw_psi_2023-08-10_11-26-36'),('mouse692643','spont_aw_psi_2023-08-31_12-41-54')
    print(mouse_name,rec_name)
    file_name = join(base_dir,mouse_name,rec_name,'experiment1','recording1')

    #%% Upload the whole experiment and generate the global clock
    exp = EEGexp(file_name, preprocess=False, make_stim_csv=False) # the data you have are already preprocessed and the stim_log file is in the folder

    #Create directory for saving to
    if xval:
        TempDir = join(ServDir,'results','ccm_xval',mID)
    else:
        TempDir = join(ServDir,'results','ccm',mID)
    if not os.path.exists(TempDir):
        os.makedirs(TempDir)

    tmp_list = sorted(glob(join(TempDir,f'{rec_name}_run_*')))
    if len(tmp_list) == 0:
        curr_run = 0
    else:
        last_run = int(tmp_list[-1][-2:])
        curr_run = last_run+1
    folder = f'{rec_name}_run_{curr_run:02d}'

    SaveDir = join(TempDir,folder)
    if not os.path.exists(SaveDir):
        os.makedirs(SaveDir)
    
    #Get running signal
    run_file = join(base_dir,mouse_name,rec_name,'experiment1','recording1','running_signal.npy')
    run_signal = np.load(run_file)

    ts_file = join(base_dir,mouse_name,rec_name,'experiment1','recording1','running_timestamps.npy')
    if os.path.exists(ts_file):
        run_timestamps = np.load(ts_file)
    else:
        ts_file = join(base_dir,mouse_name,rec_name,'experiment1','recording1','running_timestamps_master_clock.npy')
        run_timestamps = np.load(ts_file)

    #Get pupil diameter
    base_dir_server = '/allen/programs/mindscope/workgroups/templeton-psychedelics/'
    pupil_csv = join(base_dir_server,mID,rec_name,'experiment1','recording1',f'Pupil_{rec_name}.csv')

    try:
        table = pd.read_csv(pupil_csv)
        pupil_radius = table['Largest_Radius'].values

        #Pupil master clock
        pupil_ts = Movie(filepath=exp.pupilmovie_file,
                        sync_filepath=exp.sync_file,
                        sync_channel='eyetracking'
                        ).sync_timestamps
        pupil_data = True
    except:
        pupil_ts = np.array([np.nan])
        pupil_radius =np.array([np.nan])
        pupil_data = False

    #Save model parameters
    args_dict = args.__dict__
    args_dict['rec_name'] = rec_name
    args_dict['SaveDir'] = SaveDir
    with open(join(SaveDir,f"ccm_parameters_run_{curr_run}.json"), "w") as outfile:
        json.dump(args_dict, outfile)

    ## Get probe data
    probe_unit_data, probe_info, total_units = tbd_util.get_neuropixel_data(exp)
    probe_list = [x.replace('_sorted', '') for x in exp.experiment_data if 'probe' in x]

    #Get overall firing rates
    FR = np.concatenate([probe_unit_data[probei]['firing_rate'] for probei in probe_list])
    FR_mask1 = FR > fr_thresh
    neuron_indices = np.where(FR_mask1)[0]
    N = len(neuron_indices)
    print(f'{N} neurons > {fr_thresh} Hz overall')
    if mID == 'mouse678912': #[2522.03748, 3182.864709] 2582.03748 3002.864709 3242.864709
        time_window_list = [[48.0, 948.0],[948.0, 1848.0],[2582.04, 3002.9],[3242.9, 4142.9],[4142.9, 5042.9],[5042.9, 5942.9],[5942.9, 6842.9],[6842.9, 7742.9],[7142.9, 8042.9]]
        filename_list = ['spont_no-inj_0-15','spont_no-inj_15-30','spont_saline_42-52','spont_psilo_0-15','spont_psilo_15-30','spont_psilo_30-45','spont_psilo_45-60','spont_psilo_60-75','spont_psilo_65-80']
    elif  mID == 'mouse678913': #[2608.995, 3232.207429] 2668.995 3052.207429 3292.207429
        time_window_list = [[50.0, 950.0],[950.0, 1850.0],[2669, 3172.3],[3292, 4192],[4192, 5092],[5092, 5992],[5992, 6802]]
        filename_list = ['spont_no-inj_0-15','spont_no-inj_15-30','spont_saline_42-52','spont_psilo_0_15','spont_psilo_15_30','spont_psilo_30_45','spont_psilo_45_60']
    elif mID == 'mouse689241': #[996.963893, 3920.886158]  1056.963893 3740.886158 3980.886158
        filename_list = ['spont_no-inj_0-15','spont_saline_17-32','spont_saline_32-47','spont_saline_47-64','spont_psilo_66-81','spont_psilo_81-96','spont_psilo_96-111','spont_psilo_110-125']
        time_window_list = [[43.0, 937.0],[1057.0, 1957.0],[1957.0, 2857.0],[2857.0, 3741.0],[3981.0, 4881.0],[4881.0, 5781.0],[5781.0, 6681.0],[6613.0, 7513.0]]
    elif mID == 'mouse689239': #[2579.235348, 3300.724459] 2639.235348 3120.724459 3360.724459
        filename_list = ['spont_no-inj','stim_no-inj_0-119','stim_no-inj_120-239','spont_saline','spont_psilo','stim_psilo_0-119','stim_psilo_120-239']
        time_window_list = [[32.0, 748.63505],[748.63505, 1593.34048],[1593.34048, 2435.40909],[2639.24, 3120.73],[3360.73, 4018.44],[4018.43496, 4863.13915],[4863.13915, 5705.50668]]
    elif mID == 'mouse692643': #[984.3731669, 3662.870292] 1044.3731669 3482.870292 3722.870292
        time_window_list = [[43.0, 924.37],[1044.37, 1929.37],[1929.37, 2829.37],[2717.87, 3482.87],[3722.87, 4367.87],[4607.87, 5507.87],[5507.87, 6407.87]]
        filename_list = ['spont_no-inj','spont_saline_0-15','spont_saline_15-30','spont_saline_30-45','spont_psilo_0-15','spont_psilo_15-30','spont_psilo_30-45']
    else:
        raise Exception(f'Windows not set for {mID}')

    # if mID == 'mouse678912':
    #     # time_window_list = [[48.0, 648.0],[648.0, 1248.0],[1248.0, 1848.0],[1868.5781, 2400.43032],[2522.037479958106, 3122.864708975252],[3242.9, 3842.9],[3842.9, 4442.9],[4442.9, 5042.9],[5042.9, 5642.9],[5642.9, 6242.9],[6242.9, 6842.9],[6842.9, 7442.9],[7442.9, 8042.9]]
    #     # filename_list = ['spont_no-inj_0-10','spont_no-inj_10-20','spont_no-inj_20-30','evoked_no-inj_31-40','spont_saline_42-52','spont_psilo_54-64','spont_psilo_64-74','spont_psilo_74-84','spont_psilo_84-94','spont_psilo_94-104','spont_psilo_104-114','spont_psilo_114-124','spont_psilo_124-134']
    #     time_window_list = [[48.0, 948.0],[948.0, 1848.0],[2522.04, 3122.9],[3242.9, 4142.9],[4142.9, 5042.9],[5042.9, 5942.9],[5942.9, 6842.9],[6842.9, 7742.9],[7142.9, 8042.9]]
    #     filename_list = ['spont_no-inj_0-15','spont_no-inj_15-30','spont_saline_42-52','spont_psilo_0-15','spont_psilo_15-30','spont_psilo_30-45','spont_psilo_45-60','spont_psilo_60-75','spont_psilo_65-80']
    # elif  mID == 'mouse678913':
    #     # filename_list = ['spont_no-inj_0-10','spont_no-inj_10-20','spont_no-inj_20-30','evoked_no-inj_31-40','spont_saline_42-52','spont_psilo_54-64','spont_psilo_64-74','spont_psilo_74-84','spont_psilo_84-94','spont_psilo_94-104','spont_psilo_103-113']
    #     # time_window_list = [[50.0, 650.0],[650.0, 1250.0],[1250.0, 1850.0],[1856.931, 2388.783],[2608.995, 3172.2074],[3292.2, 3892.2],[3892.2, 4492.2],[4492.2, 5092.2],[5092.2, 5692.2],[5692.2, 6292.2],[6202.0, 6802.0]]
    #     time_window_list = [[50.0, 950.0],[950.0, 1850.0],[2608.995, 3172.2074],[3292, 4192],[4192, 5092],[5092, 5992],[5992, 6802]]
    #     filename_list = ['spont_no-inj_0-15','spont_no-inj_15-30','spont_saline_42-52','spont_psilo_0_15','spont_psilo_15_30','spont_psilo_30_45','spont_psilo_45_60']
    # elif mID == 'mouse689241':
    #     filename_list = ['spont_no-inj_0-15','spont_saline_17-32','spont_saline_32-47','spont_saline_47-64','spont_psilo_66-81','spont_psilo_81-96','spont_psilo_96-111','spont_psilo_110-125']
    #     time_window_list = [[43.0, 937.0],[1057.0, 1957.0],[1957.0, 2857.0],[2857.0, 3861.0],[3981.0, 4881.0],[4881.0, 5781.0],[5781.0, 6681.0],[6613.0, 7513.0]]
    # elif mID == 'mouse689239':
    #     filename_list = ['spont_no-inj','stim_no-inj_0-119','stim_no-inj_120-239','spont_saline','spont_psilo','stim_psilo_0-119','stim_psilo_120-239']
    #     time_window_list = [[32.0, 748.63505],[748.63505, 1593.34048],[1593.34048, 2435.40909],[2609.24, 3240.73],[3330.73, 4018.44],[4018.43496, 4863.13915],[4863.13915, 5705.50668]]
    #     # filename_list = ['spont_no-inj', 'stim_no-inj', 'spont_saline', 'spont_psilo', 'stim_psilo']
    #     # time_window_list = [[32.0, 748.63505], [748.63505, 2435.40909], [2609.2353481797095, 3240.7244587390646], [3330.7244587390646, 4018.43496],  [4018.43496, 5705.50668]]
    # elif mID == 'mouse692643':  
    #     time_window_list = [[43.0, 924.37],[1029.37, 1929.37],[1929.37, 2829.37],[2717.87, 3617.87],[3707.87, 4367.87],[4607.87, 5507.87],[5507.87, 6407.87]]
    #     filename_list = ['spont_no-inj','spont_saline_0-15','spont_saline_15-30','spont_saline_30-45','spont_psilo_0-15','spont_psilo_15-30','spont_psilo_30-45']
    # elif mID == 'mouse666194':
    #     filename_list = ['spont_no-inj','spont_saline','stim_sweep_0','spont_sweep_1','stim_sweep_1','spont_sweep_2','stim_sweep_2','spont_sweep_3','stim_sweep_3','spont_sweep_4','stim_sweep_4','spont_sweep_5','stim_sweep_5','spont_sweep_6','stim_sweep_6','spont_sweep_7','stim_sweep_7','spont_sweep_8','stim_sweep_8','spont_sweep_9','stim_sweep_9']
    #     time_window_list = [[33.0, 277.0084],[337.0084, 842.8856],[1057.14291, 1536.91339],[1536.91339, 1893.77959],[1893.77959, 2373.54988],[2373.54988, 2677.77027],[2677.77027, 3157.54058],[3157.54058, 3463.01564],[3463.01564, 3942.78598],[3942.78598, 4266.56828],[4266.56828, 4746.33865],[4746.33865, 5055.5823],[5055.5823, 5535.35298],[5535.35298, 5798.73814],[5798.73814, 6278.50897],[6278.50897, 6590.49455],[6590.49455, 7070.26535],[7070.26535, 7432.33897],[7432.33897, 7912.10973],[7912.10973, 8222.34515],[8222.34515, 8702.11589]]

    #Save time windows
    np.savez(join(SaveDir,f'time_windows.npz'),time_window_list = np.array(time_window_list),filename_list=filename_list,neuron_indices=neuron_indices)

    #Let's loop over taken's dimension to see what the optimal embedding dimension is
    # takens_range = np.concatenate(([1],np.arange(2,51,2))) 
    takens_range = np.concatenate((np.arange(1,11),np.arange(12,31,2)))

    processes = []; running_list = []; pupil_list = []
    # tmpDir = '/data/projects/zap-n-zip/data'
    # DataDir = join(tmpDir,mID,'regressed')
    # print('Using regressed z-scored firing rates')
    def usr_zscore(X):
        mx = np.mean(X,axis=0); std = np.std(X,axis=0)
        Xz = np.divide(X-mx,std,out=np.zeros(X.shape),where = std!= 0)
        return Xz
    
    #Loop over different time blocks
    for ii, tWindow in enumerate(time_window_list):
        if ii < 2:
            continue
        print(filename_list[ii])
        print(f'\tBinning spikes in {time_bin_ms} ms windows')
        if ('saline' not in filename_list[ii]):
            print('Already calculated')
            continue

        #Get spike counts
        # spks = np.concatenate([get_firing_rates(probe_unit_data[probei]['spike_times'], probe_unit_data[probei]['spike_clusters'], probe_unit_data[probei]['units'], tWindow, time_bin)[0] for probei in probe_list])
        spk_list = []
        for probei in probe_list:
            firingrate, bins = util.get_spike_counts(probe_unit_data[probei]['spike_times'], probe_unit_data[probei]['spike_clusters'], probe_unit_data[probei]['units'], tWindow, time_bin)
            spk_list.append(firingrate)
        spks = np.concatenate(spk_list)
        X = spks[neuron_indices].T
        ts = bins[:-1] + time_bin/2
        
        #Interpolate behavioral traces to spike count time series
        f_run = interp1d(run_timestamps, run_signal)
        run_sub = f_run(ts); running_list.append(run_sub)
        X = np.hstack((X,run_sub.reshape(-1,1)))

        if pupil_data:
            indy = np.where(~np.isnan(pupil_radius))[0]
            f_pupil = interp1d(pupil_ts[indy], pupil_radius[indy])
            pupil_sub = f_pupil(ts); pupil_list.append(pupil_sub)
            X = np.hstack((X,pupil_sub.reshape(-1,1)))

        T,N = X.shape
        print('\t# of time points and neurons: ',X.shape)

        #z-score data so scale of different neural activities doesn't affect kNN weights
        if zscore:
            print('\tz-score! ')
            X = usr_zscore(X)

        print(f'\tCalculating functional causal flow for {filename_list[ii]}')
        t0 = time.perf_counter()

        #Parallelize
        ray.init(num_cpus=30,include_dashboard=True, dashboard_port=5432)
        X_id = ray.put(X)

        #Create progress bar
        if xval:
            num_ticks = len(takens_range)*nKfold
        else:
            num_ticks = len(takens_range)
        pb = H.ProgressBar(num_ticks); actor = pb.actor

        ## Loop over taken's dimension
        obj_list = [ccm.FCF.remote(X=X_id,delay=delay,dim=dim,rand_proj=rand_proj,n_neighbors=np.max([dim+1,4]),rand_proj_dist=rand_proj_dist) for dim in takens_range]

        #Start a series of remote Ray tasks 
        if xval:
            print('Cross validation...')
            processes = [fcf.cross_validate.remote(nKfold=nKfold, pba=actor) for fcf in obj_list]
        else:
            processes = [fcf.calculate_connectivity.remote(pba = actor) for fcf in obj_list]

        #And then print progress bar until done
        pb.print_until_done()

        #Initiate parallel processing
        results = list(ray.get(processes))
        FCF_takens = np.array([r[0] for r in results])
        correlation = np.array([r[1] for r in results])
        # FCF_takens = np.array(ray.get(processes))

        #Shutdown parallel threads
        ray.shutdown()
        tE = (time.perf_counter() - t0)/60
        print('{}  {:.2f} mins'.format(filename_list[ii],tE), end=', ')

        #Save!!
        fsuffix = f'{filename_list[ii]}_{time_bin_ms}ms-bins_tau-{delay}'
        if zscore:
           fsuffix = f'{fsuffix}_z' 
        if xval:
            fsuffix = f'{fsuffix}_xval'
        if rand_proj:
            fsuffix = f'{fsuffix}_randproj'

        np.savez(join(SaveDir,f'FCF_{fsuffix}.npz'),FCF_takens=FCF_takens,correlation=correlation)
    print('Done!')

