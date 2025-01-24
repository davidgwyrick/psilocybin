#Base
import argparse
from glob import glob
from os.path import join
import json, os, time, sys
import numpy as np
import xarray as xr
import pandas as pd
import scipy.stats as st
from tqdm.notebook import trange, tqdm
import ray

#Plot
PlotDir = '/home/david.wyrick/projects/zap-n-zip/plots'
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from ssm.plots import gradient_cmap, white_to_color_cmap

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

#CCM
from delay_embedding import ccm
from delay_embedding import evaluation as E
from delay_embedding import helpers as H
from delay_embedding import surrogate as S
import ray

ProjDir = '/home/david.wyrick/projects/zap-n-zip/'
ServDir = '/data/projects/zap-n-zip/'
PlotDir = os.path.join(ProjDir,'plots','ccm')

##===== ============================ =====##
##===== Parse Command Line Arguments =====##
parser = argparse.ArgumentParser(description='FCF')

##===== Data Options =====##
parser.add_argument('--sID',type=str, default='766640955',
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

parser.add_argument('--zscore',type=int, default=0,
                    help='zscore spike counts?')

parser.add_argument('--fr_thresh',type=float, default=2.5,
                    help='Firing rate threshold for neurons to include in analysis')

parser.add_argument('--rand_proj_dist',type=str, default='normal',
                    help='Distribution to draw random matrix')
if __name__ == '__main__':

    # Parse the arguments
    args = parser.parse_args()
    mouse_name = args.sID 
    mID = f'm{args.sID}'
    time_bin_ms = args.time_bin_ms
    time_bin = time_bin_ms/1000
    delay = args.delay
    
    rand_proj = bool(args.rand_proj)
    xval = bool(args.xval)
    zscore = bool(args.zscore)
    nKfold = args.nKfold
    fr_thresh = args.fr_thresh
    rand_proj_dist = args.rand_proj_dist
    
    base_dir = '/data/projects/zap-n-zip/data/visual_coding_np' #local
    rec_name = 'dg75'

    #Create directory for saving to
    TempDir = os.path.join(ServDir,'results','ccm',mID)
    if not os.path.exists(TempDir):
        os.makedirs(TempDir)

    tmp_list = sorted(glob(join(TempDir,f'{rec_name}_run_*')))
    if len(tmp_list) == 0:
        curr_run = 0
    else:
        last_run = int(tmp_list[-1][-1])
        curr_run = last_run+1
    folder = f'{rec_name}_run_{curr_run}'

    SaveDir = os.path.join(TempDir,folder)
    if not os.path.exists(SaveDir):
        os.makedirs(SaveDir)
    # import pdb; pdb.set_trace()
        
    #Save model parameters
    args_dict = args.__dict__
    args_dict['rec_name'] = rec_name
    args_dict['SaveDir'] = SaveDir
    with open(join(SaveDir,f"ccm_parameters_run_{curr_run}.json"), "w") as outfile:
        json.dump(args_dict, outfile)

    #Let's loop over taken's dimension to see what the optimal embedding dimension is
    takens_range = np.arange(2,51,2)

    data = np.load(join(base_dir,f'data_arrays_{mouse_name}.npz'))
    evoked_spk_counts=data['evoked_spk_counts']
    spont_spk_counts=data['spont_spk_counts']
    data_list = [evoked_spk_counts[0],spont_spk_counts[0],spont_spk_counts[1],evoked_spk_counts[1]]
    filename_list = ['evoked1','spont1','spont2','evoked2']
    processes = []
    
    #Loop over different time blocks
    for ii, X in enumerate(data_list):
        print(f'\tBinning spikes in {time_bin_ms} ms windows')

        T,N = X.shape
        print('\t# of time points and neurons: ',X.shape)

        #z-score data so scale of different neural activities doesn't affect kNN weights
        if zscore:
            print('\tz-score! ')
            mx = np.mean(X,axis=0); std = np.std(X,axis=0)
            Xz = np.divide(X-mx,std,out=np.zeros(X.shape),where = std!= 0)
            X = Xz

        t0 = time.perf_counter()

        #Parallelize
        ray.init(num_cpus=26,include_dashboard=True, dashboard_port=5432)
        X_id = ray.put(X)

        #Create progress bar
        if xval:
            num_ticks = len(takens_range)*nKfold
        else:
            num_ticks = len(takens_range)
        pb = H.ProgressBar(num_ticks); actor = pb.actor

        ## Loop over taken's dimension
        obj_list = [ccm.FCF.remote(X=X_id,delay=delay,dim=dim,rand_proj=rand_proj,n_neighbors=dim+1,rand_proj_dist=rand_proj_dist) for dim in takens_range]

        #Start a series of remote Ray tasks 
        if xval:
            print('Cross validation...')
            processes = [fcf.cross_validate.remote(nKfold = nKfold, pba = actor) for fcf in obj_list]
        else:
            processes = [fcf.calculate_connectivity.remote(pba = actor) for fcf in obj_list]

        #And then print progress bar until done
        pb.print_until_done()

        #Initiate parallel processing
        FCF_takens = np.array(ray.get(processes))
        ray.shutdown()

        tE = (time.perf_counter() - t0)/60

        #Save!!
        fsuffix = f'{filename_list[ii]}_{time_bin_ms}ms-bins_tau-{delay}'
        if zscore:
           fsuffix = f'{fsuffix}_z' 
        if xval:
            fsuffix = f'{fsuffix}_xval'
        if rand_proj:
            fsuffix = f'{fsuffix}_randproj'

        np.save(os.path.join(SaveDir,f'FCF_{fsuffix}.npy'),FCF_takens)

    print('Done!')

