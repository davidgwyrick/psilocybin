   # from itertools import combinations
    # reorg = []
    # for jj, tW_c in enumerate(time_window_centers):

    #     if tW_c < injection_times[0]:
    #         e = 'pre-inj'
    #     elif (tW_c >= injection_times[0]) & (tW_c < injection_times[1]):
    #         e = f'post-{injection_types[0]}-inj'
    #     else:
    #         e = f'post-{injection_types[1]}-inj'

    #     reorg.append((mID,rec_name,drug_type,e,'FCF','algorithm',mod_FCF[jj,0]))
    #     reorg.append((mID,rec_name,drug_type,e,'FCF','area',mod_FCF[jj,1]))
    #     reorg.append((mID,rec_name,drug_type,e,'corr','algorithm',mod_corr[jj,0]))
    #     reorg.append((mID,rec_name,drug_type,e,'corr','area',mod_corr[jj,1]))
    # mod_df = pd.DataFrame(np.stack(reorg),columns=['mID','rec_name','drug','epoch','mat','type','mod'])
    # mod_df = mod_df.astype({'mID':str,'rec_name':str,'drug':str,'epoch':str,'mat':str,'type':str,'mod':float})   

    # ## Plot modularity ----------------------------------------
    # label_list = [f'pre-inj',f'post-{injection_types[0]}-inj',f'post-{injection_types[1]}-inj']
    # fig, axes = plt.subplots(1,2,figsize=(10,5))
    # plt.suptitle(f'Modularity on FCF matrix; {mID}, {rec_name}',y=1.05)
    # ax = axes[0]
    # sns.histplot(data=mod_df[(mod_df.type == 'algorithm') & (mod_df.mat == 'FCF')],x='mod',hue='epoch',stat='probability',common_norm=True,multiple='layer',kde=True,ax=ax,palette=cmap,legend=True)
    # ax.set_title('Modularity calculated on\n communities algorithmically defined')

    # ylim = ax.get_ylim()
    # for kk, cb in enumerate(combinations(label_list,2)):
    #     x = mod_df[(mod_df.type == 'algorithm') & (mod_df.mat == 'FCF') & (mod_df.epoch == cb[0])]['mod'].values
    #     y = mod_df[(mod_df.type == 'algorithm') & (mod_df.mat == 'FCF') & (mod_df.epoch == cb[1])]['mod'].values
    #     if (len(x) == 0) | (len(y) == 0):
    #         continue
    #     res = pg.ttest(x,y)
    #     pval = res['p-val'].values[0]
    #     if pval < 0.05/3:
    #         ax.text(np.round(np.nanpercentile(mod_FCF[:,0],50),2),(ylim[1]/10)*(kk+1),f'{cb[0]} - {cb[1]} *')

    # ax = axes[1]
    # # sns.histplot(mod_FCF[:,1],ax=axes[1],color=usrplt.cc[2])
    # sns.histplot(data=mod_df[(mod_df.type == 'area') & (mod_df.mat == 'FCF')],x='mod',hue='epoch',stat='probability',common_norm=True,multiple='layer',kde=True,ax=ax,palette=cmap,legend=True)
    # ax.set_title('Modularity calculated on\n communities anatomically defined')
    # # ax.legend()

    # ylim = ax.get_ylim()
    # for kk, cb in enumerate(combinations(label_list,2)):
    #     x = mod_df[(mod_df.type == 'area') & (mod_df.mat == 'FCF') & (mod_df.epoch == cb[0])]['mod'].values
    #     y = mod_df[(mod_df.type == 'area') & (mod_df.mat == 'FCF') & (mod_df.epoch == cb[1])]['mod'].values
    #     if (len(x) == 0) | (len(y) == 0):
    #         continue
    #     res = pg.ttest(x,y)
    #     pval = res['p-val'].values[0]
    #     if pval < 0.05/3:
    #         ax.text(np.round(np.nanpercentile(mod_FCF[:,1],50),2),(ylim[1]/10)*(kk+1),f'{cb[0]} - {cb[1]} *')
    # plt.savefig(join(PlotDir,f'modularity_FCF-mat_distri_{mID}_{rec_name}.png'),facecolor='white',dpi=300,bbox_inches='tight')
    # pdfdoc.savefig(fig)

    # ## Plot modularity ----------------------------------------
    # label_list = [f'pre-inj',f'post-{injection_types[0]}-inj',f'post-{injection_types[1]}-inj']
    # fig, axes = plt.subplots(1,2,figsize=(10,5))
    # plt.suptitle(f'Modularity on correlation matrix; {mID}, {rec_name}',y=1.05)
    # ax = axes[0]
    # sns.histplot(data=mod_df[(mod_df.type == 'algorithm') & (mod_df.mat == 'corr')],x='mod',hue='epoch',stat='probability',common_norm=True,multiple='layer',kde=True,ax=ax,palette=cmap,legend=True)
    # ax.set_title('Modularity calculated on\n communities algorithmically defined')

    # ylim = ax.get_ylim()
    # for kk, cb in enumerate(combinations(label_list,2)):
    #     x = mod_df[(mod_df.type == 'algorithm') & (mod_df.mat == 'corr') & (mod_df.epoch == cb[0])]['mod'].values
    #     y = mod_df[(mod_df.type == 'algorithm') & (mod_df.mat == 'corr') & (mod_df.epoch == cb[1])]['mod'].values
    #     if (len(x) == 0) | (len(y) == 0):
    #         continue
    #     res = pg.ttest(x,y)
    #     pval = res['p-val'].values[0]
    #     if pval < 0.05/3:
    #         ax.text(np.round(np.nanpercentile(mod_corr[:,0],50),2),(ylim[1]/10)*(kk+1),f'{cb[0]} - {cb[1]} *')

    # ax = axes[1]
    # # sns.histplot(mod_FCF[:,1],ax=axes[1],color=usrplt.cc[2])
    # sns.histplot(data=mod_df[(mod_df.type == 'area') & (mod_df.mat == 'corr')],x='mod',hue='epoch',stat='probability',common_norm=True,multiple='layer',kde=True,ax=ax,palette=cmap,legend=True)
    # ax.set_title('Modularity calculated on\n communities anatomically defined')
    # # ax.legend()

    # ylim = ax.get_ylim()
    # for kk, cb in enumerate(combinations(label_list,2)):
    #     x = mod_df[(mod_df.type == 'area') & (mod_df.mat == 'corr') & (mod_df.epoch == cb[0])]['mod'].values
    #     y = mod_df[(mod_df.type == 'area') & (mod_df.mat == 'corr') & (mod_df.epoch == cb[1])]['mod'].values
    #     if (len(x) == 0) | (len(y) == 0):
    #         continue
    #     res = pg.ttest(x,y)
    #     pval = res['p-val'].values[0]
    #     if pval < 0.05/3:
    #         ax.text(np.round(np.nanpercentile(mod_corr[:,1],50),2),(ylim[1]/10)*(kk+1),f'{cb[0]} - {cb[1]} *')

    # plt.savefig(join(PlotDir,f'modularity_corr-mat_distri_{mID}_{rec_name}.png'),facecolor='white',dpi=300,bbox_inches='tight')
    # pdfdoc.savefig(fig)
