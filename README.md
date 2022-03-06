# Med263-Group4
# Introduction
  Genetic mutations are thought to be the main cause of cancer.  These functional changes in protein products result in cancers that may have radically different 
  behaviors in terms of disease progression and therefore treatment options. Genes that are known to be mutated in breast cancers include BRCA1 and 2, TP53, PIK3CA, HER2. 
  However, there are hundreds of other genes whose role in breast cancer is yet to be fully understood, 
  including those which influence the expression levels of a gene which may otherwise be normal. 
  RNA Seq is a next-generation sequencing (NGS) tool which allows for the quantitative measure of gene expression. 
  This coupled with gene set enrichment analysis can allow for physicians and researchers to better understand the complexities within patients’ 
  cancer and lead to more effective therapies.
    
  In this exercise, we will be looking at a breast cancer data set from TCGA (BRCA cohort) consisting of 1097 different patients that have had their gene expression
  quantified with RNAseq and their somatic mutation profile assessed by NGS.  We will be performing dimensionality reduction via non-negative matrix factorization (NMF) 
  in order to reduce the complex TCGA dataset to two |W| x |H|, for genes and patient ID, respectively. Patients were then reorganized into a z-normalized, 
  hierarchically-clustered heatmap, with 11 clusters identified with 1 to 465 patients per group. Kaplan-Meier curves comparing survivability of various clusters. 
  was then performed. We will be using the patients’ RNAseq expression as a readout for cancer type to cluster them into functional groups.  
 The resulting groups of this unsupervised clustering will then be annotated using ssGSEA (single sample GSEA) in order to assign biological meaning 
  to the different groups of samples.  We will also be examining which specific mutations are more associated with certain groups to determine the “root cause” 
  of the observed cancer expression pattern. 

  
# STEP 0: Download Software and Data
### Pandas
  [Pandas](https://pandas.pydata.org/) is a data analytics tool built on python which we will use to import, visualize and clean our data. 
### Seaborn 
 [Seaborn](https://seaborn.pydata.org/) is what we will use to visualive our data. 
### Lifelines
  [Lifelines](https://github.com/CamDavidsonPilon/lifelines/) is a survival analysis library used to create Kaplan-Meier survival plots. 
### SciPy
### sklearn

```
import pandas as pd
import seaborn as sns
import lifelines

from scipy.spatial import distance
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import dendrogram, linkage,fcluster,cut_tree,set_link_color_palette
import sklearn.decomposition
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from matplotlib import cm,colors
import itertools

from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import sys
```
  Import Data from [The Cancer Genome Atlas](https://portal.gdc.cancer.gov/projects/TCGA-BRCA).
  ```
  Insert download data code here.
  ```
  # Step 1: Data Cleanup
  
  
  First we will want to import the gene expression for each patient as a pandas dataframe.
  ```
expression_df = pd.read_table("./data/TCGA_BRCA_EXP.v1.gct",index_col=0,skiprows=2)
expression_df = expression_df[[c for c in expression_df.columns if c !="Description"]]
expression_df = expression_df.rename(columns={c:c.replace("_","-") for c in expression_df.columns})
expression_df.head()
```

Next, we can import the patient's clininically relevant data into a seperate pandas table
```
clinical_info_df = pd.read_csv("./data/TCGA_BRCA_clinical_FH.csv",index_col=0)
clinical_info_df.head()
```

Since we are interested in understanding the survival of patients based on their RNA Seq data, we can focus on deceased patients.
```
df = clinical_info_df[["days_to_last_followup",'days_to_death', 'vital_status']] 
df.loc[df["vital_status"]=="Dead"].head()
```

Additionally, we will want to remove patients whose vital status is 'null'.
```
df["days_to_last_followup"] = pd.to_numeric(df["days_to_last_followup"],errors="coerce")
df["days_to_death"] = pd.to_numeric(df["days_to_death"],errors="coerce")

vital_status_df_dict = {"sample":[],"event_time":[],"vital_status":[]}
for index,row in df.iterrows():
    timepoint = 0
    if row["vital_status"] =="Alive":
        if row.isnull()["days_to_last_followup"]==False:
            vital_status_df_dict["sample"].append(index)
            vital_status_df_dict["event_time"].append(row["days_to_last_followup"])
            vital_status_df_dict["vital_status"].append(row["vital_status"])
        elif row.isnull()["days_to_death"]==False:
            vital_status_df_dict["sample"].append(index)
            vital_status_df_dict["event_time"].append(row["days_to_death"])
            vital_status_df_dict["vital_status"].append(row["vital_status"])
    elif row["vital_status"] =="Dead":
        if row.isnull()["days_to_death"]==False:
            vital_status_df_dict["sample"].append(index)
            vital_status_df_dict["event_time"].append(row["days_to_death"])
            vital_status_df_dict["vital_status"].append(row["vital_status"])
        elif row.isnull()["days_to_last_followup"]==False:
            vital_status_df_dict["sample"].append(index)
            vital_status_df_dict["event_time"].append(row["days_to_last_followup"])
            vital_status_df_dict["vital_status"].append(row["vital_status"])
vital_status_df = pd.DataFrame(vital_status_df_dict).set_index("sample")
vital_status_df
```


  We can also have a look at the patients who are deceased,
  ```
  vital_status_df.loc[vital_status_df["vital_status"]=="Dead"]
  ```
  
  and we can identify what the endpoint time of our study will be. 
  ```
  df["days_to_last_followup"].iloc[0]
  ```
  
# Step 2: Dimensionality Reduction
  Dimensionality Reduction is a means of transforming highly dimensional data (like TCGA data) to a lower dimension data set, or matrix, 
  that can more easily be analized. The tool we will be using is Non-Negative Matrix Factorization (NMF) which will transform our original dataset |V| 
  into two,|W| and |H|, such that **W x H = V**. We will also z-normalize the outputted matrices so that we can more easily interpret and visualize the data.
  
  This code creates the NMF decomposition and z-normalize functions.
  ```
def NMF_decomposition(data,n_comp,max_iter=1000):
    model = sklearn.decomposition.NMF(n_components = n_comp,
                                      init = 'nndsvdar',
                                      solver = 'cd',
                                      max_iter = max_iter,
                                      tol = 1e-10,
                                      random_state = 12345, shuffle = False, verbose = False)
    w = pd.DataFrame(model.fit_transform(data + 0.001), index = data.index, columns = ['F{}'.format(x) for x in range(n_comp)])
    h = pd.DataFrame(model.components_, index = ['F{}'.format(x) for x in range(n_comp)], columns = data.columns)
    return w,h

def z_normalize_group(exp_df_in,do_clip = False,do_shift = False,do_rank = False):
    exp_df_in_norm = exp_df_in.copy()
    exp_df_in_norm = exp_df_in_norm[~exp_df_in_norm.index.duplicated(keep='first')]
    if do_rank==True:
        exp_df_in_norm = exp_df_in_norm.rank(axis=0, method='average', numeric_only=None, na_option='keep', ascending=True, pct=False)
    exp_df_in_means = exp_df_in_norm.mean(axis=1)
    exp_df_in_stds = exp_df_in_norm.std(axis=1)
    for i in exp_df_in_norm.index:
        #print(exp_df_in_norm.loc[i,:])
        exp_df_in_norm.loc[i,:] = (exp_df_in_norm.loc[i,:] - exp_df_in_means.loc[i])/exp_df_in_stds.loc[i]
    if do_clip==True:
        exp_df_in_norm.clip(-2, 2, inplace=True) 
    if do_shift==True:
        exp_df_in_norm = exp_df_in_norm + 2
    return exp_df_in_norm
```

Here we'll normalize and perform dimensionality reduction using the functions created above. 
```
normalized_expression_df = expression_df.rank(axis=0, method='dense', numeric_only=None, na_option='keep', 
                           ascending=True, pct=False)
W_df,H_df = NMF_decomposition(normalized_expression_df,10,max_iter=1000)
```

We can plot heatmaps of the resulting **W** and **H** using seaborn where **W** contains patient sample ID, and **H** contains the gene expression data.
```
fig, ax = plt.subplots(figsize=(15,12))
sns.heatmap(z_normalize_group(H_df),cmap="bwr",vmin=-2,vmax=2,center=0)
plt.show()
```
```
fig, ax = plt.subplots(figsize=(15,12))
sns.heatmap(z_normalize_group(W_df),cmap="bwr",vmin=-2,vmax=2,center=0)
plt.show()
```

# Step 3: Cluster

```
W_row_linkage_obj = linkage(distance.pdist(W_df), method='average')
W_col_linkage_obj = linkage(distance.pdist(W_df.T), method='average')

H_row_linkage_obj = linkage(distance.pdist(H_df), method='average')
H_col_linkage_obj = linkage(distance.pdist(H_df.T), method='average')
```

Cluster of genes (**W**):
```
sns.clustermap(W_df,
               row_linkage=W_row_linkage_obj,
               col_linkage = W_col_linkage_obj,
               figsize=(15,20),
               #square=True,
               z_score=0,
               center=0,
               vmin=-2,
               vmax=2,
               cmap="bwr"
              )
```

Cluster of Patients (**H**):
```
sns.clustermap(H_df,
               row_linkage=H_row_linkage_obj,
               col_linkage = H_col_linkage_obj,
               figsize=(15,15),
               #square=True,
               z_score=0,
               center=0,
               vmin=-2,
               vmax=2,
               cmap="bwr"
              )
```

Now that we have clusters, it is always nice to visualize them using color:
```
colormap_hex = []
colormap_obj = cm.get_cmap('Paired')
for i in range(0,colormap_obj.N):
    colormap_hex.append(colors.rgb2hex(colormap_obj(i)))
colormap_hex
```
```
height_threshold = 100
plt.figure(figsize=(20, 12))
set_link_color_palette(colormap_hex)


#f, (ax1, ax2) = plt.subplots(2,1,sharex=True,figsize=(15,12), constrained_layout=True)
dendrogram_dict = dendrogram(H_col_linkage_obj,
                             orientation='top',
                             #labels=cluster_assignments_series.index,
                             distance_sort='descending',
                             color_threshold = height_threshold,
                             #link_color_func={i:color_palette_lst[i] for i in range(0,len(color_palette_lst))},
                             show_leaf_counts= False,
                             #ax=ax1
                            )
#ax2.imshow([cluster_assignments_series.iloc[dendrogram_dict["leaves"]].values]*100, cmap='Set2', interpolation='nearest')
plt.show()
```

# *** *NOT SURE IF THIS IS NEEDED SINCE IT OUTPUTS AN ERROR MESSAGE* ***
```
pd.DataFrame({"color":dendrogram_dict['color_list'],"cluster":cluter_assignments_arr})
```

Now that we have nice clusters, we can create a dataframe, assigning each patient to its cluster:
```
cluter_assignments_arr = cut_tree(H_col_linkage_obj,height=height_threshold).flatten()[dendrogram_dict["leaves"]]
plt.figure(figsize=(20, 12))
sns.scatterplot(x=range(0,len(cluter_assignments_arr)),y=cluter_assignments_arr)
plt.xlabel("Sample Index")
plt.ylabel("Cluster Number")
plt.show()
```

Make sure the dataframe is good:
```


cluster_assignments_dict = {"sample":pd.Series(H_df.columns).iloc[dendrogram_dict["leaves"]].values,
                            "cluster":cluter_assignments_arr}
cluster_assignments_series = pd.DataFrame(cluster_assignments_dict).set_index("sample")["cluster"]
cluster_assignments_series = cluster_assignments_series
cluster_assignments_series.head()
```

How many patients are in each cluster?
```
cluster_assignments_series.value_counts()
```

# Step 3: Survival Analysis
*Insert background on Kaplan-Meier plots*


```
def KM_plot(cluster_assignments_series,
            vital_status_df,
            colormap_lst = None,
            title_txt = "KM Plot",
            vital_status_alive_txt = "Alive",
            vital_status_column = "vital_status",
            vital_status_time_column = 'event_time',
            fig_size = (20,15),):
    ax = None
    sample_group_medians_dict = {"sample_group":[],"median_survival":[]}
    sample_group_kmf_event_data_dict = dict()
    colormap_to_use = None
    
    for sample_group_name in np.sort(cluster_assignments_series.unique()): #sample_group_dict:
        samples_in_group_lst = cluster_assignments_series.loc[cluster_assignments_series==sample_group_name].index
        sample_group_count = len(samples_in_group_lst)
        time_data = vital_status_df.loc[samples_in_group_lst][vital_status_time_column].values
        event_data = vital_status_df.loc[samples_in_group_lst][vital_status_column].values
        event_data = np.where(event_data == vital_status_alive_txt, 0, 1)
        sample_group_kmf_event_data_dict[sample_group_name] = {"time_data":time_data,"event_data":event_data}
        
        kmf = KaplanMeierFitter()
        kmf.fit(time_data, event_data, label="{} (n={})".format(sample_group_name,sample_group_count))
        sample_group_medians_dict["sample_group"].append(sample_group_name)
        sample_group_medians_dict["median_survival"].append(kmf.median_survival_time_)
        color_to_use = None
        if colormap_lst is not None:
            color_to_use = colormap_lst[sample_group_name]
        if ax==None:
            ax = kmf.plot(show_censors=True, ci_show=False,figsize=fig_size,title=title_txt,color=color_to_use)
        else:
            ax = kmf.plot(show_censors=True, ci_show=False, ax=ax,color=color_to_use)

    sample_group_medians_df = pd.DataFrame(sample_group_medians_dict).set_index("sample_group")
    logrank_test_df_dict = {"cluster_A":[],
                              "cluster_A_median_survival":[],
                              "cluster_B":[],
                              "cluster_B_median_survival":[],
                              "p-value":[]} 
    for cluster_pair in itertools.combinations(sample_group_medians_df.index,2):
        logrank_test_df_dict["cluster_A"].append(cluster_pair[0])
        logrank_test_df_dict["cluster_A_median_survival"].append(sample_group_medians_df.loc[cluster_pair[0]]["median_survival"])
        logrank_test_df_dict["cluster_B"].append(cluster_pair[1])
        logrank_test_df_dict["cluster_B_median_survival"].append(sample_group_medians_df.loc[cluster_pair[1]]["median_survival"])
        logrank_test_result = logrank_test(sample_group_kmf_event_data_dict[cluster_pair[0]]["time_data"],
                                   sample_group_kmf_event_data_dict[cluster_pair[1]]["time_data"], 
                                   event_observed_A=sample_group_kmf_event_data_dict[cluster_pair[0]]["event_data"], 
                                   event_observed_B=sample_group_kmf_event_data_dict[cluster_pair[1]]["event_data"])
        logrank_test_df_dict["p-value"].append(logrank_test_result.p_value)
        ax.set_ylim(0,1.05)
        ax.set_xlim(0,)
    return ax, pd.DataFrame(logrank_test_df_dict).sort_values(by="p-value",ascending=True), sample_group_medians_df
```
    
Visualize:
```
    colormap_cluster_index_arr = np.array(dendrogram_dict['color_list'])[np.unique(cluter_assignments_arr, return_index=True)[1]]
KM_plot_ax, KM_stats_df, KM_medians_df = KM_plot(cluster_assignments_series,vital_status_df,colormap_lst=colormap_cluster_index_arr)
```
Plot Stats:
```
KM_medians_df
```

```
KM_stats_df
```





















# References
  1. F. Blows et al. Subtyping of Breast Cancer by Immunohistochemistry to Investigate a Relationship between Subtype and Short and Long Term Survival: A Collaborative Analysis of Data for 10,159 Cases from 12 Studies. PLOS Medicine. 2010.  https://doi.org/10.1371/journal.pmed.1000279
  2. https://portal.gdc.cancer.gov/projects/TCGA-BRCA 
  3. Subramanian A, Tamayo P, et.al. Gene set enrichment analysis: A knowledge-based approach for interpreting genome-wide expression profiles. PNAS. 
  4. Pereira, B., Chin, SF., Rueda, O. et al. The somatic mutation profiles of 2,433 breast cancers refine their genomic and transcriptomic landscapes. Nat Commun 7, 11479 (2016). https://doi.org/10.1038/ncomms11479