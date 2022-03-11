# Med263-Group4
# Introduction
  Genetic mutations are thought to be the main cause of cancer.  These functional changes in protein products result in cancers that may have radically different 
  behaviors in terms of disease progression and therefore treatment options. Genes that are known to be mutated in breast cancers include BRCA1 and 2, TP53, PIK3CA, HER2.<sup>4</sup> Traditionally, breast cancers are categorized into one of four types: Luminal A, Luminal B, HER2E, and Basal (triple negative). These categories are based on the histological appearance, immunostaining, and sometimes mutation profiles of well characterized genes, like those listed above. <sup>1</sup>
  However, there are hundreds of other genes whose role in breast cancer is yet to be fully understood, 
  including those which influence the expression levels of a gene which may otherwise be normal. 
  RNA Seq is a next-generation sequencing (NGS) tool which allows for the quantitative measure of gene expression. 
  This coupled with gene set enrichment analysis can allow for physicians and researchers to better understand the complexities within patients’ 
  cancer and lead to more effective therapies.
    
  In this exercise, we will be looking at a breast cancer data set from TCGA (BRCA cohort) consisting of 1097 different patients that have had their gene expression
  quantified with RNAseq and their somatic mutation profile assessed by NGS.<sup>2</sup>  We will be performing dimensionality reduction via non-negative matrix factorization (NMF) 
  in order to reduce the complex TCGA dataset to two |W| x |H|, for genes and patient ID, respectively. Patients were then reorganized into a z-normalized, 
  hierarchically-clustered heatmap, with 11 clusters identified with 1 to 465 patients per group. Kaplan-Meier curves comparing survivability of various clusters. 
  was then performed. We will be using the patients’ RNAseq expression as a readout for cancer type to cluster them into functional groups.  
 The resulting groups of this unsupervised clustering will then be annotated using ssGSEA (single sample GSEA) in order to assign biological meaning 
 to the different groups of samples.<sup>3</sup>  We will also be examining which specific mutations are more associated with certain groups to determine the “root cause” 
  of the observed cancer expression pattern. 
  
# Biological/clinical interpretation of analysis results
Several of the clusters formed during the analysis correlate highly, through ssGSEA-based annotation, with the four traditional means of categorizing breast tumors (Luminal A, Luminal B, Basal, HER2+).  This project would allow researchers to subset the traditional categories in order to identify additional treatment targets which could be used in conjunction with available treatments, and also allow for a more accurate prognosis assessment. 

# Mathematical/statistical meaning of analysis
Clusters were biologically annotated using one-vs-all (cluster of interest vs samples not in the cluster) comparisons of their ssGSEA scores, statistically qantified using the Mann-Whitney test.  In clusters 1 through 5 in the tutorial, the gene set corresponding with the breast cancer subtype (Luminal A, Luminal B, HER2E, and Basal \[triple negative\]) was the one with the lowest p-value for the cluster, even after multiple intra-group multiple hypothesis correction.  Kaplan-Meier plots were also generated to examine differences in survival over time in the different clusters.  In general, known trends such as triple negative cancers having a lower median survival time compared to other subtypes were captured.<sup>1</sup>  Pairwise differences between the survival curves of the different group were statistically quantified with the Cox Log-Rank test.  After multiple hypothesis correction, ```Cluster 2: Luminal A``` and ```Cluster 5: ERBB2-driven``` were the only pair of meaningful clusters with a statistically significant difference (p = 0.046) in survival, meaning they had a reasonable amount of patients (Cluster 9 only had 1 patient and was likely due to noise)  In multiple hypothesis testing, p-values were corrected using the Benjamini-Hochberg False Discovery Rate (FDR) procedure as implemented in the ```fdrcorrection``` function of the ```statsmodel``` package.

# Common pitfalls and how to avoid them 
1. Be sure to add the clinical and gene expression data files into your data directory:

    Clinical Data: https://docs.google.com/spreadsheets/d/1dpBjMe0RNiGxcJWYNHcMOBDDSTGDzQJmNq8C_4Bd_4E/edit?usp=sharing
    
    Expression Data: https://drive.google.com/file/d/1MU4dM7mpBTy933Nx5jNAzVZ8y1EaZ7T0/view?usp=sharing
    
    Gene Sets (Already in this repository in the data directory): https://drive.google.com/file/d/1-BA3hxGLmQhFs77b8Hno9N4_FuEUXLIW/view?usp=sharing

2. While Clustering, be sure to not set the height threshold too low since this will generate many clusters due to all the dendrogram lines it will intersect with.  The code will take an incredibly long time to run, may crash some computers, and may not return useful clusters since there will be fewer samples per cluster. To avoid this, you can start with a higher threshold and decrease until you are satisfied with the results. 
 
3. If you are working on a separate problem using NMF_decomposition, setting a random state is important for reproducibility. 

4. Make sure you take time to explore the data you are given since it might not be in a form that is suitable for analysis, or that your code can use.  See **Step 1: Data Cleanup**.
5. Always make sure you do multiple hypothesis corrections when testing multiple hypotheses because if everything is "significant", then nothing is.  

  
# STEP 0: Download Software and Data
This tutorial uses Python 3 and the Jupyter notebook environment.  You may already have these on your system.

Install Python 3: https://www.python.org/downloads/

You may have to install pip separately on some systems.

Then install Jupyter Notebook: https://jupyter.org/install

Test if your install was successful by opening a terminal and running one of these:

```python3 -m notebook```

```py -3.X -m notebook``` 

X = your version of python, for sample Python X = 7 for Python 3.7

```jupyter notebook```


## Python Packages:
### Pandas
  [Pandas](https://pandas.pydata.org/) is a data analytics tool built on python which we will use to import, visualize and clean our data. 
### NumPy
  [NumPy](https://scipy.org/install/) is a mathematical library optimized for very fast calculations.
### SciPy
  [SciPy](https://scipy.org/install/) is a scientific computing library that we use for hierarchical clustering.  
### MatplotLib
  [MatplotLib](https://matplotlib.org/stable/users/installing/index.html) is a library for plotting graphs and other basic visualization functionality. 
### Seaborn 
 [Seaborn](https://seaborn.pydata.org/) is what we will use to visualize our data. 
### Lifelines
  [Lifelines](https://github.com/CamDavidsonPilon/lifelines/) is a survival analysis library used to create Kaplan-Meier survival plots. 

  
### sklearn
  [ScikitLearn](https://scikit-learn.org/stable/install.html) is a machine learning library.  Here we use it to perform Non-Negative Matrix Factorization (NMF).
### StatsModels
  [statsmodels](https://www.statsmodels.org/dev/install.html) is a statistical package for doing a variety of data analysis and statistics.  Here it is used for False Discovery Rate (FDR) p-value correction.  
  
All of the above packages can be installed with pip, anaconda, or whatever other package/environment manager you prefer.  You may already have some or all of these installed.  Pip commands are given here:
```
pip install -U pandas numpy scipy matplotlib seaborn lifelines scikit-learn statsmodels
```
Add this cell to a jupyter notebook and run it:
```python
import pandas as pd
import seaborn as sns

from scipy.spatial import distance
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import dendrogram, linkage,fcluster,cut_tree,set_link_color_palette
from scipy.stats import mannwhitneyu,norm,sem

import sklearn.decomposition
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from matplotlib import cm,colors
import itertools

from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import sys
from ssGSEA import single_sample_gseas

from statsmodels.stats.multitest import fdrcorrection
import numpy as np

```
If the above import statements do not result in an error, then your installs were successful.  This was also the first piece of code needed for the tutorial.

## Data:
Data for this tutorial comes from the Breast Cancer (BRCA) Cohort of [The Cancer Genome Atlas](https://portal.gdc.cancer.gov/projects/TCGA-BRCA).

Clinical Data: https://docs.google.com/spreadsheets/d/1dpBjMe0RNiGxcJWYNHcMOBDDSTGDzQJmNq8C_4Bd_4E/edit?usp=sharing

Expression Data: https://drive.google.com/file/d/1MU4dM7mpBTy933Nx5jNAzVZ8y1EaZ7T0/view?usp=sharing

Gene Sets (Already in this repository in the [data directory](data/MSigDB_breast_cancer_subtypes_gene_sets.gmt)): https://drive.google.com/file/d/1-BA3hxGLmQhFs77b8Hno9N4_FuEUXLIW/view?usp=sharing

Place all downloaded data into the [data directory](data/).  

# Step 1: Data Cleanup
  
  
First we will want to import the gene expression for each patient as a pandas dataframe.
```python
expression_df = pd.read_table("./data/TCGA_BRCA_EXP.v1.gct",index_col=0,skiprows=2)
expression_df = expression_df[[c for c in expression_df.columns if c !="Description"]]
expression_df = expression_df.rename(columns={c:c.replace("_","-") for c in expression_df.columns})
expression_df.head()
```

Next, we can import the patient's clininically relevant data into a seperate pandas table
```python
clinical_info_df = pd.read_csv("./data/TCGA_BRCA_clinical_FH.csv",index_col=0)
clinical_info_df.head()
```

TCGA data was collected from many different institutions ("sites") nationwide, and each has slightly different ways to record clinical data.  This results in a very "scattered" data table.  For example, you can see here that the time of event stored in two different columns depending on whether the patient is dead or alive.  There are also patients who have timepoint values recorded in both, or the wrong column, so you cannot just "pick one" or else you will lose info.
```python
df = clinical_info_df[["days_to_last_followup",'days_to_death', 'vital_status']] 
df.loc[df["vital_status"]=="Dead"].head()
```

This code below "regularizes" the timepoint data into one column by choosing the approrpiate column depending on the patient's vital status (Dead/Alive).  

```python
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


The data is now cleaned, even for dead patients:
```python
vital_status_df.loc[vital_status_df["vital_status"]=="Dead"]
```
  
# Step 2: Dimensionality Reduction
  Dimensionality Reduction is a means of transforming highly dimensional data (like TCGA data) to a lower dimension data set, or matrix, 
  that can more easily be analyzed. The method we will be using is Non-Negative Matrix Factorization (NMF) which will transform/approximate our original dataset |V| 
  into two matrices,|W| and |H|, such that **W x H = V** plus an error matrix that is not used. We will also z-normalize the outputted matrices so that we can more easily interpret and visualize the data.
  
  This code creates the NMF decomposition and z-normalize functions.
```python
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
```python
normalized_expression_df = expression_df.rank(axis=0, method='dense', numeric_only=None, na_option='keep', 
                           ascending=True, pct=False)
W_df,H_df = NMF_decomposition(normalized_expression_df,10,max_iter=1000)
```

We can plot heatmaps of the resulting **W** and **H** using seaborn where **W** contains patient sample ID, and **H** contains the gene expression data.
```python
fig, ax = plt.subplots(figsize=(15,12))
sns.heatmap(z_normalize_group(H_df),cmap="bwr",vmin=-2,vmax=2,center=0)
plt.show()
```

```python
fig, ax = plt.subplots(figsize=(15,12))
sns.heatmap(z_normalize_group(W_df),cmap="bwr",vmin=-2,vmax=2,center=0)
plt.show()
```



# Step 3: Cluster

```python
W_row_linkage_obj = linkage(distance.pdist(W_df), method='average')
W_col_linkage_obj = linkage(distance.pdist(W_df.T), method='average')

H_row_linkage_obj = linkage(distance.pdist(H_df), method='average')
H_col_linkage_obj = linkage(distance.pdist(H_df.T), method='average')
```

Creates a hierarchically clustered heatmap of the genes dataset (**W**) *Red is indicative of genese which are up regulated and blue are genes which are down regulated within a latent category (f0-f9)*:
```python
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

Creates a hierarchically clustered heatmap of the Clinical dataset (**H**) *Here the colors indicate how much each patient correlates with a specific latent category (f0-f9), with bed being correlated and blue being uncorrelation*:
```python
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

Now we can assign each clinical datapoint to a cluster based on the dendrogram for further analysis. This step is necessary to make sure future plots of multiple clusters have the same color/cluster assignments as the cluster dendrogram.
```python
colormap_hex = []
colormap_obj = cm.get_cmap('Paired')
for i in range(0,colormap_obj.N):
    colormap_hex.append(colors.rgb2hex(colormap_obj(i)))
colormap_hex
```
```python
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

Now that we have nice clusters, we can create a dataframe, assigning each patient to its cluster:
```python
cluter_assignments_arr = cut_tree(H_col_linkage_obj,height=height_threshold).flatten()[dendrogram_dict["leaves"]]
plt.figure(figsize=(20, 12))
sns.scatterplot(x=range(0,len(cluter_assignments_arr)),y=cluter_assignments_arr)
plt.xlabel("Sample Index")
plt.ylabel("Cluster Number")
plt.show()
```

Make sure the dataframe is good:
```python
cluster_assignments_dict = {"sample":pd.Series(H_df.columns).iloc[dendrogram_dict["leaves"]].values,
                            "cluster":cluter_assignments_arr}
cluster_assignments_series = pd.DataFrame(cluster_assignments_dict).set_index("sample")["cluster"]
cluster_assignments_series = cluster_assignments_series
cluster_assignments_series.head()
```

How many patients are in each cluster?
```python
cluster_assignments_series.value_counts()
```


# Step 4: Annotate Clusters of Patients

```python
from pandas import DataFrame


def read_gmt(gmt_file_path, drop_description=True):

    lines = []

    with open(gmt_file_path) as gmt_file:

        for line in gmt_file:

            split = line.strip().split(sep="\t")

            lines.append(split[:2] + [gene for gene in set(split[2:]) if gene])

    df = DataFrame(lines)

    df.set_index(0, inplace=True)

    df.index.name = "Gene Set"

    if drop_description:

        df.drop(1, axis=1, inplace=True)

        df.columns = tuple("Gene {}".format(i) for i in range(0, df.shape[1]))

    else:

        df.columns = ("Description",) + tuple(
            "Gene {}".format(i) for i in range(0, df.shape[1] - 1)
        )

    return df

from pandas import concat



def read_gmts(gmt_file_paths, sets=None, drop_description=True, collapse=False):

    dfs = []

    for gmt_file_path in gmt_file_paths:

        dfs.append(read_gmt(gmt_file_path, drop_description=drop_description))

    df = concat(dfs, sort=True)

    if sets is not None:

        df = df.loc[(df.index & sets)].dropna(axis=1, how="all")

    if collapse:

        return df.unstack().dropna().sort_values().unique()

    else:

        return df
MSigDB_breast_cancer_subtypes_gene_sets_df = read_gmt("./data/MSigDB_breast_cancer_subtypes_gene_sets.gmt")
MSigDB_breast_cancer_subtypes_gene_sets_df
```

Perform single sample Gene Set Enrichment Analysis (ssGSEA):
```python
TCGA_breast_ssGSEA_df = single_sample_gseas(expression_df,
                                            MSigDB_breast_cancer_subtypes_gene_sets_df,
                                            n_job=4) #change n_job to whatever the number of cores you have on your computer
#TCGA_breast_ssGSEA_df.to_csv("./data/TCGA_breast_ssGSEA_scores.csv")
```

Compute statistics of ssGSEA scores per cluster (1 vs others):

```python
## 1 vs all mann whitneys
def mann_whitney_cluster_1_vs_others(cluster,cluster_assignment_series, data_df):
    cluster_samples = cluster_assignment_series.loc[cluster_assignment_series==cluster].index
    other_samples = cluster_assignment_series.loc[cluster_assignment_series!=cluster].index
    results_df_dict = {"gene_set":[],
                       "cluster_avg":[],
                       "cluster_95%_CI_lower":[],
                       "cluster_95%_CI_upper":[],
                       "others_avg":[],
                       "others_95%_CI_lower":[],
                       "others_95%_CI_upper":[],
                       "mann-whitney_p-value":[]}
    for index,row in data_df.iterrows():
        cluster_data = row[cluster_samples].dropna()
        other_data = row[other_samples].dropna()
        cluster_data_mean = cluster_data.mean()
        cluster_data_95_CI = norm.interval(alpha=0.95, loc=cluster_data_mean, scale=sem(cluster_data))
        other_data_mean = other_data.mean()
        other_data_95_CI = norm.interval(alpha=0.95, loc=other_data_mean, scale=sem(other_data))
        mann_whitney_result = mannwhitneyu(cluster_data,other_data)
        mann_whitney_pval =  mann_whitney_result.pvalue
        results_df_dict["gene_set"].append(index)
        results_df_dict["cluster_avg"].append(cluster_data_mean)
        results_df_dict["cluster_95%_CI_lower"].append(cluster_data_95_CI[0])
        results_df_dict["cluster_95%_CI_upper"].append(cluster_data_95_CI[1])
        results_df_dict["others_avg"].append(other_data_mean)
        results_df_dict["others_95%_CI_lower"].append(other_data_95_CI[0])
        results_df_dict["others_95%_CI_upper"].append(other_data_95_CI[1])
        results_df_dict["mann-whitney_p-value"].append(mann_whitney_pval)
    results_df = pd.DataFrame(results_df_dict).set_index("gene_set")
    results_df["group_FDR_corrected_p-value"] = fdrcorrection(results_df_dict["mann-whitney_p-value"])[1]
    results_df["group_avg_diff"] = results_df["cluster_avg"]-results_df["others_avg"]
    
    return results_df.sort_values(by="group_avg_diff",ascending=False)
```
Statistics on the ssGSEA score distributions for each cluster can be displayed.  Which gene sets, when ordered by the difference between the average group of interest and others rises to the top?  Do they cover the main described subtypes of breast cancer?  
```python
cluster_1_results_df = mann_whitney_cluster_1_vs_others(1,cluster_assignments_series, TCGA_breast_ssGSEA_df)
cluster_1_results_df
```

```python
cluster_2_results_df = mann_whitney_cluster_1_vs_others(2,cluster_assignments_series, TCGA_breast_ssGSEA_df)
cluster_2_results_df
```


```python
cluster_3_results_df = mann_whitney_cluster_1_vs_others(3,cluster_assignments_series, TCGA_breast_ssGSEA_df)
cluster_3_results_df
```


```python
cluster_4_results_df = mann_whitney_cluster_1_vs_others(4,cluster_assignments_series, TCGA_breast_ssGSEA_df)
cluster_4_results_df
```


```python
cluster_5_results_df = mann_whitney_cluster_1_vs_others(5,cluster_assignments_series, TCGA_breast_ssGSEA_df)
cluster_5_results_df
```
Fill in the ```cluster_relabel_dict``` below with your interpretation of what each cluster is:

```python
cluster_relabel_dict = {1:"Cluster 1: ",
                        2:"Cluster 2: ",
                        3:"Cluster 3: ",
                        4:"Cluster 4: ",
                        5:"Cluster 5: "}
#Don't worry about the other clusters.  
for cluster_number in cluster_assignments_series.unique():
    if cluster_number not in cluster_relabel_dict:
        cluster_relabel_dict[cluster_number] = "Cluster {}".format(cluster_number)
cluster_relabel_dict
```

```python
cluster_assignments_series = cluster_assignments_series.replace(to_replace=cluster_relabel_dict)

```
# Step 5: Survival Analysis
Kaplan-Meier Survival Analysis is a simple tool which incorporates successive probabilities of an event to calculate the overall probability of an event occurring, accounting for right-censored data points due to loss of followup, study ending, etc. 
In this section, we will be using the lifelines KaplanMeierFitter function to calculate and graph the Kaplan-Meier Curve to compare the survival probabilities over time between all the different clusters of patients we created. 


```python
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
    logrank_test_df = pd.DataFrame(logrank_test_df_dict)
    logrank_test_df["FDR_corrected_p-value"] = fdrcorrection(logrank_test_df["p-value"])[1]
    return ax, pd.DataFrame(logrank_test_df_dict).sort_values(by="FDR_corrected_p-value",ascending=True), sample_group_medians_df
```
    
Visualize:
```python
colormap_cluster_index_arr = np.array(dendrogram_dict['color_list'])[np.unique(cluter_assignments_arr, return_index=True)[1]]
colormap_cluster_index_dict = {cluster_relabel_dict[k]:colormap_cluster_index_arr[k] for k in range(0,len(colormap_cluster_index_arr))}

KM_plot_ax, KM_stats_df, KM_medians_df = KM_plot(cluster_assignments_series,vital_status_df,colormap_lst=colormap_cluster_index_dict)
```
Plot Stats:
```python
KM_medians_df
```

```python
KM_stats_df
```

# References
  1. F. Blows et al. Subtyping of Breast Cancer by Immunohistochemistry to Investigate a Relationship between Subtype and Short and Long Term Survival: A Collaborative Analysis of Data for 10,159 Cases from 12 Studies. PLOS Medicine. 2010.  https://doi.org/10.1371/journal.pmed.1000279
  2. https://portal.gdc.cancer.gov/projects/TCGA-BRCA 
  3. Subramanian A, Tamayo P, et.al. Gene set enrichment analysis: A knowledge-based approach for interpreting genome-wide expression profiles. PNAS. 
  4. Pereira, B., Chin, SF., Rueda, O. et al. The somatic mutation profiles of 2,433 breast cancers refine their genomic and transcriptomic landscapes. Nat Commun 7, 11479 (2016). https://doi.org/10.1038/ncomms11479
