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
 [Seaborn](https://seaborn.pydata.org/) is what we will use to visualize our data. 
### Lifelines
  [Lifelines](https://github.com/CamDavidsonPilon/lifelines/) is a survival analysis library used to create Kaplan-Meier survival plots. 
### SciPy
  [SciPy] (https://scipy.org/install/) is a scientific computing library that we use for hierarchical clustering.  
### sklearn
  [ScikitLearn](https://scikit-learn.org/stable/install.html) is a machine learning library.  Here we use it to perform Non-Negative Matrix Factorization (NMF).
### StatsModels
  [statsmodels](https://www.statsmodels.org/dev/install.html) is a statistical package for doing a variety of data analysis and statistics.  Here it is used for False Discovery Rate (FDR) p-value correction.  

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


# Step 3: Annotate Clusters of Patients

```
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
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Gene 0</th>
      <th>Gene 1</th>
      <th>Gene 2</th>
      <th>Gene 3</th>
      <th>Gene 4</th>
      <th>Gene 5</th>
      <th>Gene 6</th>
      <th>Gene 7</th>
      <th>Gene 8</th>
      <th>Gene 9</th>
      <th>...</th>
      <th>Gene 637</th>
      <th>Gene 638</th>
      <th>Gene 639</th>
      <th>Gene 640</th>
      <th>Gene 641</th>
      <th>Gene 642</th>
      <th>Gene 643</th>
      <th>Gene 644</th>
      <th>Gene 645</th>
      <th>Gene 646</th>
    </tr>
    <tr>
      <th>Gene Set</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>VANTVEER_BREAST_CANCER_ESR1_UP</th>
      <td>ZNF587</td>
      <td>FBP1</td>
      <td>P4HTM</td>
      <td>CHAD</td>
      <td>IRS1</td>
      <td>FAM110C</td>
      <td>TMBIM6</td>
      <td>GLI3</td>
      <td>ELOVL5</td>
      <td>HHAT</td>
      <td>...</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>VANTVEER_BREAST_CANCER_METASTASIS_UP</th>
      <td>LYPD6</td>
      <td>FBP1</td>
      <td>WISP1</td>
      <td>ODZ3</td>
      <td>KIAA1217</td>
      <td>RBP3</td>
      <td>MYRIP</td>
      <td>MS4A7</td>
      <td>NEO1</td>
      <td>SCUBE2</td>
      <td>...</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>VANTVEER_BREAST_CANCER_POOR_PROGNOSIS</th>
      <td>PALM2-AKAP2</td>
      <td>WISP1</td>
      <td>ESM1</td>
      <td>PITRM1</td>
      <td>GMPS</td>
      <td>NUSAP1</td>
      <td>MS4A7</td>
      <td>CCNE2</td>
      <td>TSPYL5</td>
      <td>SCUBE2</td>
      <td>...</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>BIOCARTA_HER2_PATHWAY</th>
      <td>PIK3CA</td>
      <td>MAPK3</td>
      <td>PIK3CG</td>
      <td>CARM1</td>
      <td>MAP2K1</td>
      <td>ERBB4</td>
      <td>EP300</td>
      <td>STAT3</td>
      <td>EGFR</td>
      <td>IL6R</td>
      <td>...</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>CHARAFE_BREAST_CANCER_BASAL_VS_MESENCHYMAL_UP</th>
      <td>PKP3</td>
      <td>S100A14</td>
      <td>TLCD1</td>
      <td>LOC93622</td>
      <td>GSN</td>
      <td>CLCA2</td>
      <td>FXYD3</td>
      <td>KRT5</td>
      <td>RHOD</td>
      <td>FOSB</td>
      <td>...</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>CHARAFE_BREAST_CANCER_LUMINAL_VS_BASAL_UP</th>
      <td>CTXN1</td>
      <td>CLSTN2</td>
      <td>ETNK2</td>
      <td>ATXN7L3B</td>
      <td>RAB40C</td>
      <td>AVL9</td>
      <td>MAPT</td>
      <td>RALGAPA1</td>
      <td>SHANK2</td>
      <td>PPP2R2C</td>
      <td>...</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>SMID_BREAST_CANCER_BASAL_UP</th>
      <td>PDXK</td>
      <td>ARPC4</td>
      <td>E2F3</td>
      <td>CX3CL1</td>
      <td>KIAA1609</td>
      <td>ASPM</td>
      <td>ART3</td>
      <td>GPR37</td>
      <td>ACP1</td>
      <td>ARL4C</td>
      <td>...</td>
      <td>REG1A</td>
      <td>CSRP2</td>
      <td>KLF6</td>
      <td>GJC1</td>
      <td>ITGA6</td>
      <td>FOXM1</td>
      <td>CENPN</td>
      <td>NFE2L3</td>
      <td>CLIP4</td>
      <td>ING1</td>
    </tr>
    <tr>
      <th>SMID_BREAST_CANCER_ERBB2_UP</th>
      <td>ASS1</td>
      <td>ACE2</td>
      <td>LBP</td>
      <td>GCAT</td>
      <td>CEACAM6</td>
      <td>CRISP3</td>
      <td>FGG</td>
      <td>IGKV1D-13</td>
      <td>CLCA2</td>
      <td>AR</td>
      <td>...</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>SMID_BREAST_CANCER_LUMINAL_A_UP</th>
      <td>MYH11</td>
      <td>DUSP1</td>
      <td>LMOD1</td>
      <td>OGN</td>
      <td>FBLN1</td>
      <td>NKX3-1</td>
      <td>COL14A1</td>
      <td>TNN</td>
      <td>SVEP1</td>
      <td>ADH1B</td>
      <td>...</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>SMID_BREAST_CANCER_LUMINAL_B_UP</th>
      <td>FBP1</td>
      <td>KCNK15</td>
      <td>AGL</td>
      <td>RET</td>
      <td>CAMK2B</td>
      <td>SLC8A2</td>
      <td>ARNT2</td>
      <td>CLSTN2</td>
      <td>C17orf75</td>
      <td>TPPP3</td>
      <td>...</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>DOANE_BREAST_CANCER_ESR1_UP</th>
      <td>FBP1</td>
      <td>KCNK15</td>
      <td>RET</td>
      <td>AZGP1</td>
      <td>CHAD</td>
      <td>CLSTN2</td>
      <td>FLJ22184</td>
      <td>AR</td>
      <td>MAPT</td>
      <td>ALCAM</td>
      <td>...</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>GOZGIT_ESR1_TARGETS_UP</th>
      <td>KRT71</td>
      <td>OSBPL1A</td>
      <td>FAM13A</td>
      <td>CEBPA</td>
      <td>B4GALT4</td>
      <td>KRT14</td>
      <td>POU5F1P4</td>
      <td>RNF144A</td>
      <td>NAGA</td>
      <td>AGXT</td>
      <td>...</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
  </tbody>
</table>
<p>12 rows × 647 columns</p>
</div>

Perform single sample Gene Set Enrichment Analysis (ssGSEA):
```python
TCGA_breast_ssGSEA_df = single_sample_gseas(expression_df,
                                            MSigDB_breast_cancer_subtypes_gene_sets_df,
                                            n_job=4) #change n_job to whatever the number of cores you have on your computer
#TCGA_breast_ssGSEA_df.to_csv("./data/TCGA_breast_ssGSEA_scores.csv")
```

Compute statistics of ssGSEA scores per cluster (1 vs others):

```python
## 1 vs all mann whitneys + distribution charts
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


cluster_1_results_df = mann_whitney_cluster_1_vs_others(1,cluster_assignments_series, TCGA_breast_ssGSEA_df)
cluster_1_results_df
```
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cluster_avg</th>
      <th>cluster_95%_CI_lower</th>
      <th>cluster_95%_CI_upper</th>
      <th>others_avg</th>
      <th>others_95%_CI_lower</th>
      <th>others_95%_CI_upper</th>
      <th>mann-whitney_p-value</th>
      <th>group_FDR_corrected_p-value</th>
      <th>group_avg_diff</th>
    </tr>
    <tr>
      <th>gene_set</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>SMID_BREAST_CANCER_LUMINAL_B_UP</th>
      <td>0.776654</td>
      <td>0.770879</td>
      <td>0.782429</td>
      <td>0.695605</td>
      <td>0.688254</td>
      <td>0.702957</td>
      <td>3.176803e-30</td>
      <td>1.906082e-29</td>
      <td>0.081049</td>
    </tr>
    <tr>
      <th>VANTVEER_BREAST_CANCER_ESR1_UP</th>
      <td>0.789605</td>
      <td>0.785899</td>
      <td>0.793312</td>
      <td>0.735907</td>
      <td>0.730373</td>
      <td>0.741442</td>
      <td>6.833185e-20</td>
      <td>2.733274e-19</td>
      <td>0.053698</td>
    </tr>
    <tr>
      <th>DOANE_BREAST_CANCER_ESR1_UP</th>
      <td>0.819564</td>
      <td>0.815403</td>
      <td>0.823725</td>
      <td>0.769526</td>
      <td>0.762347</td>
      <td>0.776705</td>
      <td>2.267145e-07</td>
      <td>3.022860e-07</td>
      <td>0.050038</td>
    </tr>
    <tr>
      <th>CHARAFE_BREAST_CANCER_LUMINAL_VS_BASAL_UP</th>
      <td>0.730741</td>
      <td>0.727899</td>
      <td>0.733582</td>
      <td>0.690788</td>
      <td>0.687520</td>
      <td>0.694056</td>
      <td>2.648443e-35</td>
      <td>3.178131e-34</td>
      <td>0.039952</td>
    </tr>
    <tr>
      <th>SMID_BREAST_CANCER_LUMINAL_A_UP</th>
      <td>0.727970</td>
      <td>0.719077</td>
      <td>0.736863</td>
      <td>0.689214</td>
      <td>0.681971</td>
      <td>0.696457</td>
      <td>7.005126e-05</td>
      <td>7.641956e-05</td>
      <td>0.038756</td>
    </tr>
    <tr>
      <th>VANTVEER_BREAST_CANCER_METASTASIS_UP</th>
      <td>0.790000</td>
      <td>0.786233</td>
      <td>0.793767</td>
      <td>0.769924</td>
      <td>0.766647</td>
      <td>0.773201</td>
      <td>4.032648e-08</td>
      <td>6.913111e-08</td>
      <td>0.020076</td>
    </tr>
    <tr>
      <th>BIOCARTA_HER2_PATHWAY</th>
      <td>0.785695</td>
      <td>0.781547</td>
      <td>0.789844</td>
      <td>0.768087</td>
      <td>0.765325</td>
      <td>0.770848</td>
      <td>1.608954e-10</td>
      <td>3.861490e-10</td>
      <td>0.017609</td>
    </tr>
    <tr>
      <th>SMID_BREAST_CANCER_ERBB2_UP</th>
      <td>0.778391</td>
      <td>0.773951</td>
      <td>0.782831</td>
      <td>0.764753</td>
      <td>0.761459</td>
      <td>0.768047</td>
      <td>8.741091e-08</td>
      <td>1.311164e-07</td>
      <td>0.013638</td>
    </tr>
    <tr>
      <th>VANTVEER_BREAST_CANCER_POOR_PROGNOSIS</th>
      <td>0.691500</td>
      <td>0.683094</td>
      <td>0.699906</td>
      <td>0.679593</td>
      <td>0.674515</td>
      <td>0.684672</td>
      <td>3.020176e-03</td>
      <td>3.020176e-03</td>
      <td>0.011907</td>
    </tr>
    <tr>
      <th>CHARAFE_BREAST_CANCER_BASAL_VS_MESENCHYMAL_UP</th>
      <td>0.721684</td>
      <td>0.717685</td>
      <td>0.725684</td>
      <td>0.731282</td>
      <td>0.728601</td>
      <td>0.733962</td>
      <td>2.694420e-06</td>
      <td>3.233305e-06</td>
      <td>-0.009597</td>
    </tr>
    <tr>
      <th>GOZGIT_ESR1_TARGETS_UP</th>
      <td>0.581263</td>
      <td>0.575972</td>
      <td>0.586554</td>
      <td>0.605898</td>
      <td>0.603013</td>
      <td>0.608783</td>
      <td>3.053384e-16</td>
      <td>9.160151e-16</td>
      <td>-0.024635</td>
    </tr>
    <tr>
      <th>SMID_BREAST_CANCER_BASAL_UP</th>
      <td>0.621474</td>
      <td>0.617563</td>
      <td>0.625385</td>
      <td>0.649769</td>
      <td>0.645716</td>
      <td>0.653822</td>
      <td>2.182602e-09</td>
      <td>4.365204e-09</td>
      <td>-0.028295</td>
    </tr>
  </tbody>
</table>
</div>




```python
cluster_2_results_df = mann_whitney_cluster_1_vs_others(2,cluster_assignments_series, TCGA_breast_ssGSEA_df)
cluster_2_results_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cluster_avg</th>
      <th>cluster_95%_CI_lower</th>
      <th>cluster_95%_CI_upper</th>
      <th>others_avg</th>
      <th>others_95%_CI_lower</th>
      <th>others_95%_CI_upper</th>
      <th>mann-whitney_p-value</th>
      <th>group_FDR_corrected_p-value</th>
      <th>group_avg_diff</th>
    </tr>
    <tr>
      <th>gene_set</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>SMID_BREAST_CANCER_LUMINAL_A_UP</th>
      <td>0.747922</td>
      <td>0.742889</td>
      <td>0.752955</td>
      <td>0.662869</td>
      <td>0.654233</td>
      <td>0.671506</td>
      <td>5.735646e-42</td>
      <td>6.882776e-41</td>
      <td>0.085052</td>
    </tr>
    <tr>
      <th>DOANE_BREAST_CANCER_ESR1_UP</th>
      <td>0.820685</td>
      <td>0.816470</td>
      <td>0.824901</td>
      <td>0.753371</td>
      <td>0.744711</td>
      <td>0.762031</td>
      <td>3.521502e-21</td>
      <td>1.056451e-20</td>
      <td>0.067314</td>
    </tr>
    <tr>
      <th>SMID_BREAST_CANCER_LUMINAL_B_UP</th>
      <td>0.748766</td>
      <td>0.743168</td>
      <td>0.754364</td>
      <td>0.690970</td>
      <td>0.681709</td>
      <td>0.700231</td>
      <td>6.954724e-10</td>
      <td>1.043209e-09</td>
      <td>0.057796</td>
    </tr>
    <tr>
      <th>VANTVEER_BREAST_CANCER_ESR1_UP</th>
      <td>0.781803</td>
      <td>0.778615</td>
      <td>0.784990</td>
      <td>0.725099</td>
      <td>0.718264</td>
      <td>0.731935</td>
      <td>4.885957e-20</td>
      <td>1.172630e-19</td>
      <td>0.056704</td>
    </tr>
    <tr>
      <th>VANTVEER_BREAST_CANCER_POOR_PROGNOSIS</th>
      <td>0.698818</td>
      <td>0.691843</td>
      <td>0.705793</td>
      <td>0.670638</td>
      <td>0.665245</td>
      <td>0.676032</td>
      <td>6.317703e-11</td>
      <td>1.083035e-10</td>
      <td>0.028179</td>
    </tr>
    <tr>
      <th>BIOCARTA_HER2_PATHWAY</th>
      <td>0.786601</td>
      <td>0.783462</td>
      <td>0.789740</td>
      <td>0.762031</td>
      <td>0.758883</td>
      <td>0.765180</td>
      <td>5.682502e-24</td>
      <td>2.273001e-23</td>
      <td>0.024569</td>
    </tr>
    <tr>
      <th>VANTVEER_BREAST_CANCER_METASTASIS_UP</th>
      <td>0.788614</td>
      <td>0.785906</td>
      <td>0.791322</td>
      <td>0.764774</td>
      <td>0.760737</td>
      <td>0.768810</td>
      <td>2.718239e-14</td>
      <td>5.436478e-14</td>
      <td>0.023840</td>
    </tr>
    <tr>
      <th>CHARAFE_BREAST_CANCER_LUMINAL_VS_BASAL_UP</th>
      <td>0.711085</td>
      <td>0.708299</td>
      <td>0.713870</td>
      <td>0.692786</td>
      <td>0.688569</td>
      <td>0.697003</td>
      <td>1.017657e-04</td>
      <td>1.221188e-04</td>
      <td>0.018298</td>
    </tr>
    <tr>
      <th>GOZGIT_ESR1_TARGETS_UP</th>
      <td>0.603720</td>
      <td>0.600413</td>
      <td>0.607027</td>
      <td>0.597173</td>
      <td>0.593377</td>
      <td>0.600969</td>
      <td>3.798903e-04</td>
      <td>4.144258e-04</td>
      <td>0.006547</td>
    </tr>
    <tr>
      <th>SMID_BREAST_CANCER_ERBB2_UP</th>
      <td>0.766310</td>
      <td>0.762927</td>
      <td>0.769694</td>
      <td>0.769329</td>
      <td>0.765292</td>
      <td>0.773365</td>
      <td>2.531350e-01</td>
      <td>2.531350e-01</td>
      <td>-0.003018</td>
    </tr>
    <tr>
      <th>CHARAFE_BREAST_CANCER_BASAL_VS_MESENCHYMAL_UP</th>
      <td>0.722666</td>
      <td>0.719682</td>
      <td>0.725650</td>
      <td>0.733513</td>
      <td>0.730308</td>
      <td>0.736717</td>
      <td>6.325670e-06</td>
      <td>8.434226e-06</td>
      <td>-0.010847</td>
    </tr>
    <tr>
      <th>SMID_BREAST_CANCER_BASAL_UP</th>
      <td>0.615393</td>
      <td>0.612546</td>
      <td>0.618239</td>
      <td>0.662852</td>
      <td>0.658132</td>
      <td>0.667573</td>
      <td>1.110056e-38</td>
      <td>6.660333e-38</td>
      <td>-0.047460</td>
    </tr>
  </tbody>
</table>
</div>




```python
cluster_3_results_df = mann_whitney_cluster_1_vs_others(3,cluster_assignments_series, TCGA_breast_ssGSEA_df)
cluster_3_results_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cluster_avg</th>
      <th>cluster_95%_CI_lower</th>
      <th>cluster_95%_CI_upper</th>
      <th>others_avg</th>
      <th>others_95%_CI_lower</th>
      <th>others_95%_CI_upper</th>
      <th>mann-whitney_p-value</th>
      <th>group_FDR_corrected_p-value</th>
      <th>group_avg_diff</th>
    </tr>
    <tr>
      <th>gene_set</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>SMID_BREAST_CANCER_BASAL_UP</th>
      <td>0.735232</td>
      <td>0.731251</td>
      <td>0.739213</td>
      <td>0.628171</td>
      <td>0.625385</td>
      <td>0.630957</td>
      <td>3.957483e-75</td>
      <td>1.187245e-74</td>
      <td>0.107061</td>
    </tr>
    <tr>
      <th>GOZGIT_ESR1_TARGETS_UP</th>
      <td>0.609219</td>
      <td>0.602436</td>
      <td>0.616002</td>
      <td>0.598441</td>
      <td>0.595626</td>
      <td>0.601256</td>
      <td>2.046998e-03</td>
      <td>2.046998e-03</td>
      <td>0.010778</td>
    </tr>
    <tr>
      <th>CHARAFE_BREAST_CANCER_BASAL_VS_MESENCHYMAL_UP</th>
      <td>0.736887</td>
      <td>0.729256</td>
      <td>0.744519</td>
      <td>0.727688</td>
      <td>0.725372</td>
      <td>0.730004</td>
      <td>1.585526e-03</td>
      <td>1.729665e-03</td>
      <td>0.009199</td>
    </tr>
    <tr>
      <th>VANTVEER_BREAST_CANCER_POOR_PROGNOSIS</th>
      <td>0.654742</td>
      <td>0.646865</td>
      <td>0.662619</td>
      <td>0.686908</td>
      <td>0.682066</td>
      <td>0.691750</td>
      <td>3.723831e-06</td>
      <td>4.468597e-06</td>
      <td>-0.032166</td>
    </tr>
    <tr>
      <th>BIOCARTA_HER2_PATHWAY</th>
      <td>0.736225</td>
      <td>0.731277</td>
      <td>0.741173</td>
      <td>0.778124</td>
      <td>0.775693</td>
      <td>0.780554</td>
      <td>7.253501e-33</td>
      <td>9.671335e-33</td>
      <td>-0.041899</td>
    </tr>
    <tr>
      <th>SMID_BREAST_CANCER_ERBB2_UP</th>
      <td>0.724146</td>
      <td>0.718000</td>
      <td>0.730291</td>
      <td>0.775070</td>
      <td>0.772299</td>
      <td>0.777841</td>
      <td>2.763853e-37</td>
      <td>4.145780e-37</td>
      <td>-0.050924</td>
    </tr>
    <tr>
      <th>VANTVEER_BREAST_CANCER_METASTASIS_UP</th>
      <td>0.708722</td>
      <td>0.701283</td>
      <td>0.716161</td>
      <td>0.785338</td>
      <td>0.783087</td>
      <td>0.787589</td>
      <td>5.330270e-57</td>
      <td>9.137606e-57</td>
      <td>-0.076616</td>
    </tr>
    <tr>
      <th>CHARAFE_BREAST_CANCER_LUMINAL_VS_BASAL_UP</th>
      <td>0.622159</td>
      <td>0.618740</td>
      <td>0.625578</td>
      <td>0.712977</td>
      <td>0.710664</td>
      <td>0.715289</td>
      <td>1.044507e-76</td>
      <td>4.178029e-76</td>
      <td>-0.090818</td>
    </tr>
    <tr>
      <th>VANTVEER_BREAST_CANCER_ESR1_UP</th>
      <td>0.601404</td>
      <td>0.597504</td>
      <td>0.605304</td>
      <td>0.772476</td>
      <td>0.769243</td>
      <td>0.775708</td>
      <td>1.294168e-83</td>
      <td>1.553002e-82</td>
      <td>-0.171072</td>
    </tr>
    <tr>
      <th>SMID_BREAST_CANCER_LUMINAL_A_UP</th>
      <td>0.547887</td>
      <td>0.535333</td>
      <td>0.560442</td>
      <td>0.722670</td>
      <td>0.717481</td>
      <td>0.727859</td>
      <td>1.913581e-64</td>
      <td>3.827163e-64</td>
      <td>-0.174783</td>
    </tr>
    <tr>
      <th>SMID_BREAST_CANCER_LUMINAL_B_UP</th>
      <td>0.542894</td>
      <td>0.532927</td>
      <td>0.552861</td>
      <td>0.742771</td>
      <td>0.737756</td>
      <td>0.747785</td>
      <td>3.452304e-73</td>
      <td>8.285530e-73</td>
      <td>-0.199877</td>
    </tr>
    <tr>
      <th>DOANE_BREAST_CANCER_ESR1_UP</th>
      <td>0.601821</td>
      <td>0.590701</td>
      <td>0.612941</td>
      <td>0.810365</td>
      <td>0.806386</td>
      <td>0.814344</td>
      <td>1.337880e-78</td>
      <td>8.027278e-78</td>
      <td>-0.208544</td>
    </tr>
  </tbody>
</table>
</div>




```python
cluster_4_results_df = mann_whitney_cluster_1_vs_others(4,cluster_assignments_series, TCGA_breast_ssGSEA_df)
cluster_4_results_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cluster_avg</th>
      <th>cluster_95%_CI_lower</th>
      <th>cluster_95%_CI_upper</th>
      <th>others_avg</th>
      <th>others_95%_CI_lower</th>
      <th>others_95%_CI_upper</th>
      <th>mann-whitney_p-value</th>
      <th>group_FDR_corrected_p-value</th>
      <th>group_avg_diff</th>
    </tr>
    <tr>
      <th>gene_set</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>SMID_BREAST_CANCER_BASAL_UP</th>
      <td>0.742625</td>
      <td>0.731523</td>
      <td>0.753727</td>
      <td>0.639134</td>
      <td>0.635959</td>
      <td>0.642309</td>
      <td>5.382287e-20</td>
      <td>6.458745e-19</td>
      <td>0.103491</td>
    </tr>
    <tr>
      <th>GOZGIT_ESR1_TARGETS_UP</th>
      <td>0.640950</td>
      <td>0.616463</td>
      <td>0.665437</td>
      <td>0.598372</td>
      <td>0.595871</td>
      <td>0.600872</td>
      <td>1.807520e-04</td>
      <td>1.807520e-04</td>
      <td>0.042578</td>
    </tr>
    <tr>
      <th>CHARAFE_BREAST_CANCER_BASAL_VS_MESENCHYMAL_UP</th>
      <td>0.755225</td>
      <td>0.734534</td>
      <td>0.775916</td>
      <td>0.727960</td>
      <td>0.725765</td>
      <td>0.730155</td>
      <td>6.967029e-06</td>
      <td>8.360435e-06</td>
      <td>0.027265</td>
    </tr>
    <tr>
      <th>SMID_BREAST_CANCER_ERBB2_UP</th>
      <td>0.735925</td>
      <td>0.722281</td>
      <td>0.749569</td>
      <td>0.769276</td>
      <td>0.766506</td>
      <td>0.772046</td>
      <td>2.228184e-06</td>
      <td>2.970911e-06</td>
      <td>-0.033351</td>
    </tr>
    <tr>
      <th>VANTVEER_BREAST_CANCER_METASTASIS_UP</th>
      <td>0.742200</td>
      <td>0.726864</td>
      <td>0.757536</td>
      <td>0.776026</td>
      <td>0.773318</td>
      <td>0.778733</td>
      <td>8.564229e-06</td>
      <td>9.342796e-06</td>
      <td>-0.033826</td>
    </tr>
    <tr>
      <th>VANTVEER_BREAST_CANCER_POOR_PROGNOSIS</th>
      <td>0.636625</td>
      <td>0.610851</td>
      <td>0.662399</td>
      <td>0.684216</td>
      <td>0.679826</td>
      <td>0.688605</td>
      <td>7.432857e-07</td>
      <td>1.114929e-06</td>
      <td>-0.047591</td>
    </tr>
    <tr>
      <th>BIOCARTA_HER2_PATHWAY</th>
      <td>0.725050</td>
      <td>0.716016</td>
      <td>0.734084</td>
      <td>0.774147</td>
      <td>0.771784</td>
      <td>0.776509</td>
      <td>1.715973e-13</td>
      <td>4.118335e-13</td>
      <td>-0.049097</td>
    </tr>
    <tr>
      <th>CHARAFE_BREAST_CANCER_LUMINAL_VS_BASAL_UP</th>
      <td>0.631000</td>
      <td>0.621973</td>
      <td>0.640027</td>
      <td>0.703105</td>
      <td>0.700382</td>
      <td>0.705828</td>
      <td>6.069697e-17</td>
      <td>3.367518e-16</td>
      <td>-0.072105</td>
    </tr>
    <tr>
      <th>SMID_BREAST_CANCER_LUMINAL_A_UP</th>
      <td>0.581475</td>
      <td>0.555689</td>
      <td>0.607261</td>
      <td>0.703044</td>
      <td>0.697081</td>
      <td>0.709008</td>
      <td>1.703582e-12</td>
      <td>2.920427e-12</td>
      <td>-0.121569</td>
    </tr>
    <tr>
      <th>VANTVEER_BREAST_CANCER_ESR1_UP</th>
      <td>0.631700</td>
      <td>0.620729</td>
      <td>0.642671</td>
      <td>0.753364</td>
      <td>0.748930</td>
      <td>0.757798</td>
      <td>2.535543e-16</td>
      <td>7.606628e-16</td>
      <td>-0.121664</td>
    </tr>
    <tr>
      <th>SMID_BREAST_CANCER_LUMINAL_B_UP</th>
      <td>0.581600</td>
      <td>0.551808</td>
      <td>0.611392</td>
      <td>0.720316</td>
      <td>0.714294</td>
      <td>0.726338</td>
      <td>2.395797e-13</td>
      <td>4.791594e-13</td>
      <td>-0.138716</td>
    </tr>
    <tr>
      <th>DOANE_BREAST_CANCER_ESR1_UP</th>
      <td>0.634325</td>
      <td>0.608121</td>
      <td>0.660529</td>
      <td>0.787235</td>
      <td>0.781701</td>
      <td>0.792768</td>
      <td>8.418795e-17</td>
      <td>3.367518e-16</td>
      <td>-0.152910</td>
    </tr>
  </tbody>
</table>
</div>




```python
cluster_5_results_df = mann_whitney_cluster_1_vs_others(5,cluster_assignments_series, TCGA_breast_ssGSEA_df)
cluster_5_results_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cluster_avg</th>
      <th>cluster_95%_CI_lower</th>
      <th>cluster_95%_CI_upper</th>
      <th>others_avg</th>
      <th>others_95%_CI_lower</th>
      <th>others_95%_CI_upper</th>
      <th>mann-whitney_p-value</th>
      <th>group_FDR_corrected_p-value</th>
      <th>group_avg_diff</th>
    </tr>
    <tr>
      <th>gene_set</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>SMID_BREAST_CANCER_ERBB2_UP</th>
      <td>0.833159</td>
      <td>0.824801</td>
      <td>0.841518</td>
      <td>0.763691</td>
      <td>0.761027</td>
      <td>0.766355</td>
      <td>1.561588e-27</td>
      <td>1.873905e-26</td>
      <td>0.069469</td>
    </tr>
    <tr>
      <th>CHARAFE_BREAST_CANCER_BASAL_VS_MESENCHYMAL_UP</th>
      <td>0.752043</td>
      <td>0.744348</td>
      <td>0.759739</td>
      <td>0.727405</td>
      <td>0.725076</td>
      <td>0.729734</td>
      <td>1.203850e-08</td>
      <td>2.407701e-08</td>
      <td>0.024639</td>
    </tr>
    <tr>
      <th>SMID_BREAST_CANCER_BASAL_UP</th>
      <td>0.662043</td>
      <td>0.654390</td>
      <td>0.669697</td>
      <td>0.641624</td>
      <td>0.638162</td>
      <td>0.645085</td>
      <td>2.567898e-07</td>
      <td>3.851847e-07</td>
      <td>0.020420</td>
    </tr>
    <tr>
      <th>DOANE_BREAST_CANCER_ESR1_UP</th>
      <td>0.788884</td>
      <td>0.777594</td>
      <td>0.800174</td>
      <td>0.781174</td>
      <td>0.775167</td>
      <td>0.787181</td>
      <td>1.241216e-02</td>
      <td>1.354054e-02</td>
      <td>0.007710</td>
    </tr>
    <tr>
      <th>GOZGIT_ESR1_TARGETS_UP</th>
      <td>0.598986</td>
      <td>0.590292</td>
      <td>0.607679</td>
      <td>0.599987</td>
      <td>0.597264</td>
      <td>0.602710</td>
      <td>4.727725e-01</td>
      <td>4.727725e-01</td>
      <td>-0.001002</td>
    </tr>
    <tr>
      <th>CHARAFE_BREAST_CANCER_LUMINAL_VS_BASAL_UP</th>
      <td>0.693754</td>
      <td>0.686700</td>
      <td>0.700808</td>
      <td>0.700927</td>
      <td>0.698020</td>
      <td>0.703835</td>
      <td>2.305823e-03</td>
      <td>2.766988e-03</td>
      <td>-0.007173</td>
    </tr>
    <tr>
      <th>VANTVEER_BREAST_CANCER_METASTASIS_UP</th>
      <td>0.755739</td>
      <td>0.748282</td>
      <td>0.763196</td>
      <td>0.776071</td>
      <td>0.773258</td>
      <td>0.778884</td>
      <td>7.807888e-08</td>
      <td>1.338495e-07</td>
      <td>-0.020332</td>
    </tr>
    <tr>
      <th>BIOCARTA_HER2_PATHWAY</th>
      <td>0.741072</td>
      <td>0.735358</td>
      <td>0.746787</td>
      <td>0.774456</td>
      <td>0.772017</td>
      <td>0.776896</td>
      <td>6.468585e-13</td>
      <td>1.940575e-12</td>
      <td>-0.033384</td>
    </tr>
    <tr>
      <th>VANTVEER_BREAST_CANCER_ESR1_UP</th>
      <td>0.716884</td>
      <td>0.708506</td>
      <td>0.725262</td>
      <td>0.751079</td>
      <td>0.746340</td>
      <td>0.755817</td>
      <td>1.032150e-11</td>
      <td>2.477159e-11</td>
      <td>-0.034195</td>
    </tr>
    <tr>
      <th>VANTVEER_BREAST_CANCER_POOR_PROGNOSIS</th>
      <td>0.644261</td>
      <td>0.632354</td>
      <td>0.656168</td>
      <td>0.685046</td>
      <td>0.680501</td>
      <td>0.689590</td>
      <td>2.139576e-06</td>
      <td>2.852768e-06</td>
      <td>-0.040785</td>
    </tr>
    <tr>
      <th>SMID_BREAST_CANCER_LUMINAL_B_UP</th>
      <td>0.647377</td>
      <td>0.632871</td>
      <td>0.661882</td>
      <td>0.719814</td>
      <td>0.713475</td>
      <td>0.726153</td>
      <td>4.992451e-13</td>
      <td>1.940575e-12</td>
      <td>-0.072437</td>
    </tr>
    <tr>
      <th>SMID_BREAST_CANCER_LUMINAL_A_UP</th>
      <td>0.609623</td>
      <td>0.594295</td>
      <td>0.624951</td>
      <td>0.704585</td>
      <td>0.698462</td>
      <td>0.710707</td>
      <td>6.416670e-17</td>
      <td>3.850002e-16</td>
      <td>-0.094961</td>
    </tr>
  </tbody>
</table>
</div>

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
