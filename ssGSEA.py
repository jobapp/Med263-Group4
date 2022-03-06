from pandas import concat, DataFrame

#from ._single_sample_gseas import _single_sample_gseas
#from .multiprocess import multiprocess
#from .split_df import split_df

from numpy import full, nan, absolute, in1d, asarray, round

#from .single_sample_gsea import single_sample_gsea

from warnings import warn

import sys

from multiprocessing.pool import Pool

from numpy.random import seed

def split_df(df, axis, n_split):

    if not (0 < n_split <= df.shape[axis]):

        raise ValueError(
            "Invalid: 0 < n_split ({}) <= n_slices ({})".format(n_split, df.shape[axis])
        )

    n = df.shape[axis] // n_split

    dfs = []

    for i in range(n_split):

        start_i = i * n

        end_i = (i + 1) * n

        if axis == 0:

            dfs.append(df.iloc[start_i:end_i])

        elif axis == 1:

            dfs.append(df.iloc[:, start_i:end_i])

    i = n * n_split

    if i < df.shape[axis]:

        if axis == 0:

            dfs.append(df.iloc[i:])

        elif axis == 1:

            dfs.append(df.iloc[:, i:])

    return dfs

def multiprocess(callable_, args, n_job, random_seed=20121020):

    seed(random_seed)

    with Pool(n_job) as process:

        return process.starmap(callable_, args)

def _single_sample_gseas(gene_x_sample,
                         gene_sets,
                         statistic,
                         alpha,
                         sample_norm_type):

    print("Running single-sample GSEA with {} gene sets ...".format(gene_sets.shape[0]))

    score__gene_set_x_sample = full((gene_sets.shape[0], gene_x_sample.shape[1]), nan)

    for sample_index, (sample_name, gene_score) in enumerate(gene_x_sample.items()):

        for gene_set_index, (gene_set_name, gene_set_genes) in enumerate(
            gene_sets.iterrows()):

            score__gene_set_x_sample[gene_set_index, sample_index] = single_sample_gsea(
                                                                                      gene_score,
                                                                                      gene_set_genes,
                                                                                      statistic=statistic,
                                                                                      alpha=alpha,
                                                                                      sample_norm_type = sample_norm_type)

    score__gene_set_x_sample = DataFrame(
        score__gene_set_x_sample, index=gene_sets.index, columns=gene_x_sample.columns
    )

    return score__gene_set_x_sample

def single_sample_gseas(
    gene_x_sample,
    gene_sets,
    statistic="ks",
    alpha=1.0,
    n_job=1,
    file_path=None,
        sample_norm_type = None):

    score__gene_set_x_sample = concat(
        multiprocess(
            _single_sample_gseas,
            (
                (gene_x_sample, gene_sets_, statistic, alpha, sample_norm_type)
                for gene_sets_ in split_df(gene_sets, 0, min(gene_sets.shape[0], n_job))
            ),
            n_job,
        )
    )

    if file_path is not None:

        score__gene_set_x_sample.to_csv(file_path, sep="\t")

    return score__gene_set_x_sample




def single_sample_gsea(
    gene_score,
    gene_set_genes,
    statistic="ks",
    alpha=1.0,
    plot_gene_names = False,
    title=None,
    gene_score_name=None,
    annotation_text_font_size=12,
    annotation_text_width=100,
    annotation_text_yshift=50,
    sample_norm_type = None,
):

    if sample_norm_type == 'rank':
        gene_score = gene_score.rank(method='average', numeric_only=None, na_option='keep', ascending=True, pct=False)
        gene_score = 10000 * (gene_score - gene_score.min())/(gene_score.max() - gene_score.min())
    elif sample_norm_type == 'zscore':
        gene_score = (gene_score - gene_score.mean())/gene_score.std()
    elif sample_norm_type is not None:
        sys.exit('ERROR: unknown sample_norm_type: {}'.format(sample_norm_type))
        
    gene_score_sorted = gene_score.sort_values(ascending=False)
    
    gene_set_gene_None = {gene_set_gene: None for gene_set_gene in gene_set_genes}

    in_ = asarray(
        [
            gene_score_gene in gene_set_gene_None
            #for gene_score_gene in gene_score.index.values
            for gene_score_gene in gene_score_sorted.index.values
        ],
        dtype=int,
    )

    #print(in_)
    #print(gene_score_sorted)
    
    up = in_ * absolute(gene_score_sorted.values)**alpha
    up /= up.sum()
    down = 1.0 - in_
    down /= down.sum()
    cumsum = (up - down).cumsum()
    up_CDF = up.cumsum()
    down_CDF = down.cumsum()

    if statistic == "ks":

        max_ = cumsum.max()
        min_ = cumsum.min()
        if absolute(min_) < absolute(max_):
            gsea_score = max_
        else:
            gsea_score = min_
            
    elif statistic == "auc":
        gsea_score = cumsum.sum()

    gsea_score = round(gsea_score, 3)
        


    return gsea_score
