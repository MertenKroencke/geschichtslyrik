import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

import researchpy
import random

import scipy.stats
from scipy.stats import ttest_ind
from scipy.stats import mannwhitneyu
from scipy.stats import chi2_contingency
from scipy.stats import fisher_exact
from scipy.stats import pearsonr
from scipy.stats import pointbiserialr

import statsmodels
from statsmodels.stats.proportion import confint_proportions_2indep
from statsmodels.stats.proportion import proportion_confint

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report

from resources_geschichtslyrik import *

# def get_cramers_v (contingency_table, posneg = False):
#     chi_square = chi2_contingency(contingency_table, correction = False, lambda_= None)[0]
#     n = np.sum(contingency_table)
#     m = min(contingency_table.shape)-1
#     
#     cramers_v = np.sqrt(chi_square/(n*m))
#     if posneg: cramers_v = -cramers_v if contingency_table[0,0] / contingency_table[0,1] > contingency_table[1,0] / contingency_table[1,1] else cramers_v
#     return cramers_v
# 
def get_phi (contingency_table, posneg = False):
    chi_square = chi2_contingency(contingency_table, correction = False, lambda_= None)[0]
    n = np.sum(contingency_table)
    
    phi = np.sqrt(chi_square/n)
    if posneg: phi = -phi if contingency_table[0,0] / contingency_table[0,1] > contingency_table[1,0] / contingency_table[1,1] else phi
    return phi

# def get_phi_max (contingency_table):
#     Px = contingency_table[0,0] + contingency_table[0,1]
#     Qx = contingency_table[1,0] + contingency_table[1,1]
#     Py = contingency_table[0,0] + contingency_table[1,0]
#     Qy = contingency_table[0,1] + contingency_table[1,1]
#     
#     phi_max_positive = min([np.sqrt(Px*Qy/(Py*Qx)), np.sqrt(Py*Qx/(Px*Qy))])
#     phi_max_negative = max([-np.sqrt(Px*Py/(Qx*Qy)), -np.sqrt(Qx*Qy/(Px*Py))])
#     phi_max_total = max([abs(phi_max_positive), abs(phi_max_negative)])
#     
#     return phi_max_total
# 
def get_pooled_stdev (list1, list2):
    return np.sqrt((np.std(list1) ** 2 + np.std(list2) ** 2) / 2)

def get_cohens_d (list1, list2):
    mean_list1 = np.mean(list1)
    mean_list2 = np.mean(list2)
    pooled_stdev = get_pooled_stdev (list1, list2)
    
    cohens_d = (mean_list1 - mean_list2) / pooled_stdev
    return cohens_d
    
# def get_hedges_g (list1, list2):
#     return researchpy.ttest(pd.Series(list1), pd.Series(list2))[1].iloc[7,1]
#   
# def get_pooled_prop(contingency_table):
#     return [(contingency_table[0,0] + contingency_table[1,0]) / np.sum(contingency_table),
#            (contingency_table[0,1] + contingency_table[1,1]) / np.sum(contingency_table)]
#   
# def z_test_prop(contingency_table, adjust = True, warnings = False):
#     n1 = np.sum(contingency_table[0,:])
#     n2 = np.sum(contingency_table[1,:])
#     prop1 = contingency_table[0,1] / np.sum(contingency_table[0,:])
#     prop2 = contingency_table[1,1] / np.sum(contingency_table[1,:])
#     prop_pooled = get_pooled_prop(contingency_table)[1]
#     adjustor = 0 if adjust == False else 0.5 * (1/n1 + 1/n2)
#     
#     if warnings:
#         if n1*prop_pooled < 5: print("warning: n1*prop_pooled < 5")
#         if n1*(1-prop_pooled) < 5: print("warning: n1*(1-prop_pooled) < 5")
#         if n2*prop_pooled < 5: print("warning: n2*prop_pooled < 5")
#         if n2*(1-prop_pooled) < 5: print("warning: n2*(1-prop_pooled) < 5")
#     
#     z = (prop1 - prop2 - adjustor) / np.sqrt( (prop_pooled * (1 - prop_pooled) / n1) + (prop_pooled * (1 - prop_pooled) / n2) )
#     z_pvalue = scipy.stats.norm.sf(abs(z))*2
#     
#     return [z, z_pvalue]
# 
# def get_confint_simple(contingency_table):
#     # equal to confint_proportions_2indep with method = 'wald'
# 
#     n1 = np.sum(contingency_table.iloc[0,:])
#     n2 = np.sum(contingency_table.iloc[1,:])
#     p1 = contingency_table.iloc[0,1] / np.sum(contingency_table.iloc[0,:])
#     p2 = contingency_table.iloc[1,1] / np.sum(contingency_table.iloc[1,:])
#     
#     zstar = 1.96
#     s = np.sqrt( (p1*(1-p1)/n1) + (p2*(1-p2)/n2) )
#     me = zstar*s
#     
#     confint = [p1-p2-me, p1-p2+me]
#     
#     return confint
#     
# def get_risk_ratio(contingency_table):
#     p1 = contingency_table[0,0] / np.sum(contingency_table[0,:])
#     p2 = contingency_table[1,0] / np.sum(contingency_table[1,:])
#         
#     rr = p2/p1
#     
#     return rr
#     
# def get_number_needed_to_treat (contingency_table):
# 	# interpretation: number of texts you need to look at to see a difference
#     p1 = contingency_table[0,0] / np.sum(contingency_table[0,:])
#     p2 = contingency_table[1,0] / np.sum(contingency_table[1,:])
#      
#     nnt = float('NaN') if p1 == p2 else abs(1/(p1-p2))
#     
#     return nnt
#     
# def get_odds_ratio(contingency_table, adjust = False):
#     # interpretation: how many times greater are the odds that 
#     # a member of a population (e.g., realism) will fall into a category (e.g., is_ballad) 
#     # than the odds that a member of another population (e.g., modernism)
#     # will fall into that category.
#     # odds are not probabilities
#     if adjust: contingency_table = contingency_table + 0.5
#     
#     odds1 = contingency_table[0,0] / contingency_table[0,1]
#     odds2 = contingency_table[1,0] / contingency_table[1,1]
#     
#     odds_ratio = odds2 / odds1
#     
#     return odds_ratio

def get_features(corr_series, threshold = 0.2, mode = 'bin'):
    corr_series = corr_series.sort_values().abs()
    corr_series = corr_series[corr_series < 1]
    corr_categories = corr_series[corr_series >= threshold].index.tolist()
    corr_categories = [x for x in corr_categories if feature_dict[x] == mode]
    
    if mode == 'bin':
        corr_categories = corr_categories + [
            'stoffgebiet_neutral', 'stoffgebiet_positiv', 'stoffgebiet_negativ', 'stoffgebiet_ambivalent',
            'entity_neutral', 'entity_positiv', 'entity_negativ', 'entity_ambivalent',
            'bekanntes_individuum_positiv', 'bekanntes_individuum_negativ',
            'unbekanntes_individuum_positiv', 'unbekanntes_individuum_negativ',
            'kollektiv_positiv', 'kollektiv_negativ',
        ]
    corr_categories = list(set(corr_categories))
    
    return corr_categories
    
def get_contingency_table_for_ratings (meta, main_feature, comp_feature):
    meta_with_main = meta[meta[main_feature] == 1]
    meta_without_main = meta[meta[main_feature] == 0]
    
    if 'stoffgebiet' in comp_feature:
        meta_with_main_ratings = get_rating_table(meta_with_main, mode = 'themes')
        meta_without_main_ratings = get_rating_table(meta_without_main, mode = 'themes')
    else:
        meta_with_main_ratings = get_rating_table(meta_with_main, mode = 'entity')
        meta_without_main_ratings = get_rating_table(meta_without_main, mode = 'entity')
    
    if '_neutral' in comp_feature:
        this_rating = '0'
    elif '_positiv' in comp_feature:
        this_rating = '1'
    elif '_negativ' in comp_feature:
        this_rating = '2'
    elif '_ambivalent' in comp_feature:
        this_rating = '3'
    
    if 'entity' in comp_feature or 'stoffgebiet' in comp_feature:
        contingency_table = [
            [meta_without_main_ratings.query("rating != @this_rating").shape[0], 
             meta_without_main_ratings.query("rating == @this_rating").shape[0]],
            [meta_with_main_ratings.query("rating != @this_rating").shape[0], 
             meta_with_main_ratings.query("rating == @this_rating").shape[0]],
        ]
        
    else:
        if 'unbekanntes_individuum' in comp_feature:
            this_type = '2'
        elif 'bekanntes_individuum' in comp_feature:
            this_type = '1'
        elif 'kollektiv' in comp_feature:
            this_type = '3'
        elif 'nichtmensch' in comp_feature:
            this_type = '4'

        contingency_table = [
            [meta_without_main_ratings.query("type == @this_type and rating != @this_rating").shape[0], 
             meta_without_main_ratings.query("type == @this_type and rating == @this_rating").shape[0]],
            [meta_with_main_ratings.query("type == @this_type and rating != @this_rating").shape[0], 
             meta_with_main_ratings.query("type == @this_type and rating == @this_rating").shape[0]],
        ]
    
    contingency_table = pd.DataFrame(contingency_table)
    
    return contingency_table

def relations_binbin (meta, main_feature, comp_features, main_feature_label = 'main_feature'):
    results = pd.DataFrame()
 
    for comp_feature in comp_features:
        # print(comp_feature)
        contingency_table = pd.crosstab(meta[main_feature], meta[comp_feature])
        
        if any(x in comp_feature for x in ['_neutral', '_positiv', '_negativ', '_ambivalent']):
            if any(x in comp_feature for x in ['individuum', 'kollektiv', 'nichtmensch', 'entity', 'stoffgebiet']):
                contingency_table = get_contingency_table_for_ratings(meta, main_feature, comp_feature)
        
        if any(contingency_table.sum(axis = 0) == 0) or any(contingency_table.sum(axis = 1) == 0):
            continue
            
        chi2_results = chi2_contingency(contingency_table, correction=False)
        phi = get_phi(np.array(contingency_table))

        if 0 not in contingency_table.columns:
            contingency_table[0] = 0
        if 1 not in contingency_table.columns:
            contingency_table[1] = 0
        if 0 not in contingency_table.index:
            contingency_table.loc[0] = 0
        if 1 not in contingency_table.index:
            contingency_table.loc[1] = 0
        contingency_table = contingency_table.iloc[:,:2]

        fisher_exact_results = fisher_exact(contingency_table) 

        all_not_main = contingency_table.loc[0].sum() # texts without main_feature
        comp_when_not_main = contingency_table.loc[0,1] # texts without main_feature with comp_feature
        share_when_not_main = float('NaN') if all_not_main == 0 else comp_when_not_main/all_not_main

        all_main = contingency_table.loc[1].sum() # texts with main_feature
        comp_when_main = contingency_table.loc[1,1] # texts with main_feature with comp_feature
        share_when_main = float('NaN') if all_main == 0 else comp_when_main/all_main

        confint_of_diff = confint_proportions_2indep(
            count1 = comp_when_main, nobs1 = all_main, 
            count2 = comp_when_not_main, nobs2 = all_not_main,
            method = 'wald',
        )

        # bootstrap
        nomain_data = [0] * contingency_table.iloc[0, 0] + [1] * contingency_table.iloc[0, 1]
        main_data = [0] * contingency_table.iloc[1, 0] + [1] * contingency_table.iloc[1, 1]
        diffs_of_means = []
        for i in range(1000):
            nomain_data_sample = random.choices(nomain_data, k = len(nomain_data))
            main_data_sample = random.choices(main_data, k = len(main_data))
            diffs_of_means.append(np.mean(main_data_sample) - np.mean(nomain_data_sample))

        results.loc[comp_feature, f'wenn_nicht'] = share_when_not_main
        results.loc[comp_feature, f'wenn_nicht_detail'] = str(comp_when_not_main) + '/' + str(all_not_main)
        results.loc[comp_feature, f'wenn_ja'] = share_when_main
        results.loc[comp_feature, f'wenn_ja_detail'] = str(comp_when_main) + '/' + str(all_main)
        results.loc[comp_feature, 'diff_low_bootstrap'] = np.percentile(diffs_of_means,q=2.5)
        results.loc[comp_feature, 'diff_low'] = confint_of_diff[0]
        results.loc[comp_feature, 'diff'] = share_when_main - share_when_not_main
        results.loc[comp_feature, 'diff_high'] = confint_of_diff[1]
        results.loc[comp_feature, 'diff_high_bootstrap'] = np.percentile(diffs_of_means,q=97.5)

        results.loc[comp_feature, 'chi2'] = chi2_results[0]
        results.loc[comp_feature, 'chi2_p'] = chi2_results[1]
        results.loc[comp_feature, 'fisher_p'] = fisher_exact_results[1]
        results.loc[comp_feature, 'phi'] = phi
        results.loc[comp_feature, 'min_real'] = contingency_table.min().min()
        results.loc[comp_feature, 'min_expected'] = chi2_results[3].min()
        
    return results
    
def relations_bincont (meta, main_feature, comp_features, main_feature_label = 'main_feature'):
    results = pd.DataFrame()
    
    max_count = 4

    for comp_feature in comp_features:
        
        meta_nona = meta.dropna(subset=[main_feature, comp_feature])
        meta_with_main = meta_nona[meta_nona[main_feature] == 1]
        meta_without_main = meta_nona[meta_nona[main_feature] == 0]
        
        this_meta = meta_without_main
        this_all_count = this_meta.shape[0]
        this_label = f"wenn_nicht"
        
        results.loc[comp_feature, this_label] = this_meta[comp_feature].mean()  
        for i in range(0,max_count+1):
            if i < max_count:
                this_count = this_meta[this_meta[comp_feature] == i].shape[0]
                this_share = this_count/this_meta.shape[0]
                results.loc[comp_feature, f"a_merkmal={i}"] = f"{round(this_share, 2)} [{this_count}/{this_all_count}]"
        
            if i == max_count: 
                this_count = this_meta[this_meta[comp_feature] >= i].shape[0]
                this_share = this_count/this_meta.shape[0]
                results.loc[comp_feature, f"a_merkmal>={i}"] = f"{round(this_share, 2)} [{this_count}/{this_all_count}]"
                
        this_meta = meta_with_main
        this_all_count = this_meta.shape[0]
        this_label = f"wenn_ja"
        
        results.loc[comp_feature, this_label] = this_meta[comp_feature].mean()  
        for i in range(0,max_count+1):
            if i < max_count:
                this_count = this_meta[this_meta[comp_feature] == i].shape[0]
                this_share = this_count/this_meta.shape[0]
                results.loc[comp_feature, f"b_merkmal={i}"] = f"{round(this_share, 2)} [{this_count}/{this_all_count}]"
        
            if i == max_count: 
                this_count = this_meta[this_meta[comp_feature] >= i].shape[0]
                this_share = this_count/this_meta.shape[0]
                results.loc[comp_feature, f"b_merkmal>={i}"] = f"{round(this_share, 2)} [{this_count}/{this_all_count}]"
                   
        pointbiserialr_results = pointbiserialr(meta_nona[main_feature], meta_nona[comp_feature])
        results.loc[comp_feature, 'pointbiserialr_corr'] = pointbiserialr_results[0]
        results.loc[comp_feature, 'pointbiserialr_p'] = pointbiserialr_results[1]

        ttest_results = ttest_ind(
            meta_without_main[comp_feature],
            meta_with_main[comp_feature],
        )
        results.loc[comp_feature, 'ttest_p'] = ttest_results[1]

        cohens_d = get_cohens_d(
            meta_without_main[comp_feature],
            meta_with_main[comp_feature],
        )
        results.loc[comp_feature, 'cohens_d'] = cohens_d
        
        mannwhitneyu_results = mannwhitneyu(
            meta_without_main[comp_feature],
            meta_with_main[comp_feature],
        )
        results.loc[comp_feature, 'mannwhitneyu_stat'] = mannwhitneyu_results[0]
        results.loc[comp_feature, 'mannwhitneyu_p'] = mannwhitneyu_results[1]
        
        # bootstrap
        nomain_data = meta_without_main[comp_feature].tolist()
        main_data = meta_with_main[comp_feature].tolist()
        diffs_of_means = []
        for i in range(1000):
            nomain_data_sample = random.choices(nomain_data, k = len(nomain_data))
            main_data_sample = random.choices(main_data, k = len(main_data))
            diffs_of_means.append(np.mean(main_data_sample) - np.mean(nomain_data_sample))
        results.loc[comp_feature, 'meandiffs_ci_lower'] = ttest_results.confidence_interval()[1]*-1
        results.loc[comp_feature, 'meandiffs_ci_bootstrap_lower'] = np.percentile(diffs_of_means,q=2.5)
        results.loc[comp_feature, 'meandiffs_ci_upper'] = ttest_results.confidence_interval()[0]*-1
        results.loc[comp_feature, 'meandiffs_ci_bootstrap_upper'] = np.percentile(diffs_of_means,q=97.5)
    
    return results
    
    
def relations_contbin (meta, main_feature, comp_features):
    results = pd.DataFrame()
    
    max_count = 3
    for comp_feature in comp_features:
        for i in range(0,max_count+1):
            main_feature_count = meta[(meta[main_feature] == i) & (meta[comp_feature].notna())].shape[0]
            main_and_comp_feature_count = meta[(meta[main_feature] == i) & (meta[comp_feature] == 1)].shape[0]
            share = 0 if main_feature_count == 0 else main_and_comp_feature_count/main_feature_count
            input_content = f"{share} [{main_and_comp_feature_count}/{main_feature_count}]"
            results.loc[comp_feature, f'wenn {main_feature} = {i}: Anteil Texte mit Feature = ...'] = input_content
        
            if i == max_count:
                main_feature_count = meta[(meta[main_feature] > i) & (meta[comp_feature].notna())].shape[0]
                main_and_comp_feature_count = meta[(meta[main_feature] > i) & (meta[comp_feature] == 1)].shape[0]
                share = 0 if main_feature_count == 0 else main_and_comp_feature_count/main_feature_count
                input_content = f"{share} [{main_and_comp_feature_count}/{main_feature_count}]"
                results.loc[comp_feature, f'wenn {main_feature} > {i}: Anteil Texte mit Feature = ...'] = input_content
    
        meta_corr = meta.dropna(subset=[main_feature, comp_feature])
        pointbiserialr_results = pointbiserialr(meta_corr[main_feature], meta_corr[comp_feature])
        results.loc[comp_feature, 'pointbiserialr_corr'] = pointbiserialr_results[0]
        results.loc[comp_feature, 'pointbiserialr_p'] = pointbiserialr_results[1]

        ttest_results = ttest_ind(
            meta[meta[comp_feature] == 0][main_feature],
            meta[meta[comp_feature] == 1][main_feature],
        )
        results.loc[comp_feature, 'ttest_p'] = ttest_results[1]

        cohens_d = get_cohens_d(
            meta[meta[comp_feature] == 0][main_feature],
            meta[meta[comp_feature] == 1][main_feature],
        )
        results.loc[comp_feature, 'cohens_d'] = cohens_d
        
        mannwhitneyu_results = mannwhitneyu(
            meta[meta[comp_feature] == 0][main_feature],
            meta[meta[comp_feature] == 1][main_feature],
        )
        results.loc[comp_feature, 'mannwhitneyu_stat'] = mannwhitneyu_results[0]
        results.loc[comp_feature, 'mannwhitneyu_p'] = mannwhitneyu_results[1]
    
    return results
    
# def relations_contbin_ratings(meta, main_feature, max_count = 4):
#     results = pd.DataFrame()
#     
#     comp_features = [
#         'stoffgebiet_neutral', 'stoffgebiet_positiv', 'stoffgebiet_negativ', 'stoffgebiet_ambivalent',
#         'entity_neutral', 'entity_positiv', 'entity_negativ', 'entity_ambivalent',
#         'bekanntes_individuum_positiv', 'bekanntes_individuum_negativ',
#         'unbekanntes_individuum_positiv', 'unbekanntes_individuum_negativ',
#         'kollektiv_positiv', 'kollektiv_negativ',
#     ]
#     
#     ratingtable_entity = get_rating_table(meta, mode='entity')   
#     ratingtable_themes = get_rating_table(meta, mode='themes')
#     
#     for comp_feature in tqdm(comp_features):
#         if 'stoffgebiet' in comp_feature:
#             ratingtable = ratingtable_themes
#         else:
#             ratingtable = ratingtable_entity   
#         ratingtable['author_title'] = ratingtable['author'] + ' – ' + ratingtable['title']
# 
#         if '_neutral' in comp_feature:
#             this_rating = '0'
#         elif '_positiv' in comp_feature:
#             this_rating = '1'
#         elif '_negativ' in comp_feature:
#             this_rating = '2'
#         elif '_ambivalent' in comp_feature:
#             this_rating = '3'
#         
#         this_type = ''
#         if 'unbekanntes_individuum' in comp_feature:
#             this_type = '2'
#         elif 'bekanntes_individuum' in comp_feature:
#             this_type = '1'
#         elif 'kollektiv' in comp_feature:
#             this_type = '3'
#         elif 'nichtmensch' in comp_feature:
#             this_type = '4'
#     
#         for i in range(1,max_count+1):
#             meta_main = meta[meta[main_feature] == i]
#             meta_main_authortitle = meta_main['author_title']
#             ratings_main_all = ratingtable.query("author_title.isin(@meta_main_authortitle)")
#             if this_type != '':
#                 ratings_main_all = ratings_main_all.query("type == @this_type")
#             ratings_main_true = ratings_main_all.query("rating == @this_rating")            
#             this_result_str = f"{round(ratings_main_true.shape[0]/ratings_main_all.shape[0], 4)} [{ratings_main_true.shape[0]}/{ratings_main_all.shape[0]}]"
#             results.at[comp_feature, f'wenn {main_feature} = {i}: Anteil mit Feature = ...'] = this_result_str
# 
#         ratings_true = ratingtable.query("rating == @this_rating")
#         if this_type != '':
#             ratings_true = ratings_true.query("type == @this_type")
#         ratings_true_authortitle = ratings_true['author_title']
#         main_values_when_true = []
#         for author_title in ratings_true_authortitle:
#             main_values_when_true.append(meta.query("author_title == @author_title")[main_feature].tolist()[0])
# 
#         ratings_false = ratingtable.query("rating != @this_rating")
#         if this_type != '':
#             ratings_false = ratings_false.query("type == @this_type")
#         ratings_false_authortitle = ratings_false['author_title']
#         main_values_when_false = []
#         for author_title in ratings_false_authortitle:
#             main_values_when_false.append(meta.query("author_title == @author_title")[main_feature].tolist()[0])
# 
#         corr_table = pd.DataFrame([
#             [1] * len(main_values_when_true) + [0] * len(main_values_when_false), 
#             main_values_when_true + main_values_when_false
#         ]).T
# 
#         pointbiserialr_results = pointbiserialr(corr_table[0], corr_table[1])
#         results.loc[comp_feature, 'pointbiserialr_corr'] = pointbiserialr_results[0]
#         results.loc[comp_feature, 'pointbiserialr_p'] = pointbiserialr_results[1]
# 
#         mannwhitneyu_results = mannwhitneyu(
#             main_values_when_true,
#             main_values_when_false,
#         )
#         results.loc[comp_feature, 'mannwhitneyu_stat'] = mannwhitneyu_results[0]
#         results.loc[comp_feature, 'mannwhitneyu_p'] = mannwhitneyu_results[1]
# 
#     return results
    
def relations_contbin_ratings(meta, main_feature, max_count=4):
    results = pd.DataFrame()

    comp_features = [
        'stoffgebiet_neutral', 'stoffgebiet_positiv', 'stoffgebiet_negativ', 'stoffgebiet_ambivalent',
        'entity_neutral', 'entity_positiv', 'entity_negativ', 'entity_ambivalent',
        'bekanntes_individuum_positiv', 'bekanntes_individuum_negativ',
        'unbekanntes_individuum_positiv', 'unbekanntes_individuum_negativ',
        'kollektiv_positiv', 'kollektiv_negativ',
    ]

    # Precompute rating tables
    ratingtable_entity = get_rating_table(meta, mode='entity')
    ratingtable_themes = get_rating_table(meta, mode='themes')
    meta['author_title'] = meta['author'] + ' – ' + meta['title']

    for comp_feature in tqdm(comp_features):
        # Select the appropriate rating table
        ratingtable = ratingtable_themes if 'stoffgebiet' in comp_feature else ratingtable_entity
        ratingtable['author_title'] = ratingtable['author'] + ' – ' + ratingtable['title']

        # Determine the rating and type
        this_rating = '0' if '_neutral' in comp_feature else \
                      '1' if '_positiv' in comp_feature else \
                      '2' if '_negativ' in comp_feature else '3'
        
        this_type = '2' if 'unbekanntes_individuum' in comp_feature else \
                    '1' if 'bekanntes_individuum' in comp_feature else \
                    '4' if 'nichtmensch' in comp_feature else \
                    '3' if 'kollektiv' in comp_feature else ''

        # Pre-filter rating table by type
        if this_type:
            ratingtable = ratingtable[ratingtable['type'] == this_type]

        for i in range(1, max_count + 1):
            meta_main = meta[meta[main_feature] == i]
            ratings_main_all = ratingtable[ratingtable['author_title'].isin(meta_main['author_title'])]

            ratings_main_true = ratings_main_all[ratings_main_all['rating'] == this_rating]
            true_count = ratings_main_true.shape[0]
            all_count = ratings_main_all.shape[0]
            this_result_str = f"{round(true_count / all_count, 4) if all_count > 0 else 0.0} [{true_count}/{all_count}]"
            results.at[comp_feature, f'wenn {main_feature} = {i}: Anteil mit Feature = ...'] = this_result_str

        # Calculate correlations
        ratings_true = ratingtable[ratingtable['rating'] == this_rating][['author_title']]
        ratings_false = ratingtable[ratingtable['rating'] != this_rating][['author_title']]

        true_values = ratings_true.merge(meta, on='author_title', how='left')[main_feature].values
        false_values = ratings_false.merge(meta, on='author_title', how='left')[main_feature].values

        corr_table = pd.DataFrame({'label': [1] * len(true_values) + [0] * len(false_values),
                                   'value': list(true_values) + list(false_values)})

        pointbiserialr_results = pointbiserialr(corr_table['label'], corr_table['value'])
        results.loc[comp_feature, 'pointbiserialr_corr'] = pointbiserialr_results.correlation
        results.loc[comp_feature, 'pointbiserialr_p'] = pointbiserialr_results.pvalue

        mannwhitneyu_results = mannwhitneyu(true_values, false_values, alternative='two-sided')
        results.loc[comp_feature, 'mannwhitneyu_stat'] = mannwhitneyu_results.statistic
        results.loc[comp_feature, 'mannwhitneyu_p'] = mannwhitneyu_results.pvalue

    return results


def relations_contcont (meta, main_feature, comp_features):
    results = pd.DataFrame()
    
    max_count = 3
    for comp_feature in comp_features:
        for i in range(0,max_count+1):
            main_feature_meta = meta[meta[main_feature] == i]
            comp_feature_values = main_feature_meta[comp_feature]
            results.loc[comp_feature, f'wenn {main_feature} = {i}: Mittelwert Feature = ...'] = comp_feature_values.mean()
        
        if i == max_count:
            main_feature_meta = meta[meta[main_feature] > i]
            comp_feature_values = main_feature_meta[comp_feature]
            results.loc[comp_feature, f'wenn {main_feature} > {i}: Mittelwert Feature = ...'] = comp_feature_values.mean()
        
        meta_without_nans = meta.dropna(subset=[main_feature, comp_feature])
        pearsonr_results = pearsonr(meta_without_nans[main_feature], meta_without_nans[comp_feature])
        results.loc[comp_feature, 'pearsonr_corr'] = pearsonr_results[0]
        results.loc[comp_feature, 'pearsonr_p'] = pearsonr_results[1]

    return results
    
# alternative default: win_type = None/'exponential', tau = None/10
def smooth(series, window_size = 11, center = True, win_type = 'exponential', tau = 10, mode = 'mean'):
    rolling_window = series.rolling(window_size, center = center, win_type = win_type)
    if mode == 'mean':
        return rolling_window.mean(tau = tau)
    if mode == 'sum':
        return rolling_window.sum(tau = tau)
