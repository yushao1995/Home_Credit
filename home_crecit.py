#%% import packages

import pandas as pd
import category_encoders as ce
import numpy as np
from pandas_profiling import ProfileReport
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.impute import MissingIndicator
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from patsy import dmatrices
from biokit import corrplot
import scipy
import scipy.cluster.hierarchy as sch
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf


#%% read data

app = pd.read_csv("/Users/dauku/Desktop/Python/homecredit/home-credit-default-risk/application_train.csv")


#%% md

## EDA

#%% Profiling

# summary
Profile = ProfileReport(app, title = "Pandas profiling Report", html = {"style": {"full_width": True}})

#%%

# view profile
Profile

# Save profile
Profile.to_file(output_file="/Users/dauku/Desktop/Git/DavidKu_IAA2020/Home_Credit/Application_html.html")

#%% Correlation

# pear_corr = app.corr(method = "pearson")
# pear_corr[(pear_corr <= 0.5) & (pear_corr >= -0.5)] = 0
#
# spear_corr = app.corr(method = "spearman")
# spear_corr[(spear_corr <= 0.5) & (spear_corr >= -0.5)] = 0

#%% Cleaning Class
class Cleaning:

    def __init__(self, path):
        self.data = pd.read_csv(path)
        self.summary_report = None
        self.vif = None
        self.target_freq = None

    def get_data(self):
        return(self.data)

    def summary(self):
        uniques = self.data.nunique()
        dtypes = self.data.dtypes
        missing = self.data.isnull().sum()

        report = pd.DataFrame(uniques)
        report.columns = ["uniques"]
        report["dtypes"] = dtypes
        report["missing"] = missing
        report["missing_pct"] = report.missing / self.data.shape[0]

        self.summary_report = report

    def get_summary(self):
        return(self.summary_report)

    def categorical(self):
        nunique = self.data.nunique()
        binary_list = nunique[nunique == 2].index.tolist()
        self.data[binary_list] = self.data[binary_list].astype("category")
        # binary_list = self.summary()[self.summary["uniques"] == 2].index.tolist()
        # self.data[binary_list] = self.data[binary_list].astype("category")

        dtypes = self.data.dtypes
        object_list = dtypes[dtypes == "object"].index.tolist()
        # object_list = self.summary()[self.summary()["dtypes"] == "object"].index.tolist()
        self.data[object_list] = self.data[object_list].astype("category")

    def imputation(self, threshold, drop):
        self.summary()
        # vars that need imputation
        imput_list = self.summary_report[(self.summary_report["missing_pct"] < threshold) & (self.summary_report["missing_pct"] > 0)]
        imputing = self.data[imput_list.index]

        # vars that don't contain any missings
        no_missing_list = self.summary_report[self.summary_report["missing_pct"] == 0]
        no_missing = self.data[no_missing_list.index]

        # impute categorical variables
        imputing_cat = imputing.select_dtypes(exclude="number")
        cat_var = imputing_cat.columns
        cat_imputer = SimpleImputer(strategy="constant", fill_value="Missing")
        cat_imputted = pd.DataFrame(cat_imputer.fit_transform(imputing_cat))
        cat_imputted.columns = cat_var
        cat_imputted = cat_imputted.astype("category")

        # imputing numerical variables
        imputing_num = imputing.select_dtypes(include="number")
        num_var = imputing_num.columns.tolist()
        num_var_suffix = [x + "_indicator" for x in num_var]
        num_var = num_var + num_var_suffix
        num_imputer = SimpleImputer(strategy="median", add_indicator=True)
        num_imputted = pd.DataFrame(num_imputer.fit_transform(imputing_num))
        num_imputted.columns = num_var
        num_imputted[num_var_suffix] = num_imputted[num_var_suffix].astype("category")

        imputed_data = pd.concat([cat_imputted, num_imputted], axis=1, sort=False)
        imputed_data = pd.concat([imputed_data, no_missing], axis=1, sort=False)

        # if drop == True:
        #     missing_list = self.summary_report[self.summary_report["missing_pct"] != 0]
        #     drop_list = imput_list.index.difference(missing_list.index)
        self.data = imputed_data
        self.summary()

    def missing_visualization(self):
        sns.heatmap(self.data.isnull(), cbar=False)

    def multicollinearity(self):
        # Calculating VIF
        nums = self.data._get_numeric_data()

        vif = pd.DataFrame()
        vif["factor"] = [variance_inflation_factor(nums.values, i) for i in range(nums.shape[1])]
        vif["features"] = nums.columns
        vif_list = vif[vif["factor"] >= 5]["features"]
        self.vif = vif

        nums = nums[vif_list]

        # Cluster the correlation matrix
        Corr = nums.corr()
        d = sch.distance.pdist(Corr.values)
        L = sch.linkage(d, method="complete")
        ind = sch.fcluster(L, 0.5 * d.max(), "distance")
        ind = ind.reshape(len(ind), -1)
        ind = np.concatenate((ind, np.arange(ind.shape[0]).reshape(ind.shape[0], -1)), axis=1)
        ind_sorted = ind[ind[:, 0].argsort()]
        columns = [nums.columns.tolist()[i] for i in list(ind_sorted[:, 1])]
        ind_sorted = pd.DataFrame(ind_sorted)
        ind_sorted.columns = ["clusters", "number"]
        ind_sorted["var"] = columns
        freq = ind_sorted["clusters"].value_counts()
        ind_sorted = ind_sorted.merge(freq, how="left", left_on="clusters", right_index=True)
        ind_sorted_noone = ind_sorted[ind_sorted["clusters_y"] != 1]

        # conduct non-parametric ANOVA to decide which variables need to be dropped
        cluster_list = np.unique(ind_sorted_noone["clusters_x"].values)
        drop_list = []
        for i in cluster_list:
            vars = ind_sorted_noone[ind_sorted_noone["clusters_x"] == i]["var"]
            corr = Corr.loc[vars, vars]
            corr = corr.where(np.triu(np.ones(corr.shape)).astype(np.bool)).stack().reset_index()
            cluster_num = np.ones(corr.shape[0]) * i
            cluster_num = cluster_num.reshape(corr.shape[0], -1)
            corr = np.concatenate([corr, cluster_num], axis=1)
            corr = pd.DataFrame(corr)
            corr.columns = ["row", "columns", "corr", "clusters"]
            corr = corr[corr["corr"] != 1]
            if corr.shape[0] == 1:
                value = np.array(corr["corr"])
                if value < 0.7:
                    continue
            uniques = np.unique(corr[["row", "columns"]].values)
            p_value = []
            for ii in uniques:
                x = self.data[self.data["TARGET"] == 1][ii]
                y = self.data[self.data["TARGET"] == 0][ii]
                test = stats.kruskal(x, y)
                p_value.append(test[1])

            min = [i for i, j in enumerate(p_value) if j == max(p_value)]
            drop = np.delete(uniques, min)
            for var in drop:
                drop_list.append(var)

        self.data.drop(drop_list, axis = 1, inplace = True)

    def vif_corr_map(self):
        nums = self.data._get_numeric_data()
        vif = pd.DataFrame()
        vif["factor"] = [variance_inflation_factor(nums.values, i) for i in range(nums.shape[1])]
        vif["features"] = nums.columns
        vif_list = vif[vif["factor"] >= 5]["features"]
        self.vif = vif
        nums = nums[vif_list]
        Corr = nums.corr()

        d = sch.distance.pdist(Corr.values)
        L = sch.linkage(d, method="complete")
        ind = sch.fcluster(L, 0.5 * d.max(), "distance")
        ind = ind.reshape(len(ind), -1)
        ind = np.concatenate((ind, np.arange(ind.shape[0]).reshape(ind.shape[0], -1)), axis=1)
        ind_sorted = ind[ind[:, 0].argsort()]
        columns = [nums.columns.tolist()[i] for i in list(ind_sorted[:, 1])]

        nums = nums.reindex(columns, axis = 1)
        Corr = nums.corr()
        fig, ax = plt.subplots(figsize=(10, 10))
        cax = ax.matshow(Corr, cmap="RdYlBu")
        plt.xticks(range(len(Corr.columns)), Corr.columns, rotation=90)
        plt.yticks(range(len(Corr.columns)), Corr.columns)
        cbar = fig.colorbar(cax, ticks=[-1, 0, 1], aspect=40, shrink=0.8)

    def get_target_freq(self, target):
        self.target_freq = self.data[target].value_counts()




#%%
app_class = Cleaning("/Users/dauku/Desktop/Python/homecredit/home-credit-default-risk/application_train.csv")
# app = app_class.data
app_class.summary()
summary_report = app_class.get_summary()
app_class.categorical()
app_class.imputation(0.5, drop = True)
app_class.vif_corr_map()
app_class.multicollinearity()

report = app_class.summary
data = app_class.data

#%%
# down sampling
Target = data.TARGET.value_counts()
Target = Target.rename(index = str)
dominating_class = Target.idxmax()
non_dominating_class = Target.idxmin()
sample_size = Target[Target.idxmin()]

sample_data = data[data["TARGET"] == int(Target.idxmax())]
non_sample_data = data[data["TARGET"] == int(Target.idxmin())]
sampled = data.sample(n = sample_size, random_state = 99)

sampled_data = pd.concat([sampled, non_sample_data], axis = 0, ignore_index = True)




#%%
# a = np.array([[1, 0.4, 0.8], [0.4, 1, 0.8], [0.4, 0.8, 1]])
# dis = sch.distance.pdist(a)
# L = sch.linkage(dis, method = "complete")
#
# dn = sch.dendrogram(L)
# plt.show()
#
#%%
# x = [[i] for i in [2, 8, 0, 4, 1, 9, 9, 0]]
# z = sch.linkage(x, "single")
# clus = sch.fcluster(z, 1, "inconsistent")
# dn = sch.dendrogram(z)
# plt.show()

#%% multicollinearity
nums = data._get_numeric_data()

vif = pd.DataFrame()
vif["factor"] = [variance_inflation_factor(nums.values, i) for i in range(nums.shape[1])]
vif["features"] = nums.columns
vif_list = vif[vif["factor"] >= 5]["features"]

nums = nums[vif_list]

def plot_corr(df, size = 10):

    # Compute the correlation matrix for the received dataframe
    corr = df.corr()

    # plot the correlation matrix
    fig, ax = plt.subplots(figsize = (size, size))
    cax = ax.matshow(corr, cmap = "RdYlBu")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation = 90)
    plt.yticks(range(len(corr.columns)), corr.columns)

    cbar = fig.colorbar(cax, ticks = [-1, 0, 1], aspect = 40, shrink = 0.8)

plot_corr(nums)

# Cluster the correlation matrix and visualize it
X = nums.corr().values
d = sch.distance.pdist(X)
L = sch.linkage(d, method = "complete")
ind = sch.fcluster(L, 0.5*d.max(), "distance")
ind = ind.reshape(len(ind), -1)
ind = np.concatenate((ind, np.arange(ind.shape[0]).reshape(ind.shape[0], -1)), axis = 1)
ind_sorted = ind[ind[:, 0].argsort()]
columns = [nums.columns.tolist()[i] for i in list(ind_sorted[:, 1])]

ind_sorted = pd.DataFrame(ind_sorted)
ind_sorted.columns = ["clusters", "number"]
ind_sorted["var"] = columns

freq = ind_sorted["clusters"].value_counts()
ind_sorted = ind_sorted.merge(freq, how = "left", left_on = "clusters", right_index = True)
ind_sorted_noone = ind_sorted[ind_sorted["clusters_y"] != 1]

Corr = nums.corr()

cluster_list = np.unique(ind_sorted_noone["clusters_x"].values)

# for j, i in enumerate(cluster_list):
#     vars = ind_sorted_noone[ind_sorted_noone["clusters_x"] == i]["var"]
#     corr = Corr.loc[vars, vars]
#     corr = corr.where(np.triu(np.ones(corr.shape)).astype(np.bool)).stack().reset_index()
#     cluster_num = np.ones(corr.shape[0])*i
#     cluster_num = cluster_num.reshape(corr.shape[0], -1)
#     corr = np.concatenate([corr, cluster_num], axis = 1)
#     corr = pd.DataFrame(corr)
#     corr.columns = ["row", "columns", "corr", "clusters"]
#     corr = corr[corr["corr"] != 1]
#     if j == 0:
#         corr_df = corr.copy()
#     else:
#         corr_df = pd.concat([corr_df, corr], ignore_index = True).drop_duplicates()

drop_list = []
for i in cluster_list:
    vars = ind_sorted_noone[ind_sorted_noone["clusters_x"] == i]["var"]
    corr = Corr.loc[vars, vars]
    corr = corr.where(np.triu(np.ones(corr.shape)).astype(np.bool)).stack().reset_index()
    cluster_num = np.ones(corr.shape[0]) * i
    cluster_num = cluster_num.reshape(corr.shape[0], -1)
    corr = np.concatenate([corr, cluster_num], axis=1)
    corr = pd.DataFrame(corr)
    corr.columns = ["row", "columns", "corr", "clusters"]
    corr = corr[corr["corr"] != 1]
    if corr.shape[0] == 1:
        value = np.array(corr["corr"])
        if value < 0.7:
            continue
    uniques = np.unique(corr[["row", "columns"]].values)
    p_value = []
    for ii in uniques:
        x = data[data["TARGET"] == 1][ii]
        y = data[data["TARGET"] == 0][ii]
        test = stats.kruskal(x, y)
        p_value.append(test[1])

    min = [i for i, j in enumerate(p_value) if j == max(p_value)]
    drop = np.delete(uniques, min)
    for var in drop:
        drop_list.append(var)

a = data.drop(drop_list, axis = 1)


second = ind_sorted[ind_sorted["clusters_x"] == 3]["var"]
Corr_second = Corr.loc[second, second]
Corr_second = Corr_second.where(np.triu(np.ones(Corr_second.shape)).astype(np.bool)).stack().reset_index()
Corr_second.columns = ["row", "column", "corr"]
Corr_second = Corr_second[Corr_second["corr"] != 1]


uniques = np.unique(Corr_second[["row", "column"]].values)
p_value = []
for i in uniques:
    x = data[data["TARGET"] == 1][i]
    y = data[data["TARGET"] == 0][i]
    test = stats.kruskal(x, y)
    p_value.append(test[1])

uniques = np.delete(uniques, 0)

# conducting anova to examine which vars has the better association with target
for i in (range(Corr_first.shape[0])):
    var1 = Corr_first.iloc[i, 0]
    var2 = Corr_first.iloc[i, 1]



df = nums.reindex(columns, axis = 1)

plot_corr(df, size = 6)




variables = list(range(nums.shape[1]))
vif2 = [variance_inflation_factor(nums.iloc[:, variables].values, ix) for ix in range(nums.iloc[:, variables].shape[1])]
# runa multiple regression
vif = pd.DataFrame()
vif["factor"] = [variance_inflation_factor(nums.values, i) for i in range(nums.shape[1])]
vif["features"] = nums.columns
vif_list = vif[vif["factor"] >= 5]["features"]
pear_corr = nums.corr(method = "pearson")
a = pear_corr.loc[vif_list, vif_list]
c = corrplot.Corrplot(a)
c.plot(method='color', cmap='PRGn_r', shrink=1, rotation=90, upper='color', lower='text')
fig = plt.gcf()
fig.set_size_inches(10, 8);

#%% Creating Summary report
def summary(data):
    uniques = data.nunique()
    dtypes = data.dtypes
    Missing = data.isnull().sum()

    summary = pd.DataFrame(uniques)
    summary.columns = ["uniques"]
    summary["dtypes"] = dtypes
    summary["missing"] = Missing
    summary["missing_pct"] = summary.missing / app.shape[0]

    return summary

#%% creating summary report
summary = summary(app)

#%% Data type check
def categorical(data, summary):
    binary_list = summary[summary["uniques"] == 2].index.tolist()
    data[binary_list] = data[binary_list].astype("category")

    object_list = summary[summary["dtypes"] == "object"].index.tolist()
    data[object_list] = data[object_list].astype("category")

#%%
categorical(app, summary)

#%% Imputing Missing Values
def imputation(data, summary, threshold):

    # vars that need imputation
    imput_list = summary[(summary["missing_pct"] < threshold) & (summary["missing_pct"] > 0)]
    imputing = data[imput_list.index]

    # vars that don't contain any missings
    no_missing_list = summary[summary["missing_pct"] == 0]
    no_missing = data[no_missing_list.index]

    # impute categorical variables
    imputing_cat = imputing.select_dtypes(exclude = "number")
    cat_var = imputing_cat.columns
    cat_imputer = SimpleImputer(strategy = "constant", fill_value = "Missing")
    cat_imputted = pd.DataFrame(cat_imputer.fit_transform(imputing_cat))
    cat_imputted.columns = cat_var

    # imputing numerical variables
    imputing_num = imputing.select_dtypes(include="number")
    num_var = imputing_num.columns.tolist()
    num_var_suffix = [x + "_indicator" for x in num_var]
    num_var = num_var + num_var_suffix
    num_imputer = SimpleImputer(strategy = "median", add_indicator = True)
    num_imputted = pd.DataFrame(num_imputer.fit_transform(imputing_num))
    num_imputted.columns = num_var

    imputed_data = pd.concat([cat_imputted, num_imputted], axis = 1, sort = False)
    imputed_data = pd.concat([imputed_data, no_missing], axis = 1, sort = False)

    return imputed_data
#%%
imputed_data = imputation(app, summary, 0.5)

#%%
# visulizing missing position
msno.matrix(app)
plt.show()

sns.heatmap(imputing.isnull(), cbar = False)

# imputing categorical variables by adding one level for missing

#%%
test = imputing.select_dtypes(exclude = "number").columns
imputer_cat = SimpleImputer(strategy = "constant", fill_value = "Missing")
test2 = pd.DataFrame(imputer_cat.fit_transform(imputing[test]))

test3 = imputing.select_dtypes(include = "number").columns
imputer_num = SimpleImputer(strategy = "median", add_indicator=True)
input = imputing[test3].values
test4 = pd.DataFrame(imputer_num.fit_transform(input))
test5 = imputing[test3]

imputer_num = FeatureUnion(transformer_list = [("features", SimpleImputer(strategy = "median")),
                                               ("indicators", MissingIndicator())])
results = imputer_num.fit_transform(imputing[test3])

#%%

