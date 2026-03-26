import numpy as np
import pandas as pd


def GetDataset(name, base_path):
    """ Load a dataset
    
    Parameters
    ----------
    name : string, dataset name
    base_path : string, e.g. "path/to/datasets/directory/"
    
    Returns
    -------
    X : features (nXp)
    y : labels (n)
    
	"""
    if name == "meps19":
        df = pd.read_csv(base_path + 'meps_19_reg.csv')
        column_names = df.columns
        response_name = "UTILIZATION_reg"
        column_names = column_names[column_names != response_name]
        column_names = column_names[column_names != "Unnamed: 0"]

        col_names = ['AGE', 'PCS42', 'MCS42', 'K6SUM42', 'PERWT15F', 'REGION=1',
                     'REGION=2', 'REGION=3', 'REGION=4', 'SEX=1', 'SEX=2', 'MARRY=1',
                     'MARRY=2', 'MARRY=3', 'MARRY=4', 'MARRY=5', 'MARRY=6', 'MARRY=7',
                     'MARRY=8', 'MARRY=9', 'MARRY=10', 'FTSTU=-1', 'FTSTU=1', 'FTSTU=2',
                     'FTSTU=3', 'ACTDTY=1', 'ACTDTY=2', 'ACTDTY=3', 'ACTDTY=4',
                     'HONRDC=1', 'HONRDC=2', 'HONRDC=3', 'HONRDC=4', 'RTHLTH=-1',
                     'RTHLTH=1', 'RTHLTH=2', 'RTHLTH=3', 'RTHLTH=4', 'RTHLTH=5',
                     'MNHLTH=-1', 'MNHLTH=1', 'MNHLTH=2', 'MNHLTH=3', 'MNHLTH=4',
                     'MNHLTH=5', 'HIBPDX=-1', 'HIBPDX=1', 'HIBPDX=2', 'CHDDX=-1',
                     'CHDDX=1', 'CHDDX=2', 'ANGIDX=-1', 'ANGIDX=1', 'ANGIDX=2',
                     'MIDX=-1', 'MIDX=1', 'MIDX=2', 'OHRTDX=-1', 'OHRTDX=1', 'OHRTDX=2',
                     'STRKDX=-1', 'STRKDX=1', 'STRKDX=2', 'EMPHDX=-1', 'EMPHDX=1',
                     'EMPHDX=2', 'CHBRON=-1', 'CHBRON=1', 'CHBRON=2', 'CHOLDX=-1',
                     'CHOLDX=1', 'CHOLDX=2', 'CANCERDX=-1', 'CANCERDX=1', 'CANCERDX=2',
                     'DIABDX=-1', 'DIABDX=1', 'DIABDX=2', 'JTPAIN=-1', 'JTPAIN=1',
                     'JTPAIN=2', 'ARTHDX=-1', 'ARTHDX=1', 'ARTHDX=2', 'ARTHTYPE=-1',
                     'ARTHTYPE=1', 'ARTHTYPE=2', 'ARTHTYPE=3', 'ASTHDX=1', 'ASTHDX=2',
                     'ADHDADDX=-1', 'ADHDADDX=1', 'ADHDADDX=2', 'PREGNT=-1', 'PREGNT=1',
                     'PREGNT=2', 'WLKLIM=-1', 'WLKLIM=1', 'WLKLIM=2', 'ACTLIM=-1',
                     'ACTLIM=1', 'ACTLIM=2', 'SOCLIM=-1', 'SOCLIM=1', 'SOCLIM=2',
                     'COGLIM=-1', 'COGLIM=1', 'COGLIM=2', 'DFHEAR42=-1', 'DFHEAR42=1',
                     'DFHEAR42=2', 'DFSEE42=-1', 'DFSEE42=1', 'DFSEE42=2',
                     'ADSMOK42=-1', 'ADSMOK42=1', 'ADSMOK42=2', 'PHQ242=-1', 'PHQ242=0',
                     'PHQ242=1', 'PHQ242=2', 'PHQ242=3', 'PHQ242=4', 'PHQ242=5',
                     'PHQ242=6', 'EMPST=-1', 'EMPST=1', 'EMPST=2', 'EMPST=3', 'EMPST=4',
                     'POVCAT=1', 'POVCAT=2', 'POVCAT=3', 'POVCAT=4', 'POVCAT=5',
                     'INSCOV=1', 'INSCOV=2', 'INSCOV=3', 'RACE']

        y = df[response_name].values
        X = df[col_names].values

    if name == "meps20":
        df = pd.read_csv(base_path + 'meps_20_reg.csv')
        column_names = df.columns
        response_name = "UTILIZATION_reg"
        column_names = column_names[column_names != response_name]
        column_names = column_names[column_names != "Unnamed: 0"]

        col_names = ['AGE', 'PCS42', 'MCS42', 'K6SUM42', 'PERWT15F', 'REGION=1',
                     'REGION=2', 'REGION=3', 'REGION=4', 'SEX=1', 'SEX=2', 'MARRY=1',
                     'MARRY=2', 'MARRY=3', 'MARRY=4', 'MARRY=5', 'MARRY=6', 'MARRY=7',
                     'MARRY=8', 'MARRY=9', 'MARRY=10', 'FTSTU=-1', 'FTSTU=1', 'FTSTU=2',
                     'FTSTU=3', 'ACTDTY=1', 'ACTDTY=2', 'ACTDTY=3', 'ACTDTY=4',
                     'HONRDC=1', 'HONRDC=2', 'HONRDC=3', 'HONRDC=4', 'RTHLTH=-1',
                     'RTHLTH=1', 'RTHLTH=2', 'RTHLTH=3', 'RTHLTH=4', 'RTHLTH=5',
                     'MNHLTH=-1', 'MNHLTH=1', 'MNHLTH=2', 'MNHLTH=3', 'MNHLTH=4',
                     'MNHLTH=5', 'HIBPDX=-1', 'HIBPDX=1', 'HIBPDX=2', 'CHDDX=-1',
                     'CHDDX=1', 'CHDDX=2', 'ANGIDX=-1', 'ANGIDX=1', 'ANGIDX=2',
                     'MIDX=-1', 'MIDX=1', 'MIDX=2', 'OHRTDX=-1', 'OHRTDX=1', 'OHRTDX=2',
                     'STRKDX=-1', 'STRKDX=1', 'STRKDX=2', 'EMPHDX=-1', 'EMPHDX=1',
                     'EMPHDX=2', 'CHBRON=-1', 'CHBRON=1', 'CHBRON=2', 'CHOLDX=-1',
                     'CHOLDX=1', 'CHOLDX=2', 'CANCERDX=-1', 'CANCERDX=1', 'CANCERDX=2',
                     'DIABDX=-1', 'DIABDX=1', 'DIABDX=2', 'JTPAIN=-1', 'JTPAIN=1',
                     'JTPAIN=2', 'ARTHDX=-1', 'ARTHDX=1', 'ARTHDX=2', 'ARTHTYPE=-1',
                     'ARTHTYPE=1', 'ARTHTYPE=2', 'ARTHTYPE=3', 'ASTHDX=1', 'ASTHDX=2',
                     'ADHDADDX=-1', 'ADHDADDX=1', 'ADHDADDX=2', 'PREGNT=-1', 'PREGNT=1',
                     'PREGNT=2', 'WLKLIM=-1', 'WLKLIM=1', 'WLKLIM=2', 'ACTLIM=-1',
                     'ACTLIM=1', 'ACTLIM=2', 'SOCLIM=-1', 'SOCLIM=1', 'SOCLIM=2',
                     'COGLIM=-1', 'COGLIM=1', 'COGLIM=2', 'DFHEAR42=-1', 'DFHEAR42=1',
                     'DFHEAR42=2', 'DFSEE42=-1', 'DFSEE42=1', 'DFSEE42=2',
                     'ADSMOK42=-1', 'ADSMOK42=1', 'ADSMOK42=2', 'PHQ242=-1', 'PHQ242=0',
                     'PHQ242=1', 'PHQ242=2', 'PHQ242=3', 'PHQ242=4', 'PHQ242=5',
                     'PHQ242=6', 'EMPST=-1', 'EMPST=1', 'EMPST=2', 'EMPST=3', 'EMPST=4',
                     'POVCAT=1', 'POVCAT=2', 'POVCAT=3', 'POVCAT=4', 'POVCAT=5',
                     'INSCOV=1', 'INSCOV=2', 'INSCOV=3', 'RACE']

        y = df[response_name].values
        X = df[col_names].values

    if name == "meps21":
        df = pd.read_csv(base_path + 'meps_21_reg.csv')
        column_names = df.columns
        response_name = "UTILIZATION_reg"
        column_names = column_names[column_names != response_name]
        column_names = column_names[column_names != "Unnamed: 0"]

        col_names = ['AGE', 'PCS42', 'MCS42', 'K6SUM42', 'PERWT16F', 'REGION=1',
                     'REGION=2', 'REGION=3', 'REGION=4', 'SEX=1', 'SEX=2', 'MARRY=1',
                     'MARRY=2', 'MARRY=3', 'MARRY=4', 'MARRY=5', 'MARRY=6', 'MARRY=7',
                     'MARRY=8', 'MARRY=9', 'MARRY=10', 'FTSTU=-1', 'FTSTU=1', 'FTSTU=2',
                     'FTSTU=3', 'ACTDTY=1', 'ACTDTY=2', 'ACTDTY=3', 'ACTDTY=4',
                     'HONRDC=1', 'HONRDC=2', 'HONRDC=3', 'HONRDC=4', 'RTHLTH=-1',
                     'RTHLTH=1', 'RTHLTH=2', 'RTHLTH=3', 'RTHLTH=4', 'RTHLTH=5',
                     'MNHLTH=-1', 'MNHLTH=1', 'MNHLTH=2', 'MNHLTH=3', 'MNHLTH=4',
                     'MNHLTH=5', 'HIBPDX=-1', 'HIBPDX=1', 'HIBPDX=2', 'CHDDX=-1',
                     'CHDDX=1', 'CHDDX=2', 'ANGIDX=-1', 'ANGIDX=1', 'ANGIDX=2',
                     'MIDX=-1', 'MIDX=1', 'MIDX=2', 'OHRTDX=-1', 'OHRTDX=1', 'OHRTDX=2',
                     'STRKDX=-1', 'STRKDX=1', 'STRKDX=2', 'EMPHDX=-1', 'EMPHDX=1',
                     'EMPHDX=2', 'CHBRON=-1', 'CHBRON=1', 'CHBRON=2', 'CHOLDX=-1',
                     'CHOLDX=1', 'CHOLDX=2', 'CANCERDX=-1', 'CANCERDX=1', 'CANCERDX=2',
                     'DIABDX=-1', 'DIABDX=1', 'DIABDX=2', 'JTPAIN=-1', 'JTPAIN=1',
                     'JTPAIN=2', 'ARTHDX=-1', 'ARTHDX=1', 'ARTHDX=2', 'ARTHTYPE=-1',
                     'ARTHTYPE=1', 'ARTHTYPE=2', 'ARTHTYPE=3', 'ASTHDX=1', 'ASTHDX=2',
                     'ADHDADDX=-1', 'ADHDADDX=1', 'ADHDADDX=2', 'PREGNT=-1', 'PREGNT=1',
                     'PREGNT=2', 'WLKLIM=-1', 'WLKLIM=1', 'WLKLIM=2', 'ACTLIM=-1',
                     'ACTLIM=1', 'ACTLIM=2', 'SOCLIM=-1', 'SOCLIM=1', 'SOCLIM=2',
                     'COGLIM=-1', 'COGLIM=1', 'COGLIM=2', 'DFHEAR42=-1', 'DFHEAR42=1',
                     'DFHEAR42=2', 'DFSEE42=-1', 'DFSEE42=1', 'DFSEE42=2',
                     'ADSMOK42=-1', 'ADSMOK42=1', 'ADSMOK42=2', 'PHQ242=-1', 'PHQ242=0',
                     'PHQ242=1', 'PHQ242=2', 'PHQ242=3', 'PHQ242=4', 'PHQ242=5',
                     'PHQ242=6', 'EMPST=-1', 'EMPST=1', 'EMPST=2', 'EMPST=3', 'EMPST=4',
                     'POVCAT=1', 'POVCAT=2', 'POVCAT=3', 'POVCAT=4', 'POVCAT=5',
                     'INSCOV=1', 'INSCOV=2', 'INSCOV=3', 'RACE']

        y = df[response_name].values
        X = df[col_names].values

    if name == "star":
        df = pd.read_csv(base_path + 'STAR.csv')
        df.loc[df['gender'] == 'female', 'gender'] = 0
        df.loc[df['gender'] == 'male', 'gender'] = 1

        df.loc[df['ethnicity'] == 'cauc', 'ethnicity'] = 0
        df.loc[df['ethnicity'] == 'afam', 'ethnicity'] = 1
        df.loc[df['ethnicity'] == 'asian', 'ethnicity'] = 2
        df.loc[df['ethnicity'] == 'hispanic', 'ethnicity'] = 3
        df.loc[df['ethnicity'] == 'amindian', 'ethnicity'] = 4
        df.loc[df['ethnicity'] == 'other', 'ethnicity'] = 5

        df.loc[df['stark'] == 'regular', 'stark'] = 0
        df.loc[df['stark'] == 'small', 'stark'] = 1
        df.loc[df['stark'] == 'regular+aide', 'stark'] = 2

        df.loc[df['star1'] == 'regular', 'star1'] = 0
        df.loc[df['star1'] == 'small', 'star1'] = 1
        df.loc[df['star1'] == 'regular+aide', 'star1'] = 2

        df.loc[df['star2'] == 'regular', 'star2'] = 0
        df.loc[df['star2'] == 'small', 'star2'] = 1
        df.loc[df['star2'] == 'regular+aide', 'star2'] = 2

        df.loc[df['star3'] == 'regular', 'star3'] = 0
        df.loc[df['star3'] == 'small', 'star3'] = 1
        df.loc[df['star3'] == 'regular+aide', 'star3'] = 2

        df.loc[df['lunchk'] == 'free', 'lunchk'] = 0
        df.loc[df['lunchk'] == 'non-free', 'lunchk'] = 1

        df.loc[df['lunch1'] == 'free', 'lunch1'] = 0
        df.loc[df['lunch1'] == 'non-free', 'lunch1'] = 1

        df.loc[df['lunch2'] == 'free', 'lunch2'] = 0
        df.loc[df['lunch2'] == 'non-free', 'lunch2'] = 1

        df.loc[df['lunch3'] == 'free', 'lunch3'] = 0
        df.loc[df['lunch3'] == 'non-free', 'lunch3'] = 1

        df.loc[df['schoolk'] == 'inner-city', 'schoolk'] = 0
        df.loc[df['schoolk'] == 'suburban', 'schoolk'] = 1
        df.loc[df['schoolk'] == 'rural', 'schoolk'] = 2
        df.loc[df['schoolk'] == 'urban', 'schoolk'] = 3

        df.loc[df['school1'] == 'inner-city', 'school1'] = 0
        df.loc[df['school1'] == 'suburban', 'school1'] = 1
        df.loc[df['school1'] == 'rural', 'school1'] = 2
        df.loc[df['school1'] == 'urban', 'school1'] = 3

        df.loc[df['school2'] == 'inner-city', 'school2'] = 0
        df.loc[df['school2'] == 'suburban', 'school2'] = 1
        df.loc[df['school2'] == 'rural', 'school2'] = 2
        df.loc[df['school2'] == 'urban', 'school2'] = 3

        df.loc[df['school3'] == 'inner-city', 'school3'] = 0
        df.loc[df['school3'] == 'suburban', 'school3'] = 1
        df.loc[df['school3'] == 'rural', 'school3'] = 2
        df.loc[df['school3'] == 'urban', 'school3'] = 3

        df.loc[df['degreek'] == 'bachelor', 'degreek'] = 0
        df.loc[df['degreek'] == 'master', 'degreek'] = 1
        df.loc[df['degreek'] == 'specialist', 'degreek'] = 2
        df.loc[df['degreek'] == 'master+', 'degreek'] = 3

        df.loc[df['degree1'] == 'bachelor', 'degree1'] = 0
        df.loc[df['degree1'] == 'master', 'degree1'] = 1
        df.loc[df['degree1'] == 'specialist', 'degree1'] = 2
        df.loc[df['degree1'] == 'phd', 'degree1'] = 3

        df.loc[df['degree2'] == 'bachelor', 'degree2'] = 0
        df.loc[df['degree2'] == 'master', 'degree2'] = 1
        df.loc[df['degree2'] == 'specialist', 'degree2'] = 2
        df.loc[df['degree2'] == 'phd', 'degree2'] = 3

        df.loc[df['degree3'] == 'bachelor', 'degree3'] = 0
        df.loc[df['degree3'] == 'master', 'degree3'] = 1
        df.loc[df['degree3'] == 'specialist', 'degree3'] = 2
        df.loc[df['degree3'] == 'phd', 'degree3'] = 3

        df.loc[df['ladderk'] == 'level1', 'ladderk'] = 0
        df.loc[df['ladderk'] == 'level2', 'ladderk'] = 1
        df.loc[df['ladderk'] == 'level3', 'ladderk'] = 2
        df.loc[df['ladderk'] == 'apprentice', 'ladderk'] = 3
        df.loc[df['ladderk'] == 'probation', 'ladderk'] = 4
        df.loc[df['ladderk'] == 'pending', 'ladderk'] = 5
        df.loc[df['ladderk'] == 'notladder', 'ladderk'] = 6

        df.loc[df['ladder1'] == 'level1', 'ladder1'] = 0
        df.loc[df['ladder1'] == 'level2', 'ladder1'] = 1
        df.loc[df['ladder1'] == 'level3', 'ladder1'] = 2
        df.loc[df['ladder1'] == 'apprentice', 'ladder1'] = 3
        df.loc[df['ladder1'] == 'probation', 'ladder1'] = 4
        df.loc[df['ladder1'] == 'noladder', 'ladder1'] = 5
        df.loc[df['ladder1'] == 'notladder', 'ladder1'] = 6

        df.loc[df['ladder2'] == 'level1', 'ladder2'] = 0
        df.loc[df['ladder2'] == 'level2', 'ladder2'] = 1
        df.loc[df['ladder2'] == 'level3', 'ladder2'] = 2
        df.loc[df['ladder2'] == 'apprentice', 'ladder2'] = 3
        df.loc[df['ladder2'] == 'probation', 'ladder2'] = 4
        df.loc[df['ladder2'] == 'noladder', 'ladder2'] = 5
        df.loc[df['ladder2'] == 'notladder', 'ladder2'] = 6

        df.loc[df['ladder3'] == 'level1', 'ladder3'] = 0
        df.loc[df['ladder3'] == 'level2', 'ladder3'] = 1
        df.loc[df['ladder3'] == 'level3', 'ladder3'] = 2
        df.loc[df['ladder3'] == 'apprentice', 'ladder3'] = 3
        df.loc[df['ladder3'] == 'probation', 'ladder3'] = 4
        df.loc[df['ladder3'] == 'noladder', 'ladder3'] = 5
        df.loc[df['ladder3'] == 'notladder', 'ladder3'] = 6

        df.loc[df['tethnicityk'] == 'cauc', 'tethnicityk'] = 0
        df.loc[df['tethnicityk'] == 'afam', 'tethnicityk'] = 1

        df.loc[df['tethnicity1'] == 'cauc', 'tethnicity1'] = 0
        df.loc[df['tethnicity1'] == 'afam', 'tethnicity1'] = 1

        df.loc[df['tethnicity2'] == 'cauc', 'tethnicity2'] = 0
        df.loc[df['tethnicity2'] == 'afam', 'tethnicity2'] = 1

        df.loc[df['tethnicity3'] == 'cauc', 'tethnicity3'] = 0
        df.loc[df['tethnicity3'] == 'afam', 'tethnicity3'] = 1
        df.loc[df['tethnicity3'] == 'asian', 'tethnicity3'] = 2

        df = df.dropna()

        grade = df["readk"] + df["read1"] + df["read2"] + df["read3"]
        grade += df["mathk"] + df["math1"] + df["math2"] + df["math3"]

        names = df.columns
        target_names = names[8:16]
        data_names = np.concatenate((names[0:8], names[17:]))
        X = df.loc[:, data_names].values
        y = grade.values

    if name == "fb1":
        df = pd.read_csv(base_path + 'facebook/Features_Variant_1.csv')
        y = df.iloc[:, 53].values
        X = df.iloc[:, 0:53].values

    if name == "fb2":
        df = pd.read_csv(base_path + 'facebook/Features_Variant_2.csv')
        y = df.iloc[:, 53].values
        X = df.iloc[:, 0:53].values

    if name == "bio":
        # https://github.com/joefavergel/TertiaryPhysicochemicalProperties/blob/master/RMSD-ProteinTertiaryStructures.ipynb
        df = pd.read_csv(base_path + 'CASP.csv')
        y = df.iloc[:, 0].values
        X = df.iloc[:, 1:].values

    if name == 'blog':
        # https://github.com/xinbinhuang/feature-selection_blogfeedback
        df = pd.read_csv(base_path + 'blogData_train.csv', header=None)
        X = df.iloc[:, 0:280].values
        y = df.iloc[:, -1].values

    if name == "bike":
        # https://www.kaggle.com/rajmehra03/bike-sharing-demand-rmsle-0-3194
        df = pd.read_csv(base_path + 'bike_train.csv')

        # # seperating season as per values. this is bcoz this will enhance features.
        season = pd.get_dummies(df['season'], prefix='season')
        df = pd.concat([df, season], axis=1)

        # # # same for weather. this is bcoz this will enhance features.
        weather = pd.get_dummies(df['weather'], prefix='weather')
        df = pd.concat([df, weather], axis=1)

        # # # now can drop weather and season.
        df.drop(['season', 'weather'], inplace=True, axis=1)
        df.head()

        df["hour"] = [t.hour for t in pd.DatetimeIndex(df.datetime)]
        df["day"] = [t.dayofweek for t in pd.DatetimeIndex(df.datetime)]
        df["month"] = [t.month for t in pd.DatetimeIndex(df.datetime)]
        df['year'] = [t.year for t in pd.DatetimeIndex(df.datetime)]
        df['year'] = df['year'].map({2011: 0, 2012: 1})

        df.drop('datetime', axis=1, inplace=True)
        df.drop(['casual', 'registered'], axis=1, inplace=True)
        df.columns.to_series().groupby(df.dtypes).groups
        X = df.drop('count', axis=1).values
        y = df['count'].values

    if name == "com":
        # https://github.com/vbordalo/Communities-Crime/blob/master/Crime_v1.ipynb
        attrib = pd.read_csv(base_path + 'communities_attributes.csv', delim_whitespace=True)
        data = pd.read_csv(base_path + 'communities.data', names=attrib['attributes'])
        data = data.drop(columns=['state', 'county',
                                  'community', 'communityname',
                                  'fold'], axis=1)

        data = data.replace('?', np.nan)

        # Impute mean values for samples with missing values
        # from sklearn.preprocessing import Imputer
        # imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)

        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

        imputer = imputer.fit(data[['OtherPerCap']])
        data[['OtherPerCap']] = imputer.transform(data[['OtherPerCap']])
        data = data.dropna(axis=1)
        X = data.iloc[:, 0:100].values
        y = data.iloc[:, 100].values

    if name == "x100-y10-reg":
        # generate a random dataset
        num_samples = 10000
        X = np.random.random((num_samples, 100))
        W = np.random.random((100, 10))
        y = X.dot(W) + np.random.randn(num_samples, 10)

    if name == "x100-y1-reg":
        # generate a random dataset
        num_samples = 10000
        X = np.random.random((num_samples, 100))
        W = np.random.random((100, 1))
        y = X.dot(W) + np.random.randn(num_samples, 1)

    X = X.astype(np.float32)
    y = y.astype(np.float32)

    return X, y
