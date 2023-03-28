from env import username as u, password as p, host as h, url
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer


def get_zillow_2017():
    '''
    get_sql_url will pull the credentials present from any current env
    file in the same directory as this acquire script
    and will return a connection based on what schema and databases 
    (db) are handed to the function call
    '''

    if os.path.isfile('zillow_2017.csv'):

        return pd.read_csv('zillow_2017.csv')
    
    else:

        url = f'mysql+pymysql://{u}:{p}@{h}/zillow'

        df = pd.read_sql(f'select bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, taxamount, fips, propertylandusetypeid from properties_2017 left join propertylandusetype using (propertylandusetypeid) where propertylandusetypeid <=> 261 or propertylandusetypeid <=> 279;', url)

        df.to_csv('zillow_2017.csv')

        return df



def remove_outliers(df, df_cols, k=1.5):
    col_qs = {}
    
    for col in df_cols:
        col_qs[col] = q1, q3 = df[col].quantile([0.25, 0.75])
        # print(col_qs)
    
    for col in df_cols:    
        iqr = col_qs[col][0.75] - col_qs[col][0.25]
        lower_fence = col_qs[col][0.25] - (iqr*k)
        upper_fence = col_qs[col][0.75] + (iqr*k)
        #print(f'Lower fence of {col}: {lower_fence}')
        #print(f'Upper fence of {col}: {upper_fence}')
        df = df[(df[col] > lower_fence) & (df[col] < upper_fence)]
    return df


def get_hist(df):
    ''' Gets histographs of acquired continuous variables'''
    
    plt.figure(figsize=(16, 3))

    # List of columns
    cols = [col for col in df.columns if col not in ['fips', 'fips_location', 'propertytypeid']]

    for i, col in enumerate(cols):

        # i starts at 0, but plot nos should start at 1
        plot_number = i + 1 

        # Create subplot.
        plt.subplot(1, len(cols), plot_number)

        # Title with column name.
        plt.title(col)

        colors = ['mediumvioletred']

        # Display histogram for column.
        plt.hist(x=df[col], bins=5 , color = colors, edgecolor = 'black')

        # Hide gridlines.
        plt.grid(False)

        # turn off scientific notation
        plt.ticklabel_format(useOffset=False)

        plt.tight_layout()

    plt.show()


def get_box(df):
    ''' Gets boxplots of acquired continuous variables'''
    
    # List of columns
    cols = ['bedrooms', 'bathrooms', 'squarefeet', 'tax_value', 'taxamount', 'yearbuilt']

    plt.figure(figsize=(16, 3))

    for i, col in enumerate(cols):

        # i starts at 0, but plot should start at 1
        plot_number = i + 1 

        # Create subplot.
        plt.subplot(1, len(cols), plot_number)

        # Title with column name.
        plt.title(col)

        # Display boxplot for column.
        sns.boxplot(data=df[[col]], palette='magma')

        # Hide gridlines.
        plt.grid(False)

        # sets proper spacing between plots
        plt.tight_layout()

    plt.show()


def get_count(df):

    plt.figure(figsize=(16, 3))

    sns.countplot(data=df, x='fips_location', palette='magma')
    plt.xlabel('County Name')
    plt.ylabel('Count (mil)')
    plt.title(f'Propery Count by FIPS Codes')
    plt.show()


def prepare_zillow(df):

    df = pd.read_csv('zillow_2017.csv')

    # dropping all rows that have 0 bedrooms or 0 bathrooms
    df = df[(df.bedroomcnt != 0) | (df.bathroomcnt != 0)]

    df.rename(columns={'bedroomcnt': 'bedrooms',
                        'bathroomcnt': 'bathrooms', 
                        'calculatedfinishedsquarefeet': 'squarefeet', 
                        'taxvaluedollarcnt': 'tax_value', 
                        'propertylandusetypeid': 'propertytypeid'}, 
                        inplace=True)
    
    df = df.drop(columns=('Unnamed: 0'))

    df = remove_outliers(df, ['bedrooms', 'bathrooms', 'squarefeet', 'tax_value', 'taxamount'])

    df['fips_location'] = df.fips
    df.fips_location = df.fips_location.replace(6037.0, 'Los Angeles County')
    df.fips_location = df.fips_location.replace(6059.0, 'Orange County')
    df.fips_location = df.fips_location.replace(6111.0, 'Colusa County')

    df.fips = df.fips.astype(int)

    df.propertytypeid = df.propertytypeid.astype(int)

    get_hist(df)
    get_box(df)
    get_count(df)

    # train/validate/test split
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123)

    # impute year built using mode
    imputer = SimpleImputer(strategy='median')

    imputer.fit(train[['yearbuilt']])

    train[['yearbuilt']] = imputer.transform(train[['yearbuilt']])
    validate[['yearbuilt']] = imputer.transform(validate[['yearbuilt']])
    test[['yearbuilt']] = imputer.transform(test[['yearbuilt']])  
    
    return train, validate, test

def wrangle_zillow():

    train, validate, test = prepare_zillow(get_zillow_2017())

    return train, validate, test