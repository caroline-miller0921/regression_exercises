from env import username as u, password as p, host as h, url
import pandas as pd
import os

def get_zillow_2017(schema, u=u, p=p, h=h):
    '''
    get_sql_url will pull the credentials present from any current env
    file in the same directory as this acquire script
    and will return a connection based on what schema and databases 
    (db) are handed to the function call
    '''

    if os.path.isfile('zillow_2017.csv'):

        return pd.read_csv('zillow_2017.csv')
    
    else:

        url = f'mysql+pymysql://{u}:{p}@{h}/{schema}'

        df = pd.read_sql(f'select bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, taxamount, fips, propertylandusetypeid from properties_2017 left join propertylandusetype using (propertylandusetypeid) where propertylandusetypeid <=> 261;', url)

        df.to_csv('zillow_2017.csv')

        return df

def wrangle_zillow():

    df = pd.read_csv('zillow_2017.csv')

    # dropping all rows that have 0 bedrooms or 0 bathrooms
    df = df[(df.bedroomcnt != 0) | (df.bathroomcnt != 0)]

    df = df.dropna()

    df.yearbuilt = df.yearbuilt.astype(int)
    df.bedroomcnt = df.bedroomcnt.astype(int)
    df.bathroomcnt = df.bathroomcnt.astype(int)
    df['fips_location'] = df.fips
    df.fips_location = df.fips_location.replace(6037.0, 'Los Angeles County')
    df.fips_location = df.fips_location.replace(6059.0, 'Orange County')
    df.fips_location = df.fips_location.replace(6111.0, 'Colusa County')
    df.fips = df.fips.astype(int)
    df.propertylandusetypeid = df.propertylandusetypeid.astype(int)
    df.rename(columns={'bedroomcnt': 'bedrooms', 'bathroomcnt': 'bathrooms', 'calculatedfinishedsquarefeet': 'squarefeet', 'taxvaluedollarcnt': 'taxvalue', 'propertylandusetypeid': 'propertytypeid'}, inplace=True)
    df = df.drop(columns=('Unnamed: 0'))
    return df
