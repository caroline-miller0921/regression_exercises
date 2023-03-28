# inside acquire.py script:
from env import username as u, password as p, host as h, url
import pandas as pd
import os
db = ''

def get_sql_url(schema, db, u=u, p=p, h=h):
    '''
    get_sql_url will pull the credentials present from any current env
    file in the same directory as this acquire script
    and will return a connection based on what schema and databases 
    (db) are handed to the function call
    '''
    url = f'mysql+pymysql://{u}:{p}@{h}/{schema}'
    return pd.read_sql(f'select * from {db};', url)

def get_connection(schema, u=u, p=p, h=h):
    '''
    get_sql_url will pull the credentials present from any current env
    file in the same directory as this acquire script
    and will return a connection based on what schema and databases 
    (db) are handed to the function call
    '''
    return f'mysql+pymysql://{u}:{p}@{h}/{schema}'

def get_titanic_data(u=u, p=p, h=h):
    '''
    get_titanic_data will check to see if there is a csv locally saved. If there 
    is not, the function will pull the credentials present from any current env
    file in the same directory as this acquire script and will return a connection 
    based on what schema as handed to the function call
    '''
    if os.path.exists('titanic.csv'):
        df = pd.read_csv('titanic.csv', index_col=0)
    else:
        query = 'select * from passengers;'
        connection = get_connection('titanic_db')
        df = pd.read_sql(query, connection)
        df.to_csv('titanic.csv')
    return df

def get_iris_data(u=u, p=p, h=h):
    '''
    get_iris_data will check to see if there is a csv locally saved. If there 
    is not, the function will pull the credentials present from any current env
    file in the same directory as this acquire script and will return a connection 
    based on what schema as handed to the function call
    '''
    if os.path.exists('iris.csv'):
        df = pd.read_csv('iris.csv', index_col=0)
    else:
        query = '''
            select * 
            from measurements
                join species using (species_id);
            '''
        connection = get_connection('iris_db')
        df = pd.read_sql(query, connection)
        df.to_csv('iris.csv')
    return df

def get_telco_data(u=u, p=p, h=h):
    '''
    get_telco_data will check to see if there is a csv locally saved. If there 
    is not, the function will pull the credentials present from any current env
    file in the same directory as this acquire script and will return a connection 
    based on what schema as handed to the function call
    '''
    if os.path.exists('telco.csv'):
        df = pd.read_csv('telco.csv', index_col=0)
    else:
        query = '''
            select * 
            from customers
                left join contract_types using (contract_type_id)
                left join internet_service_types using (internet_service_type_id)
                left join payment_types using (payment_type_id)
                left join customer_churn using (customer_id)
                left join customer_signups using (customer_id);
            '''
        connection = get_connection('telco_churn')
        df = pd.read_sql(query, connection)
        df.to_csv('telco.csv')
    return df