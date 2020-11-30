'''
This is the second script used to grab the csv files
and merge/clean for the final data frame to be used
in visualization.

@author Amir Hallak
'''

###########################################################
###########################################################

# modules import
import pandas as pd

###########################################################
###########################################################

# paths
ESG_SUB = '../dataframes/esg_sub.csv'
ESG_COM = '../dataframes/esg_com.csv'

ETHICAL_SUB = '../dataframes/ethical_investing_sub.csv'
ETHICAL_COM = '../dataframes/ethical_investing_com.csv'

GREEN_SUB = '../dataframes/greenwashing_sub.csv'
GREEN_COM = '../dataframes/greenwashing_com.csv'

RESP_SUB = '../dataframes/responsible_investing_sub.csv'
RESP_COM = '../dataframes/responsible_investing_com.csv'

SUST_SUB = '../dataframes/sustainable_investing_sub.csv'
SUST_COM = '../dataframes/sustainable_investing_com.csv'

IMPACT_SUB = '../dataframes/impact_investing_sub.csv'
IMPACT_COM = '../dataframes/impact_investing_com.csv'

GREEN_SUB = '../dataframes/greenwashing_sub.csv'
GREEN_COM = '../dataframes/greenwashing_com.csv'

CENO_SUB = '../dataframes/cenovus_sub.csv'
CENO_COM = '../dataframes/cenovus_com.csv'

SUNCOR_SUB = '../dataframes/suncor_sub.csv'
SUNCOR_COM = '../dataframes/suncor_com.csv'

ENBRIDGE_SUB = '../dataframes/enbridge_sub.csv'
ENBRIDGE_COM = '../dataframes/enbridge_com.csv'

HUSKY_SUB = '../dataframes/husky_energy_sub.csv'
HUSKY_COM = '../dataframes/husky_energy_com.csv'

# export path
SUB_OUTPUT = '../dataframe_main/submission_main.csv'
COM_OUTPUT = '../dataframe_main/comment_main.csv'

###########################################################
###########################################################

# append all data frames of the same type together
def df_reader():
    '''
    This function will create the master data frame and append all the
    scraped data frames and merge them on submission_id.
    '''

    # creating the submission list
    submission_list = [
        ESG_SUB, ETHICAL_SUB, RESP_SUB,
        GREEN_SUB, IMPACT_SUB, SUST_SUB, 
        ]
    comments_list = [
        ESG_COM, ETHICAL_COM, RESP_COM,
        GREEN_COM, IMPACT_COM, SUST_COM,
        ]
    
    # initializing list
    sub = []
    com = []

    # starting submission loop
    for file in submission_list:
        df = pd.read_csv(file, index_col=None, header=0)
        sub.append(df)
    
    # create frame
    sub_df = pd.concat(sub, axis=0, ignore_index=True)

    # starting comments loop
    for file in comments_list:
        df = pd.read_csv(file, index_col=None, header=0)
        com.append(df)
    
    # create frame
    com_df = pd.concat(com, axis=0, ignore_index=True)

    return sub_df, com_df


###########################################################
###########################################################

# frame cleaning function
def clean_df():
    '''
    This function will clean the data frame from null values
    and other unwanted data.
    '''

    # read in the file
    sub_df, com_df = df_reader()

    # datetime format fix
    sub_df['date'] = sub_df['date'].astype('datetime64').dt.strftime('%d-%m-%Y')

    # export
    sub_df.to_csv(SUB_OUTPUT, index=None, encoding='utf-8')
    com_df.to_csv(COM_OUTPUT, index=None, encoding='utf-8')

###########################################################
###########################################################

# run
if __name__ == '__main__':
    clean_df()
