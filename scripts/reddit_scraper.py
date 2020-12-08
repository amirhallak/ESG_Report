'''
Reddit Scraper -- TerraHub Project.

@author Amir Hallak
'''

###########################################################
###########################################################

# importing modules.
import praw, pandas as pd, datetime

###########################################################
###########################################################

# constants
DF_PATH = '../dataframes/'

###########################################################
###########################################################

# keyword list
keyword_list = [
    'esg', 'ethical investing', 'greenwashing', 'cenovus', 'enbridge',
    'impact investing', 'husky energy','suncor', 'sustainable investing',
    ]

# subreddit list
list_of_subreddits = [
    'FinancialCareers', 'economics', 'Investing',
    'RobinHood', 'WallStreetBets', 'SecurityAnalysis',
    'InvestmentClub', 'StockMarket', 'Stock_Picks',
    'Forex', 'Options', 'FinancialIndependence', 
    'sustainability', 'personalfinance',
    'worldnews', 'finance', 'kickstarter',
    'academia', 'askeconomics', 'politics',
    'askacademia', 'badeconomics','canada', 
    'calgary', 'college', 'news', 'gradschool',
    'technology', 'tech', 'startups',
    'techolitics','futurology', 'renewableenergy','engineering',
    'cryptocurrency', 'environment', 
    'Europe', 'zerowaste', 'green', 'unitedkingdom', 'energy',
    'humanrights', 'climatechange', 'climate'
    ]

###########################################################
###########################################################

# authenticator function
def authenticator_df():
    '''
    This is the authenticator function that gives access to reddit.

    Returns:
    ---------
    reddit: praw object
        The authenticator information to log into the Reddit API. 
    '''

    # inserting the details
    reddit = praw.Reddit(
    client_id = 'GoxN9vIl_3XQjQ',
    client_secret = 'iHCNffKTJAtO-fTg8YczxTtwgOY',
    user_agent = 'testing_599',
    username = '599terrahub',
    password = '599terrahub')
    return reddit


# creating the main function for reddit scraping. 
def reddit_scraper(keyword):
    '''
    The actual reddit scraper function in which we can configure to grab what
    we need from a reddit post. 
    
    Parameters:
    -----------
    keyword: str
        The keyword we are interested in scraping off reddit.

    Returns:
    --------
    pandas dataframe
        This function will return two dataframes for each keyword; a comment dataframe and a submission dataframe. 
    '''
    
    # initialize authenticator
    reddit = authenticator_df()
    
    # create an empty list to append during for-loop. 
    submission_list = []
    comment_list = []

    # start the loop. 
    for subreddit in list_of_subreddits:
        print(f'\n--{subreddit}')
        subreddit = reddit.subreddit(subreddit)
        
        for submission in subreddit.search(keyword):
            
            # create a dictionary for the submittion section. 
            var_dict = {}
            var_dict['submission_id'] = submission.id
            var_dict['title'] = submission.title
            var_dict['author'] = submission.author.name
            var_dict['num_comments'] = submission.num_comments
            var_dict['upvotes'] = submission.ups
            var_dict['upvote_ratio'] = submission.upvote_ratio
            var_dict['date'] = datetime.datetime.fromtimestamp(submission.created)
            var_dict['subreddit'] = submission.subreddit
            var_dict['url'] =  submission.url
            var_dict['keyword'] = keyword
            
            # append variables to list.
            submission_list.append(var_dict)

            # create a dictionary for the comment section.
            comments = submission.comments
            submission.comments.replace_more(limit=None)
        
            for top_level_comment in comments:
                
                # create a dictionary with the variables we want. 
                com_dict = {}
                com_dict['submission_id'] = submission.id
                com_dict['comment_id'] = top_level_comment.id
                com_dict['comment_author'] = top_level_comment.author            
                com_dict['comment'] = top_level_comment.body
                com_dict['keyword'] = keyword

                # append comments to list
                comment_list.append(com_dict)

    # create a dataframe to hold the variables. 
    submission_df = pd.DataFrame(submission_list)
    comment_df = pd.DataFrame(comment_list)

    return submission_df, comment_df


# get various dataframes according to keyword lists.
def main():
    '''
    This function will run the main operation and exports
    the dataframes to the dataset folder. 
    '''
    
    # starting the loop
    for keyword in keyword_list:
        
        print(f'\n{keyword}')
        
        # get the data frames
        sub_df, com_df = reddit_scraper(keyword)

        # export
        sub_df.to_csv(DF_PATH+str(keyword)+'_sub.csv',encoding='utf-',index=None)
        com_df.to_csv(DF_PATH+str(keyword)+'_com.csv',encoding='utf-',index=None)


###########################################################
###########################################################

#run 
if __name__=='__main__':
    main()