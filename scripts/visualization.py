'''
This is the 3rd and final script in which will show some
visualization and analysis of the scraped data.

@author Amir Hallak
'''

###########################################################
###########################################################

# modules import
from matplotlib.pyplot import ylabel
import pandas as pd, numpy as np, seaborn as sns, datetime, nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from matplotlib import pyplot as plt
from seaborn import palettes
from sklearn.metrics import accuracy_score, f1_score

###########################################################
###########################################################

# paths
SUB_DF = '../dataframe_main/submission_main.csv'
COM_DF = '../dataframe_main/comment_main.csv'

###########################################################
###########################################################

# importing the comment data frame
df2 = pd.read_csv(COM_DF, dtype=str, encoding='utf-8')

# dropping deleted comments & duplcates
df2 = df2[df2['comment']!='[deleted]']
df2 = df2.drop_duplicates(subset=['comment'])

# cleaning strings from new lines and other characters - regular exporessions
df2['comment'] = df2['comment'].replace(r'\n',' ', regex=True)
df2['comment'] = df2['comment'].str.replace('[^\w\s]','', regex=True)


# list of lists
# creating a list for each keyword that we have, and concatenating the words


#######

tokenizer = nltk.tokenize.RegexpTokenizer('\w+')
stopwords = nltk.corpus.stopwords.words('english')

# increasing stopwords.
other_stopwords = [
    'one', 'make', 'like', 'even', 'going','www',
    'r', 'com', 'like', 'would', 'https', 'get', 'people',
    'think', 'want', 'much', 'also','could','1', 'dont', 'im',
    'time','youre','need','really','know','well','go','way',
    'see','still','lot','esg','use','take','something','year'
    ]

for i in other_stopwords:
    stopwords.append(i)


### 1
main_df = pd.DataFrame(columns=['word','frequency','keyword','total comments'])

for list in dict.keys():

    words_list = tokenizer.tokenize(dict[list])
    word_distribution = nltk.FreqDist(w for w in words_list if w not in stopwords)

    word_df = pd.DataFrame(
        word_distribution.most_common(10),
        columns = ['word', 'frequency']
        )  
    
    word_df['keyword'] = str(list)
    word_df['total comments'] = len(df2[df2['keyword']==str(list)].index)
    
    main_df = pd.concat([main_df, word_df])



main_df['per comment'] = main_df['frequency'].astype(int) / main_df['total comments'].astype(int)
main_df['per comment'] = main_df['per comment'].round(3)
main_df['word'] = main_df['word'].str.title()
main_df['keyword'] = main_df['keyword'].str.title()


plt.xticks(np.arange(0, 0.3, 0.05))


for keyword in main_df['keyword'].unique():
    
    g = sns.catplot(
            x='per comment',
            y='word',
            data=main_df[main_df['keyword']==keyword], 
            kind='bar', 
            aspect=0.7, 
            palette="Paired"
            )
    
    g.fig.set_size_inches(10,5)
    g.set_axis_labels("Frequency per Comment", "Unique Word", labelpad=10)
    plt.title(f"Word Frequencies in Comments - Keyword = {keyword}")


plt.show()


################################################################################

# word count processor. 
'''
# grab the words from the data frame. 
comm_clean = comment_df['comment'].astype(str).str.lower().str.cat(sep=' ')

# remove punctuation and common words like 'the'.
tokenizer = nltk.tokenize.RegexpTokenizer('\w+')
comm_clean1 = tokenizer.tokenize(comm_clean)
stopwords = nltk.corpus.stopwords.words('english')

# increasing stopwords.
other_stopwords = [
    'one', 'make', 'like', 'even', 'going','www',
    'r', 'com', 'like', 'would', 'https', 'get', 'people',
    'think', 'want', 'much', 'also','could','1']
for i in other_stopwords:
    stopwords.append(i)

# distribution and put in a dataframe. 
word_distribution = nltk.FreqDist(w for w in comm_clean1 if w not in stopwords) 

word_dis_df = pd.DataFrame(
    word_distribution.most_common(10), 
    columns = ['Word', 'Freq'],
    )

#plot frequency.
bar_plot = sns.barplot(data = word_dis_df, x = 'Word', y = 'Freq', palette='Set2')\
.set_title('Word Frequency for ' + word.upper() + ' comments')
fig = plt.gcf().set_size_inches(10,4)
plt.show()
bar_plot.figure.savefig(output_path + 'word_frq_' + word + '.png', format='png', dpi=750)

# sentiment analytics -- supervised model. 

# visualization
submission_df['month_year'] = submission_df['date'].dt.to_period('M')
submission_df.sort_values(by='date', inplace = True)

# plot 3
month_year_plot = sns.countplot(submission_df['month_year'], palette='Set2')\
.set_title('Month-Year Distribution for ' + word.upper() + ' submissions.')
fig = plt.gcf().set_size_inches(40,5)
plt.show()
month_year_plot.figure.savefig(output_path + 'Year_Month_' + word + '.png', format='png', dpi=750)

# plot 4
year_dist_plot = sns.countplot(submission_df['date'].dt.year, palette="Set2")\
.set_title('Year Distribution for ' + word.upper() + ' submissions.')
fig = plt.gcf().set_size_inches(10,4)
plt.show()
year_dist_plot.figure.savefig(output_path + 'Year_Dist_' + word + '.png', format='png', dpi=750)

#exporting relevant data.
submission_df.to_csv(output_path + 'sub_' + word + '.csv', index=False)
comment_df.to_csv(output_path + 'com_' + word + '.csv', index=False)
'''
