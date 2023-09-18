
import numpy as np
import pandas as pd
import pathlib
from pathlib import Path
import os
import glob
import datetime
import json


tweetlist = []
for line in open('GoT_tweets.txt', 'r'): # Open the file of tweets
    tweetlist.append(json.loads(line))  # Add to 'tweetlist' after converting
    
# We convert to a Series so that we can vectorize operations as shown below
tweets = pd.Series(tweetlist)
# Question 1

# Twitter users can have 'verified' accounts, which were previously at no cost
# but recently that has changed.  In our data, each tweet indicates that it was
# posted by a verified user by the flag 'verified' within the 'user' information.
# Determine the number of tweets posted by verified users.

q1 = sum(1 for tweet in tweets if tweet['user']['verified'])



# Question 2

# Tweets can be flagged as having potentially sensitive content, either by the
# individual posting the tweet or by an agent of Twitter.  Such tweets are
# indicated in the key 'possibly_sensitive'.  Determine the sscreen names of
# all users who posted more than one potentially sensitive tweet.  Give a
# Series with screen name as index and number of potentially sensitive tweets
# as value, sorted alphabetically by screen name.

df = pd.json_normalize(tweets)
q2df = df[df['possibly_sensitive'] == True]
user_sensitive_counts = q2df.groupby('user.screen_name')["possibly_sensitive"].count()
user_sensitive_counts = user_sensitive_counts[user_sensitive_counts > 1]
q2 = user_sensitive_counts.sort_index()


# Question 3

import re 

# One might expect that the name Daenerys to appear in our collecction of tweets.
# Determine the percentage of tweets that include the name 'Daenerys' in the text 
# of the tweet. Any combination of upper and lower case should be included, and
# also include instances where Daenerys has non-alphanumeric (letters and numbers)
# before or after, such as #daenerys or Daenerys! or @Daenerys.  Do not include
# instances where Daenerys is immediately preceded or followed by letters or
# numbers, such as GoDaenerys or Daenerys87.

df["text"] = df["text"].apply(str.lower)
zz = df.loc[df["text"].str.contains("[^A-Za-z0-9]daenerys[^A-Za-z0-9]")]
q3 = len(zz) / len(tweets) * 100


# Question 4

# Determine the number of tweets that have 0 user mentions, 1 user mention, 
# 2 user mentions, and so on.  Give your answer as a Series with the number of 
# user mentions as index (sorted smallest to largest) and the corresponding 
# number of tweets as values. Include in your Series index only the number of   
# user mentions that occur for at least one tweet, so for instance, if there
# are no tweets with 7 user mentions then 7 should not appear as an index
# entry. Use the list of user mentions (within 'entities') from each tweet, 
# not the text of the tweet. 

pq4 = tweets.str['entities'].str['user_mentions'].explode().reset_index().groupby("index").count()
q4 =  pq4.value_counts().sort_index()




# Question 5

# Determine the number of tweets that include the hashtag '#GameofThrones'.
# (You may get the wrong answer if you use the text of the tweets instead of 
# the hashtag lists.) Note that Hashtags are not case sensitive, so any 
# combination of upper and lower case are all considered matches so should be 
# counted.

pq5 = tweets.str['entities'].str['hashtags'].explode().dropna().str["text"].reset_index()
pq5[0] = pq5[0].apply(str.upper)
GOT = pq5.loc[pq5[0].str.contains(r'\bGAMEOFTHRONES\b')]
q5 = len(GOT["index"].drop_duplicates())


# Question 6

# Some tweeters like to tweet a lot.  Find the screen name for all tweeters
# with at least 3 tweets in this data.  Give a Series with the screen name
# as index and the number of tweets as value, sorting by tweet count from
# largest to smallest

tweet_counts = df['user.screen_name'].value_counts()
q6 = tweet_counts[tweet_counts >= 3]


# Question 7

# Among the screen names with 3 or more tweets, find the average
# 'followers_count' for each and then give a table with the screen and average 
# number of followers.  (Note that the number of followers might change from 
# tweet to tweet.)  Give a Series with screen name as index and the average 
# number of followers as value, sorting by average from largest to smallest.  

followers_by_user = df.groupby('user.screen_name')['user.followers_count'].mean() 
followers_table = pd.DataFrame({'tweets': q6, 'followers': followers_by_user[q6.index]}).drop("tweets", axis=1)
q7 = followers_table.squeeze().sort_values(ascending = False)


# Question 8
                                                                
# Determine the hashtags that appeared in at least 25 tweets.  Give
# a Series with the hashtags (lower case) as index and the corresponding 
# number of tweets as values, sorted alphabetically by hashtag.

hashtags = tweets.str['entities'].str['hashtags'].explode().str["text"].value_counts().reset_index()
hashtags["index"] = hashtags["index"].apply(str.lower)
combined = hashtags.groupby("index").sum().reset_index()
q8 = combined.loc[combined[0] >= 25].set_index("index").squeeze()


# Question 9

# A tweet can contain links, but the Twitter algorithm will downgrade the
# visibility of such tweets when the link is to a site other than Twitter 
# because it considers the tweet to be spam.  Links can be found within 'urls'
# contained in 'entities' for each tweet.  Among the tweets that include links,
# what percentage will not interpreted as spam by Twitter? (Note that one can
# see the whole URL in 'expanded_url'.)

links = df["entities.urls"].explode().dropna()
expand = pd.json_normalize(links)
matches = expand.loc[expand["display_url"].str.contains("twitter")]
q9 = (len(matches) / len(links)) * 100


# Question 10

# Determine which tweets contain a sequence of three or more consecutive digits
# (no spaces between the digits!).  From among those tweets, determine the
# percentage that include a user mention (starts with '@') that has a sequence 
# of three or more consecutive digits.

digit_regex = r"\d{3,}"
digit_tweets = df[df['text'].str.contains(digit_regex)] 
digit_mention_tweets = digit_tweets[digit_tweets['text'].str.contains(r"@\w*{}\w*".format(digit_regex))]
q10 = len(digit_mention_tweets) / len(digit_tweets) * 100










