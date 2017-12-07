"""
collect.py
This file will collect all Twitter API data based on some queries. The data is raw and taken from original source.
This script will create some files containing the data that is required for the subsequent phases of analysis.

"""

from collections import defaultdict
import tweepy
import pickle
import time
import sys


def get_twitter_api():
    """
    Construct an instance of TwitterAPI using the tokens generated via apps.twitter.com.

    Returns:
          An instance of TwitterAPI using Tweepy.
    """

    # Running for the below specs -------------> 1 movie, 5 screens, 250 followers and 300 followers.

    consumer_key = 'rrbclESAhmgl3eYBhuYud3PZ0'
    consumer_secret = 'qpZyaFEkgLU2twbDkxfIlCZ34vBgCqDm0dfZT7ReNiSm6IZ1eo'
    access_token = '903123775704719360-HcvDLT9iPajPJDuwhKOArcFMBuaxMxk'
    access_token_secret = 'Pd16t1BeuwrQjgNPGkumjGwywDjuCIIov7uzwU9JqtGKN'

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)

    api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True, retry_count=5)

    return api


def read_input_file(input_file):
    """
    Read a text file containing movie names, one per line.

    Params:
            input_file....Name of the file to read.
    Returns:
            A list of strings, one per movie_name, in the order they are listed in the file.
    """

    with open(input_file, 'r') as f:
        movie_names = [line.replace('\n', '').lower() for line in f]

    return movie_names


def search_query(api, movie_names):
    """Retrieve the Twitter user objects for each screen_name.
    Params:
            api........The Tweepy TwitterAPI object.
            movie_names........A list of strings, one per movie_name
            count = 1000 tweets per movie name.
    Returns:
            A Response object from Twitter Api, containing all the user information.
            (Response object will be screen_name, id, location, etc)
    """

    results = []
    for movie in movie_names:
        res = api.search(q=movie, language='en', count=1000, encode='utf-8')
        results.extend(res)

    return results


def get_screen_ids(results):
    """
    Return a list of Twitter IDs for users who tweeted on the lines of the movie.

    Args:
            tweepy.......The Response object
    Returns:
            A list of ints, one per user ID.
    Note:
        The user object will return 1000 responses, we will limit ourselves to the first 5 accounts
        for our further processing.
    """

    screen_ids = [result.user.id for result in results]
    ids = set(screen_ids[:5])
    return ids


def get_tweets(results):
    """
    Return a list of Twitter IDs for users who tweeted on the lines of the movie.
    Args:
            tweepy.......The Response object
    Returns:
            A list of tweets, per movie name.
    """

    text_tweets = [result.text for result in results]
    text_tweets = list(set(text_tweets))
    return text_tweets


def get_followers_screen_name(api, ids):
    """
    Get the list of accounts each user follows. i.e., call the followers_ids method for all the 5 candidates.
    Args:
            api...The TwitterAPI.
            ids.....The list of ids.
    Returns:
            followers of ids ....... A dict from ids to its followers_ids.
    """

    followers_screen_name_dict = defaultdict(list)

    for followers_ids in ids:
        followers_of_ids = api.followers_ids(user_id=followers_ids, count=250)
        if api.get_status(200):
            followers_screen_name_dict[followers_ids] = followers_of_ids
        else:
            print('Got error %s \n sleeping for 15 minutes.' % api.text)
            sys.stderr.flush()
            time.sleep(61 * 15)

    return followers_screen_name_dict


def get_followers_followers(api, followers_screen_name_dict):
    """
    Get the list of accounts each user follows. i.e., call the followers_ids method for all the 5 candidates.
    Args:
            api...The TwitterAPI.
            followers_dict.....The dict of followers from screen_ids to its followers_ids.
    Returns:
            followers_followers_dict ....... A dict from followers_ids to its followers.
    """

    followers_followers_dict = defaultdict(list)

    for scr_name, ids in followers_screen_name_dict.items():
        for i in ids:
            try:
                if i not in followers_followers_dict.keys():
                    followers_followers_ids = api.followers_ids(user_id=i, count=300)
                    if api.get_status(200):
                        followers_followers_dict[i] = followers_followers_ids
                        time.sleep(11)
                    else:
                        print('Got error %s \nsleeping for 15 minutes.' % api.text)
                        sys.stderr.flush()
                        time.sleep(61 * 15)
            except tweepy.TweepError:
                print("The record is protected so Skipping this record ")

    return followers_followers_dict


def dump_output(followers_followers_dict, ids, tweets):

    """
    Get the list of accounts each user follows. i.e., call the followers_ids method for all the 5 candidates.
    Args:
            ids..... The screen_ids.
            followers_followers_dict.....The dict of followers from screen_ids to its followers_ids.
            tweets..... The list of tweets obtained from search query.
    Returns:
            Nothing.
                Note: Dump all the above files to pickle for further processing.
    """

    followers = open('followers_followers_dict.pkl', 'wb')
    pickle.dump(followers_followers_dict, followers)
    followers.close()

    screen_ids = open('ids.pkl', 'wb')
    pickle.dump(ids, screen_ids)
    screen_ids.close()

    text = open('tweets.pkl', 'wb')
    pickle.dump(tweets, text)
    text.close()


def main():
    """
    Main Method.
    """
    api = get_twitter_api()
    print("Established twitter connection through TWEEPY.......... \n")
    print("This collect file will run for 5 screen_ids, 250 followers of them & 300 followers of the followers.....\n")
    print("\n ************** Collection of data is slightly huge and hence will take approx 12-14 hrs to run.*********\n")

    print("The data is collected and kept in folder which will be processed in cluster.py and classify.py")

    print("\nFollowers data is stored in -----> Followers_followers_dict.pkl, and ID's data are stored in -----> ids_bk.pkl files")

    movie_names = read_input_file('movies')
    results = search_query(api, movie_names)

    print("Collecting the most recent/popular screen_ids who tweeted the movie - JusticeforLeague.......\n\n")
    ids = get_screen_ids(results)

    tweets = get_tweets(results)
    print("Collecting followers of the above screen_names .......\n")
    followers_screen_name_dict = get_followers_screen_name(api, ids)
    print("\nNow Collecting followers of each followers collected .......\n")
    followers_followers_dict = get_followers_followers(api, followers_screen_name_dict)

    dump_output(followers_followers_dict, ids, tweets)
    print("\nCollected the twitter data successfully by querying the tweets related to the movie name "
          "- 'Justiceleague' and dumped the output files via pickle for clustering and classification parts.")


if __name__ == '__main__':
    main()
