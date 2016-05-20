



# group by clusters
def get_user_tuples(df, user='user' , cluster='cluster'):
    grouped_user = df.groupby(user)
    grouped_clusters = df.groupby([user, cluster])
    clusters = df.cluster.unique()
    users = []
    for name, group in grouped_user:
        user_tuple = []
        total_tweets = len(group)
        for each_clusters in clusters:
            try:
                g = grouped_clusters.get_group((name,each_clusters))
                user_tuple.append(len(g)/total_tweets)
            except KeyError:
                user_tuple.append(0)
        users.append((user_tuple))
    return users