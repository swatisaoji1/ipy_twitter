from pymongo import MongoClient

class Database:
    def __init__(self, db_name='stream_store'):
        client = MongoClient()
        self.db = client[db_name]



    def get_followers(self, user_id):
        return self.db.social_network.find_one({'userid' : user_id})
        #TODO: complete this method

    def is_follower(self, user_a, user_b):
        """
        :param user_a: userid of user A
        :param user_b: userid of user B
        :return: True if user_a follows user_b , returns false if does not follow and -1 is information not found
        """

        userA = self.db.social_network.find_one({'userid' : user_a})
        userB = self.db.social_network.find_one({'userid' : user_b})
        print ('*',)
        if userA and userA["friends_status"] == 2:
                if user_b in userA["friends"]:
                    return True
                else:
                    return False
        if userB and userB["following_status"] == 2:
                if user_a in userB["followers"]:
                    return True
                else:
                    return False
        return False


    def is_friend(self, user_a, user_b):
        """

        :param user_a:
        :param user_b:
        :return:
        """
        return self.is_follower(user_b,user_a)

    def get_username(self, user_id):
        """

        :param user_id:
        :return:
        """
        result = self.db.user_profiles.find_one({'id_str' : user_id})
        if result:
            return result['screen_name']
        else:
            return None
