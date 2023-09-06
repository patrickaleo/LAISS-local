import requests as req
from auth import toku

class build_rec:
    def __init__(self, antaresID):
        '''
        Builds a string and dataframe row for easy recommendation posting. Also can post to slack.

        Parameters
        ----------
        obj : prep.source object
            Object to build recommendation for. Needs to have a name, ra, dec, url, and salt_params attributes.
        
        Parameters
        ----------
        string : str
            String to post to slack. If None, will build a string from the object.
        df : pd.DataFrame
            DataFrame row to append to the recommendation DataFrame. If None, will build a row from the object.
        '''
        self.url = f'https://antares.noirlab.edu/loci/{antaresID}'
        self.name = antaresID
        self.string = self.build_str()

    def build_str(self):
        return f'<{self.url}|{self.name}> Found using automation'
        #return 'found using automation'

    def post(self,string=None,channel='D05R7RK4K8T'):
        '''
        Posts to a slack channel. If no string is provided, will use the string attribute of the object.

        Parameters
        ----------
        string : str, optional
            String to post to slack. If None, will use the string attribute of the object.
        channel : str, optional
            Channel to post to. Specific to workspace the bot token has been installed in.
        '''
        if string is None:
            string = self.string
        p1=req.post('https://slack.com/api/chat.postMessage',
                 params={'channel':channel,
                         'text':string,
                         'mrkdwn':'true',
                         'parse':'none'},
                         headers={'Authorization': f'Bearer {toku}'})
        p1.raise_for_status()
        if p1.status_code == 200:
            print('Posted to Slack')

def post(string=None, channel='D05R7RK4K8T'):
        '''
    Posts to a slack channel. If no string is provided, will use the string attribute of the object. This is a standalone function for autmation purposes.

    Parameters
    ----------
    string : str, optional
        String to post to slack. If None, will use the string attribute of the object.
    channel : str, optional
        Channel to post to. Specific to workspace the bot token has been installed in.
        '''
        if string is None:
            raise ValueError('No string provided')
        p1=req.post('https://slack.com/api/chat.postMessage',
                 params={'channel':channel,
                         'text':string,
                         'mrkdwn':'true',
                         'parse':'none'},
                         headers={'Authorization': f'Bearer {toku}'})
        p1.raise_for_status()
        if p1.status_code == 200:
            print('Posted to Slack')