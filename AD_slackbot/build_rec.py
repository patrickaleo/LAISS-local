import requests as req
from auth import toku

class build_rec:
    def __init__(self, antaresID, tns_name, tns_cls, anom_score):
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
        if tns_name == "---":
            self.url = f'https://antares.noirlab.edu/loci/{antaresID}'
        else: self.url = f'https://ziggy.ucolick.org/yse/transient_detail/{tns_name}'

        self.name = antaresID
        self.tns_name = tns_name
        self.tns_cls = tns_cls
        self.anom_score = anom_score

        self.string = self.build_str()

    def build_str(self):
        return f'<{self.url}|{self.tns_name}> TNS spec. class = {self.tns_cls}, anomaly score = {int(round(self.anom_score, 1))}%'
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