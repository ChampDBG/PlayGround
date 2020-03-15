import requests, re, os, tqdm
import pandas as pd
from bs4 import BeautifulSoup as BSoup

## function
def PlayerStandardPitchingTable():
    URL = 'https://www.baseball-reference.com/leagues/AL/2017-standard-pitching.shtml'
    req = requests.get(URL)
    soup = BSoup(re.sub(pattern='<!--|-->', repl='', string=req.text), 'lxml')
    '''
    * Alternative
    comm = re.compile(pattern='<!--|-->')
    soup = BSoup(re.sub(repl='', string=req.text), 'lxml')
    
    If you need to compare the same thing in different string, re.compile will be a better choice.
    '''
    table = soup.find(name = 'table', id = 'players_standard_pitching')
    return table

def TableToDataFrame(PSP_table):
    tb_heads = PSP_table.find(name = 'thead').findAll(name = 'th')
    tb_heads_text = [tb_head.text for tb_head in tb_heads]
    tb_heads_text.append('playerID')
    df_record = pd.DataFrame(columns = tb_heads_text)
    tb_body = PSP_table.find(name = 'tbody')
    players = tb_body.findAll(name = 'tr', attrs = {'class': ['full_table non_qual', 'partial_table non_qual', 'full_table', 'partial_table']})
    for player in tqdm.tqdm(players):
        df_record = InsertToDataFrame(df_record, player)
    return df_record

def InsertToDataFrame(df, player):
    score_soup = player.findAll(name={'th', 'td'})
    score = [s.text for s in score_soup]
    score[1] = re.sub('\xa0', ' ', score[1])
    score.append(score_soup[1].get('data-append-csv'))
    forAppend = dict(zip(df.columns, score))
    df = df.append(forAppend, ignore_index = True)
    return df

if __name__ == '__main__':
    PSP_table_soup = PlayerStandardPitchingTable()
    PSP_table = TableToDataFrame(PSP_table_soup)
    PSP_table.to_csv(os.path.join(os.getcwd(), 'Summary.csv'), index=False, header=True)