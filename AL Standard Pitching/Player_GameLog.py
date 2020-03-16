import requests, re, os, tqdm
import pandas as pd
from bs4 import BeautifulSoup as BSoup

def PlayerList():
    players = pd.read_csv(filepath_or_buffer = 'summary.csv')
    ID_Lists = players[['Name', 'playerID']]
    return ID_Lists

def PlayerGameLog(playerList, savepath):
    tmp_name, tmp_id = playerList
    prefix = 'https://www.baseball-reference.com/players/gl.fcgi?id='
    postfix = '&t=p&year=2017'
    URL = prefix + tmp_id + postfix
    req = requests.get(URL)
    soup = BSoup(req.text, 'lxml')
    soup_GamelogsTable = soup.find(name = 'table', id = 'pitching_gamelogs')
    soup_headers = soup_GamelogsTable.find(name = 'thead').findAll(name = 'th')
    headers = [header.text for header in soup_headers]
    headers.append('PlayerName')
    df_gamelog = pd.DataFrame(columns = headers)
    soup_gamelogs = soup_GamelogsTable.find(name = 'tbody').findAll(name = 'tr', id = re.compile('pitching_gamelogs\.[0-9]+'))
    for soup_gamelog in soup_gamelogs:
        df_gamelog = InsertToDataFrame(df_gamelog, soup_gamelog, tmp_name)
    if tmp_name[-1] == '*':
        tmp_name = tmp_name[:-1]
    df_gamelog.to_csv(os.path.join(savepath, tmp_name+'.csv'), index=False, header=True)

def InsertToDataFrame(df, gamelog, PlayerName):
    soup_score = gamelog.findAll(name = {'th', 'td'})
    score = [s.text for s in soup_score]
    score[3] = re.sub('\xa0', ' ', score[3])
    score.append(PlayerName)
    forAppend =dict(zip(df.columns, score))
    df = df.append(forAppend, ignore_index = True)
    return df

if __name__ == '__main__':
    path = os.path.join(os.getcwd(), 'gamelogs')
    players = PlayerList()
    for player in tqdm.tqdm(players.iterrows()):
        PlayerGameLog(player, path)