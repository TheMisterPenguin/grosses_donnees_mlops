from pymongo import MongoClient
import os
from os.path import join, dirname
from dotenv import load_dotenv
from tqdm import tqdm
from colorama import Fore

# Script qui convertie la collection github/issues en github/issues_refined
# Les issues_refined sont des issues où l'on ne garde que les champs 'title', 'body' et 'labels'


print(Fore.YELLOW, 'connexion à la base....')
dotenv_path = join(dirname("../"), '.env')
load_dotenv(dotenv_path)

uri = os.environ.get("uri")
username = os.environ.get("username")
password = os.environ.get("password")


# Create a MongoClient instance
client = MongoClient(uri, username=username, password=password)

print(Fore.GREEN, 'Connecté.')


# Access a specific database
db = client['github']



# limite de résultats pour la query (0 pour infini)
limit_query = 0

issues = list(db['issues'].find({'labels': {'$ne': []}}, {'title': 1, 'body': 1, 'labels': 1, '_id': 0}).limit(limit_query))


print(Fore.YELLOW, 'Raffinage des issues...')

for issue in tqdm(issues):
    if 'labels' in issue and isinstance(issue['labels'], list):
        issue['labels'] = [label['name'] for label in issue['labels'] if 'name' in label]
        
        

# Create or replace the 'issues_refined' collection with the new data
db['issues_refined'].drop()
db['issues_refined'].insert_many(issues)


print(Fore.GREEN, "Traitement terminé, issues rafinées : ", len(issues))
print(Fore.WHITE, "")