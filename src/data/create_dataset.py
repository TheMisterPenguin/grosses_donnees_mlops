from pymongo import MongoClient
import os
from os.path import join, dirname
from dotenv import load_dotenv
from tqdm import tqdm
from colorama import Fore

# Script qui convertie la collection github/issues en github/issues_refined
# Les issues_refined sont des issues où l'on ne garde que les champs 'title', 'body' et 'labels'


print(Fore.YELLOW, 'connexion à la base....')
dotenv_path = join(dirname("."), '.env.local')
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

count = None if limit_query == 0 else limit_query

print(Fore.BLUE, 'Traitement de', count, 'issues...')

# Open an explicit MongoDB session
with client.start_session() as session:

    issues = db['issues'].find({'labels': {'$ne': []}}, {'title': 1, 'body': 1, 'labels': 1, '_id': 0}, no_cursor_timeout=True, session=session).limit(limit_query)
    # Create or replace the 'issues_refined' collection with the new data
    db['issues_refined'].drop()
    refined_issues = db['issues_refined']

    print(Fore.YELLOW, 'Raffinage des issues...')

    try:
        for issue in tqdm(issues, total=count):
            tranformed_issue = issue

            if 'labels' in tranformed_issue and isinstance(tranformed_issue['labels'], list):
                tranformed_issue['labels'] = [label['name'] for label in tranformed_issue['labels'] if 'name' in label]

            refined_issues.insert_one(tranformed_issue)
    finally:
        issues.close()

    print(Fore.GREEN, "Traitement terminé")
    print(Fore.WHITE, "")