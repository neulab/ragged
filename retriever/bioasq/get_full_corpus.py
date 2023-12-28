import requests
from pathlib import Path
import requests
from bs4 import BeautifulSoup
from xml.etree import ElementTree as ET
from file_utils import save_jsonl, save_json
import os


import requests
from pathlib import Path
import requests
from bs4 import BeautifulSoup

# # The base URL for the downloads (you need to replace this with the actual base URL)
# base_url = 'https://lhncbc.nlm.nih.gov/ii/information/MBR/Baselines/2023/'

# # List of all file names (you need to fill this with the actual file names from the screenshot)
# # Generating a list of file names based on the given pattern

# # Define the base of the file names
# base_name = "pubmed23n"

# # Use a list comprehension to generate the full list of file names
# file_names = [f"{base_name}{str(i).zfill(4)}.xml.gz" for i in range(1, 1167)]

# # file_names[:10]  # Display the first 10 file names to check if they are correct


# # Directory where you want to save the downloaded files
# download_dir = '/data/tir/projects/tir6/general/afreens/dbqa/data/bioasq/annual_zips/'
# Path(download_dir).mkdir(exist_ok=True)

# print('All files have been downloaded.')


base_name = "pubmed23n"
file_names = [f"{base_name}{str(i).zfill(4)}.xml.gz" for i in range(1, 1167)]



# The XML content as a string - for demonstration purposes this will be a truncated version of the XML you provided



# Initialize an empty list to hold the extracted information
# articles_info = {}
articles_info = []
id2title = {}

root_dir = '/data/tir/projects/tir6/general/afreens/dbqa/data'

for file_name in file_names:
    file_name = file_name.split('.gz')[0]

    
    
    file_path = os.path.join(root_dir, f'bioasq/annual_zips/{file_name}')

    # Now let's parse the XML file
    tree = ET.parse(file_path)
    root = tree.getroot()

    # Iterate over all articles in the XML
    articles = root.findall('.//PubmedArticle')
    num_articles = len(articles)
    print(file_name, num_articles)
    for i, article in enumerate(articles):
        # if (i ==3):
        #     break
        # if (i%1000 == 0):
        #     print(i)
        # # Find and store the PMID
        id = article.findtext('.//PMID')
        
        # # Find and store the Article Title
        title = article.findtext('.//ArticleTitle')
        
        # # Find and store the Abstract Text
        # # Note: This will only get the first AbstractText if there are multiple.
        abstract = article.findtext('.//AbstractText')
        # print(id, title, abstract)
        if title != None and title != '':
            articles_info.append({'id': f'{id}_0', 'contents': title.strip()})
            id2title[id] = title.strip()
        if abstract != None and abstract != '':
            articles_info.append({'id': f'{id}_1', 'contents': abstract.strip()})
            # break
    # break

save_json(id2title, os.path.join(root_dir,'bioasq/id2title.json'))
save_jsonl(articles_info, os.path.join('bioasq/complete_medline_corpus.jsonl'))