import requests
from pathlib import Path
import requests
from bs4 import BeautifulSoup
from xml.etree import ElementTree as ET
from file_utils import save_jsonl, save_json

base_name = "pubmed23n"
file_names = [f"{base_name}{str(i).zfill(4)}.xml.gz" for i in range(1, 1167)]



# The XML content as a string - for demonstration purposes this will be a truncated version of the XML you provided



# Initialize an empty list to hold the extracted information
# articles_info = {}
articles_info = []
id2title = {}

for file_name in file_names:
    file_name = file_name.split('.gz')[0]
    
    file_path = f'/data/user_data/jhsia2/dbqa/data/bioasq/annual_zips/{file_name}'

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

save_json(id2title, '/data/user_data/jhsia2/dbqa/data/bioasq/id2title.json')
save_jsonl(articles_info, '/data/user_data/jhsia2/dbqa/data/bioasq/complete_medline_corpus.jsonl')