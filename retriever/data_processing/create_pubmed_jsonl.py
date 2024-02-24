import requests
from pathlib import Path
import requests
from bs4 import BeautifulSoup
from xml.etree import ElementTree as ET
from file_utils import save_jsonl, save_json
import os
import argparse
import requests
from pathlib import Path
import requests
from bs4 import BeautifulSoup

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_dir")
    args = parser.parse_args()

    base_name = "pubmed23n"
    file_names = [f"{base_name}{str(i).zfill(4)}.xml.gz" for i in range(1, 1167)]

    articles_info = []
    id2title = {}

    output_dir = os.path.join(args.corpus_dir, 'pubmed')
    os.makedirs(output_dir)

    for file_name in file_names:
        file_name = file_name.split('.gz')[0]
        file_path = os.path.join(output_dir, f'annual_zips/{file_name}')

        tree = ET.parse(file_path)
        root = tree.getroot()

        articles = root.findall('.//PubmedArticle')
        num_articles = len(articles)
        print(file_name, num_articles)
        for i, article in enumerate(articles):
            id = article.findtext('.//PMID')
            title = article.findtext('.//ArticleTitle')
            # # Note: This will only get the first AbstractText if there are multiple.
            abstract = article.findtext('.//AbstractText')

            if title != None and title != '':
                articles_info.append({'id': f'{id}_0', 'contents': title.strip()})
                id2title[id] = title.strip()

            if abstract != None and abstract != '':
                articles_info.append({'id': f'{id}_1', 'contents': abstract.strip()})

    save_json(id2title, os.path.join(output_dir,'id2title.json'))
    save_jsonl(articles_info, os.path.join(output_dir, 'pubmed_jsonl', 'pubmed.jsonl'))