import requests
from pathlib import Path
import requests
from bs4 import BeautifulSoup
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process input, gold, and output files")
    # parser.add_argument("--base_dir", help="retriever model name")
    parser.add_argument("--data_dir")
    args = parser.parse_args()


    base_url = 'https://lhncbc.nlm.nih.gov/ii/information/MBR/Baselines/2023/'
    base_name = "pubmed23n"
    
    download_dir = os.path.join(args.data_dir, 'bioasq/annual_zips/')
    
    Path(download_dir).mkdir(exist_ok=True)

    print('All files have been downloaded.')
