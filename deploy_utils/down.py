import pandas as pd 
from tqdm.auto import tqdm 
import subprocess 
import os.path as osp

df = pd.read_csv('/home/ubuntu/y1/truemedia.csv')
for i in tqdm(range(len(df))):
    line = df.iloc[i]
    ID = line['Id']
    fake = 'fake' in line['Ground Truth'].lower()
    if osp.exists(f'/truemedia-eval/images/fakes/{ID}') or osp.exists(f'/truemedia-eval/images/reals/{ID}'):
        continue
    if fake:
        subprocess.run(f"aws s3 cp s3://truemedia-media/{ID} /truemedia-eval/images/fakes/{ID}", shell=True)
    else:
        subprocess.run(f"aws s3 cp s3://truemedia-media/{ID} /truemedia-eval/images/reals/{ID}", shell=True)
