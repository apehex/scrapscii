import hashlib
import io
import itertools
import json
import os
import random
import subprocess
import tempfile
import urllib

import datasets
import pyarrow.lib as pl
import pyarrow.parquet as pq
import requests

import scrapscii.data
import scrapscii.unicode

# CONSTANTS ####################################################################

WIDTH_MIN = 16
WIDTH_MAX = 128

# IO ###########################################################################

HOME_PATH = os.getenv('HOME')
TEMP_PATH = tempfile.mkdtemp()
DATA_PATH = os.path.realpath(os.path.join(os.path.dirname(__file__), '../', 'datasets/images'))

# EXTRACT ######################################################################

if __name__ == '__main__':
    # shard index
    __shard = 0

    # download the dataset
    __table = []
    __dataset = datasets.load_dataset('apple/DataCompDR-12M', split='train', cache_dir='~/.cache/huggingface/datasets', streaming=True)
    __iter = itertools.islice(__dataset, 0, 1024)

    # iterate over the samples
    for __sample in __iter:
        __sample = next(__iter)

        # parse the URL
        __url = __sample['url.txt']
        __hash = hashlib.sha1(__url.encode('utf-8')).hexdigest()
        __path = urllib.parse.urlparse(__url).path
        __filename = __path.split('/')[-1]
        __extension = os.path.splitext(__filename)[-1]

        # download image from URL
        try:
            __response = requests.get(__url, timeout=2)
        except:
            print(f'Failed to download {__url}')
            continue

        # save image on disk
        __path = os.path.join(TEMP_PATH, __hash + __extension)
        with open(__path, 'b+w') as __file:
            __file.write(__response.content)

        # choose the config randomly
        __width = '--width {width}'.format(width=random.randint(WIDTH_MIN, WIDTH_MAX))
        __braille = '--braille' if random.choice([True, False]) else ''
        __complex = '--complex' if random.choice([True, False]) else ''
        __dither = '--dither' if __braille and random.choice([True, False]) else ''
        __grayscale = '--grayscale' if random.choice([False]) else '' # colorless terminal
        __negative = '--negative' if random.choice([True, False]) else ''

        # choose a caption among the synthetic text
        __index = random.randint(0, len(__sample['syn.json']['syn_text']) - 1)
        __caption = __sample['syn.json']['syn_text'][__index]

        # export the conversion config
        __labels = [__l for __l in [__width, __braille, __complex, __dither, __grayscale, __negative] if __l]

        # convert the image to ASCII art
        __flags = list(itertools.chain.from_iterable(__l.split(' ') for __l in __labels if __l))
        __process = subprocess.run(['ascii-image-converter'] + __flags + [__path], stdout=subprocess.PIPE)
        __content = __process.stdout.decode('utf-8')

        # save
        __table.append({
            'caption': __caption,
            'content': __content,
            'labels': ','.join(__labels),
            'charsets': ','.join(set(scrapscii.unicode.lookup_section(__c) for __c in __content)),
            'chartypes': ','.join(set(scrapscii.unicode.lookup_category(__c) for __c in __content)),})

    # export as parquet 
    pq.write_table(
        table=pl.Table.from_pylist(
            mapping=__table,
            schema=scrapscii.data.SCHEMA),
        where=os.path.join(DATA_PATH, '{shard:0>4d}.parquet'.format(shard=__shard)))
