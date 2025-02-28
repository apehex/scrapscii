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
import tqdm

import scrapscii.data
import scrapscii.unicode

# CONSTANTS ####################################################################

WIDTH_MIN = 16
WIDTH_MAX = 128
TABLE_LEN = 2**5
SHARD_LEN = 2**10
TOTAL_LEN = 2**12

# IO ###########################################################################

TEMP_PATH = tempfile.mkdtemp()
DATA_PATH = os.path.realpath(os.path.join(os.path.dirname(__file__), '../', 'datasets/images'))

# CHECK ########################################################################

CORRUPTED_HASH = ['4dcb57651a75abfd07fb36c70c6c5108c49bdb34']

def is_valid_image(image: bytes) -> bool:
    return (
        bool(image)
        and type(image) == bytes
        and not hashlib.sha1(image).hexdigest() in CORRUPTED_HASH)

def is_valid_ascii(ascii: str, width: int=WIDTH_MIN) -> bool:
    return (
        bool(ascii)
        and type(ascii) == str
        and len(ascii) >= width
        and not 'error: can\'t decode' in ascii.lower())

# DOWNLOAD #####################################################################

def download_image(url: str, timeout: int=1) -> bytes:
    __bytes = b''
    # retrieve the image content as bytes
    try:
        __response = requests.get(url, timeout=timeout)
        __bytes = __response.content
    # ignore exceptions
    except Exception as __e:
        __bytes = b''
    # default
    return __bytes

def format_path(url: str, temp: str=TEMP_PATH) -> str:
    # parse the URL
    __path = urllib.parse.urlparse(url).path
    __filename = __path.split('/')[-1]
    __extension = os.path.splitext(__filename)[-1]
    # reduce the filename to a fixed size
    __hash = hashlib.sha1(url.encode('utf-8')).hexdigest()
    # safe path
    return os.path.join(temp, __hash + __extension)

def export_image(data: bytes, path: str) -> None:
    with open(path, 'b+w') as __file:
        __file.write(data)

# CLEAR ########################################################################

def list_files(path: str, extension: str='') -> list:
    return [
        os.path.join(__dp, __f)
        for __dp, __dn, __fn in os.walk(path)
        for __f in __fn if extension in __f]

def clear_dir(path: str) -> None:
    __paths = list_files(path)
    for __p in __paths:
        os.remove(__p)

# RANDOM #######################################################################

def random_options(width_min: int=WIDTH_MIN, width_max: int=WIDTH_MAX) -> list:
    # choose the config randomly
    __width = '--width {width}'.format(width=random.randint(width_min, width_max))
    __braille = '--braille' if random.choice([True, False]) else ''
    __complex = '--complex' if random.choice([True, False]) else ''
    __dither = '--dither' if __braille and random.choice([True, False]) else ''
    __grayscale = '--grayscale' if random.choice([False]) else '' # colorless terminal
    __negative = '--negative' if random.choice([True, False]) else ''
    # chain all the options
    return [__width, __braille, __complex, __dither, __grayscale, __negative]

def format_args(options: list) -> list:
    return list(itertools.chain.from_iterable(__o.split(' ') for __o in options if __o))

def format_labels(options: list) -> list:
    return list(itertools.chain.from_iterable(__o.strip('--') for __o in options if __o))

# EXPORT #######################################################################

def export_table(table: iter, index: int, path: str=DATA_PATH) -> None:
    __path = os.path.join(path, '{index:0>4d}.parquet'.format(index=index))
    scrapscii.data.export_table_as_parquet(table=table, path=__path)

# CONVERT ######################################################################

def convert_shard(
    dataset: iter,
    table: iter=[],
    table_idx: int=0,
    table_len: int=TABLE_LEN,
    shard_len: int=SHARD_LEN,
    width_min: int=WIDTH_MIN,
    width_max: int=WIDTH_MAX,
    temp_path: str=TEMP_PATH,
    data_path: str=DATA_PATH,
) -> tuple:
    # current table
    __index = table_idx # index
    __table = list(table) # data

    # take a shard's worth of data
    __iter = itertools.islice(dataset, 0, shard_len)

    # iterate over the samples
    for __sample in tqdm.tqdm(__iter, total=shard_len):

        # parse the URL
        __url = __sample['url.txt']
        __path = format_path(url=__url, temp=temp_path)

        # download image from URL
        __bytes = download_image(__url)

        # check hex digest
        if is_valid_image(__bytes):
            export_image(data=__bytes, path=__path)
        else:
            tqdm.tqdm.write(f'Skip corrupted {__url}')
            continue

        # choose the config randomly
        __options = random_options(width_min=width_min, width_max=width_max)
        __args = format_args(__options)
        __labels = format_labels(__options)

        # choose a caption among the synthetic text
        __choice = random.randint(0, len(__sample['syn.json']['syn_text']) - 1)
        __caption = __sample['syn.json']['syn_text'][__choice]

        # convert the image to ASCII art
        __process = subprocess.run(['ascii-image-converter'] + __args + [__path], stdout=subprocess.PIPE)
        __content = __process.stdout.decode('utf-8')

        # check for conversion errors
        if is_valid_ascii(__content):
            __table.append({
                'caption': __caption,
                'content': __content,
                'labels': ','.join(__labels),
                'charsets': ','.join(set(scrapscii.unicode.lookup_section(__c) for __c in __content)),
                'chartypes': ','.join(set(scrapscii.unicode.lookup_category(__c) for __c in __content)),})
        else:
            tqdm.tqdm.write(f'Failed to convert {__url}')
            continue

        # chunk the dataset into shards
        if len(__table) >= table_len:
            # export as parquet
            export_table(table=__table, index=__index, path=data_path)
            # refresh
            __index += 1
            __table = []

    # return the remainder
    return (__index, __table)

# MAIN #########################################################################

if __name__ == '__main__':
    # init the table
    __index = 0 # index
    __table = [] # data

    # init the dataset
    __dataset = datasets.load_dataset('apple/DataCompDR-12M', split='train', cache_dir='~/.cache/huggingface/datasets', streaming=True)
    __iter = itertools.islice(__dataset, 0, TOTAL_LEN)

    # convert shard by shard
    while __iter:
        # export a shard
        __index, __table = convert_shard(
            dataset=__iter,
            table=__table,
            table_idx=__index,
            table_len=TABLE_LEN,
            shard_len=SHARD_LEN,
            width_min=WIDTH_MIN,
            width_max=WIDTH_MAX,
            temp_path=TEMP_PATH,
            data_path=DATA_PATH,)
        # remove the temp downloads (images)
        clear_dir(TEMP_PATH)
