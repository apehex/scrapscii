import hashlib
import io
import itertools
import json
import mimetypes
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

TIME_MAX = 0.1

WIDTH_MIN = 16
WIDTH_MAX = 128

SKIPS_LEN = 258000
TABLE_LEN = 2**4
SHARD_LEN = 2**6
TOTAL_LEN = 2**8

# IO ###########################################################################

TEMP_PATH = tempfile.mkdtemp()
DATA_PATH = os.path.realpath(os.path.join(os.path.dirname(__file__), '../', 'datasets/images'))

# FILTER BY EXT ################################################################

EXTENSION_LIST = ['jpeg', 'jpg', 'png', 'bmp', 'webp', 'tiff', 'tif', 'gif']

# CHECK ########################################################################

CORRUPTED_HASH = ['4dcb57651a75abfd07fb36c70c6c5108c49bdb34']

def is_valid_response(response: requests.models.Response) -> bool:
    return (
        bool(response)
        and type(response) == requests.models.Response
        and response.status_code == 200)

def is_valid_extension(extension: str, accepted: list=EXTENSION_LIST) -> bool:
    return (
        bool(extension)
        and type(extension) == str
        and extension.lower().strip('.') in accepted)

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
        and not 'error' in ascii.lower())

# DOWNLOAD #####################################################################

def download_image(url: str, timeout: int=1) -> requests.models.Response:
    __response = None
    # retrieve the image content as bytes
    try:
        __response = requests.get(url, timeout=timeout)
    # ignore exceptions
    except:
        __response = None
    # default
    return __response

def parse_content(response: requests.models.Response) -> bytes:
    __bytes = b''
    if is_valid_response(response):
        __bytes = response.content
    return __bytes

def parse_extension(response: requests.models.Response) -> str:
    __extension = ''
    # parse the header
    if is_valid_response(response):
        __headers = response.headers.get('content-type', '')
        __extension = mimetypes.guess_extension(__headers)
    # parse the URL
    if not is_valid_extension(__extension):
        __path = urllib.parse.urlparse(response.url).path
        __filename = __path.split('/')[-1]
        __extension = os.path.splitext(__filename)[-1]
    # favor the information coming from the header
    return __extension

def format_path(url: str, extension: str, temp: str=TEMP_PATH) -> str:
    # reduce the filename to a fixed size
    __hash = hashlib.sha1(url.encode('utf-8')).hexdigest()
    # safe path
    return os.path.join(temp, __hash + '.' + extension.strip('.'))

def export_image(data: bytes, path: str) -> None:
    with open(path, 'b+w') as __file:
        __file.write(data)

# STATS ########################################################################

def init_stats(
    index: int=0,
    saved: int=0,
    skipped: int=0,
    valid: int=0,
    response: int=0,
    extension: int=0,
    image: int=0,
    asciiart: int=0
) -> dict:
    return {
        'index': index,
        'total': max(saved, skipped),
        'saved': saved,
        'skipped': skipped,
        'valid': valid,
        'invalid': {
            'response': response,
            'extension': extension,
            'image': image,
            'asciiart': asciiart,},}

def update_stats(
    stats: dict,
    index: int=0,
    saved: int=0,
    skipped: int=0,
    valid: int=0,
    response: int=0,
    extension: int=0,
    image: int=0,
    asciiart: int=0
) -> dict:
    return {
        'index': stats['index'] + index,
        'total': stats['total'] + skipped + valid + response + extension + image + asciiart,
        'saved': saved or stats['saved'], # keep the latest
        'skipped': stats['skipped'] + skipped,
        'valid': stats['valid'] + valid,
        'invalid': {
            'response': stats['invalid']['response'] + response,
            'extension': stats['invalid']['extension'] + extension,
            'image': stats['invalid']['image'] + image,
            'asciiart': stats['invalid']['asciiart'] + asciiart,},}

def format_stats(stats: dict) -> str:
    return 'index={index} total={total} saved={saved} skipped={skipped} valid={valid} invalid={invalid} (response={response} extension={extension} image={image} asciiart={asciiart})'.format(
        index=stats['index'],
        total=stats['total'],
        saved=stats['saved'],
        skipped=stats['skipped'],
        valid=stats['valid'],
        invalid=sum([__v for __v in stats['invalid'].values()]),
        response=stats['invalid']['response'],
        extension=stats['invalid']['extension'],
        image=stats['invalid']['image'],
        asciiart=stats['invalid']['asciiart'],)

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
    __color = '--color' if random.choice([True, False]) else ''
    __complex = '--complex' if random.choice([True, False]) else ''
    __dither = '--dither' if __braille and random.choice([True, False]) else ''
    __grayscale = '--grayscale' if random.choice([False, True]) else ''
    __negative = '--negative' if random.choice([True, False]) else ''
    __threshold = '--threshold {threshold}'.format(threshold=random.randint(64, 192)) if __braille and random.choice([True, False]) else ''
    # chain all the options
    return [__width, __braille, __color, __complex, __dither, __grayscale, __negative, __threshold]

def format_args(options: list) -> list:
    return list(itertools.chain.from_iterable(__o.split(' ') for __o in options if __o))

def format_labels(options: list) -> list:
    return list(itertools.chain.from_iterable(__o.strip('--') for __o in options if __o))

# EXPORT #######################################################################

def export_table(table: iter, index: int, path: str=DATA_PATH) -> None:
    __path = os.path.join(path, '{index:0>4d}.parquet'.format(index=index))
    scrapscii.data.export_table_as_parquet(table=table, path=__path)

# CONVERT ######################################################################

def convert_image(path: str, options: list, timeout: int=TIME_MAX) -> str:
    __ascii = ''
    # run binary tool
    try:
        __process = subprocess.run(['ascii-image-converter'] + options + [path], stdout=subprocess.PIPE, timeout=timeout)
        __ascii = __process.stdout.decode('utf-8')
    # timeout longer executions
    except:
        __ascii = ''
    # default
    return __ascii

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
    time_max: int=TIME_MAX,
) -> tuple:
    # current table
    __index = table_idx # index
    __table = list(table) # data

    # take a shard's worth of data
    __iter = itertools.islice(dataset, 0, shard_len)

    # track progress
    __pbar = tqdm.tqdm(__iter, total=shard_len, smoothing=0.0)
    __stats = init_stats()

    # iterate over the samples
    for __j, __sample in enumerate(__pbar):

        # parse the URL
        __url = __sample['url.txt']

        # download image from URL
        __response = download_image(__url, timeout=time_max)
        if not is_valid_response(__response):
            __stats = update_stats(stats=__stats, response=1)
            __pbar.set_postfix_str(format_stats(__stats), refresh=True)
            continue

        # parse the extension
        __extension = parse_extension(__response)
        if not is_valid_extension(__extension):
            __stats = update_stats(stats=__stats, extension=1)
            __pbar.set_postfix_str(format_stats(__stats), refresh=True)
            continue

        # parse the image content
        __bytes = parse_content(__response)
        if not is_valid_image(__bytes):
            __stats = update_stats(stats=__stats, image=1)
            __pbar.set_postfix_str(format_stats(__stats), refresh=True)
            continue

        # save to disk
        __path = format_path(url=__url, extension=__extension, temp=temp_path)
        export_image(data=__bytes, path=__path)
        
        # choose the config randomly
        __options = random_options(width_min=width_min, width_max=width_max)
        __args = format_args(__options)
        __labels = format_labels(__options)

        # choose a caption among the synthetic text
        __choice = random.randint(0, len(__sample['syn.json']['syn_text']) - 1)
        __caption = __sample['syn.json']['syn_text'][__choice]

        # convert the image to ASCII art
        __content = convert_image(path=__path, options=__args, timeout=time_max)
        if not is_valid_ascii(__content):
            __stats = update_stats(stats=__stats, asciiart=1)
            __pbar.set_postfix_str(format_stats(__stats), refresh=True)
            continue

        # add a row
        __stats = update_stats(stats=__stats, valid=1)
        __pbar.set_postfix_str(format_stats(__stats), refresh=True)
        __table.append({
            'caption': __caption,
            'content': __content,
            'labels': ','.join(__labels),
            'charsets': ','.join(set(scrapscii.unicode.lookup_section(__c) for __c in __content)),
            'chartypes': ','.join(set(scrapscii.unicode.lookup_category(__c) for __c in __content)),})

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
        # skip samples that are already processed
        __skip = itertools.islice(__iter, 0, SKIPS_LEN)
        for _ in __skip:
            pass
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
            data_path=DATA_PATH,
            time_max=TIME_MAX,)
        # remove the temp downloads (images)
        clear_dir(TEMP_PATH)
