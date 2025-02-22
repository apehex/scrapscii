import json
import os
import random

import art
import scrapscii.unicode

# META #########################################################################

CAPTION = '`{text}` in {font} font, with {spacing} spacing and {decoration} decoration'
LABELS = '{font} font, {spacing} spacing, {decoration} decoration'
PATH = os.path.realpath(os.path.join(os.path.dirname(__file__), '../', 'datasets/graffiti/{font}.json'))

# SAMPLES ######################################################################

SAMPLES = {
    'alphabet': ''.join(chr(__c) for __c in range(32, 127)),
    'lorem': 'Lorem ipsum dolor sit amet, consectetur adipiscing elit.',
    'special': 'os.path.realpath(\n    os.path.join(os.path.dirname(__file__),\n    "../",\n    "datasets/graffiti/asciiart.json"))',
    'wiki': 'L’art ASCII consiste à réaliser des images uniquement à l\'aide des lettres et caractères spéciaux contenus dans le code ASCII.',}

# MAIN #########################################################################

if __name__ == '__main__':

    for __font in art.params.FONT_MAP.keys():
        __dataset = []

        # GENERATE #############################################################

        for __text in SAMPLES.values():
            __decorations = random.sample(sorted(art.params.DECORATIONS_MAP.keys()), 3) + ['no']
            __spacings = random.sample(range(0,4), 2)
            for __d in __decorations:
                for __s in __spacings:
                    __caption = CAPTION.format(text=__text, font=__font, spacing=__s, decoration=__d)
                    __content = art.text2art(__text, font=__font, space=__s, decoration=__d, chr_ignore=True)
                    __labels = LABELS.format(font=__font, spacing=__s, decoration=__d)
                    __dataset.append({
                        'caption': __caption,
                        'content': __content,
                        'labels': __labels,
                        'charsets': ','.join(set(scrapscii.unicode.lookup_section(__c) for __c in __content)),
                        'chartypes': ','.join(set(scrapscii.unicode.lookup_category(__c) for __c in __content)),})

        # EXPORT ###############################################################

        with open(PATH.format(font=__font), 'w') as __file:
            json.dump(__dataset, __file)
