import os

import scrapscii.data

# CONSTANTS ####################################################################

ROOT_PATH = os.path.realpath(os.path.join(os.path.dirname(__file__), '../', 'datasets'))

# BROWSE #######################################################################

ALL_PATH = [
    os.path.join(__dp, __f)
    for __dp, __dn, __fn in os.walk(ROOT_PATH)
    for __f in __fn if os.path.splitext(__f)[-1] == '.json']

# CAST #########################################################################

for __p in ALL_PATH:
    scrapscii.data.cast_json_to_parquet(path=__p)
