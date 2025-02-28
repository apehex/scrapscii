import json
import os

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.lib as pl
import pyarrow.parquet as pq

# SCHEMA ######################################################################

SCHEMA = pa.schema([
    pl.field('caption', pa.string()),
    pl.field('content', pa.string()),
    pl.field('labels', pa.string()),
    pl.field('charsets', pa.string()),
    pl.field('chartypes', pa.string()),])

# EXPORT #######################################################################

def export_table_as_parquet(table: iter, path: str, schema: pl.Schema=SCHEMA) -> None:
    pq.write_table(
        table=pl.Table.from_pylist(
            mapping=table,
            schema=schema),
        where=path)

# CONVERT ######################################################################

def cast_json_to_parquet(path: str, schema: pl.Schema=SCHEMA) -> None:
    # change the extension
    __path = os.path.splitext(path)[0] + '.parquet'
    # import the JSON data
    with open(path, 'r') as __file:
        __data = json.load(__file)
    # export as parquet
    export_table_as_parquet(table=__data, path=__path, schema=schema)
