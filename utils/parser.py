import argparse as ag
import json

def get_parser_with_args(metadata_json='metadata.json'):
    parser = ag.ArgumentParser(description='Training change detection network')

    with open(metadata_json, 'r') as fin:
        metadata = json.load(fin)
        parser.set_defaults(**metadata)
        parser.add_argument('--local_rank', default=-1, type=int,
                            help='node rank for distributed training')
        return parser, metadata

    return None
