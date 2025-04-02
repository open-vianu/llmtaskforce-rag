#!/usr/bin/env python

import argparse
import re
import pandas as pd
import plotnine as p9


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', default='eval/RESULTS.md')
    args = parser.parse_args()
    
    with open(args.input) as fileobj:
        table = parse_table(fileobj)
        plot = (p9.ggplot(table, p9.aes(x = 'embedding_model', y = 'percent_correct', fill = 'chat_model'))
            + p9.geom_col(stat='identity', position='dodge'))
        plot.save('RESULTS.png', dpi=600)


def parse_table(markdown_file):
    rows = []
    for line in markdown_file:
        if line.startswith('|'):
            parts = line.strip('|\n').split('|')
            if len(parts) != 6:
                continue
            rows.append([part.strip() for part in parts])

    df = pd.DataFrame(rows[2:], columns=['embedding_model', 'chat_model', 'date', 'percent_correct', 'evaluator_model', 'dataset'])
    df['percent_correct'] = df['percent_correct'].astype(float)
    return df
            

if __name__ == '__main__':
    import sys
    sys.exit(main())
    