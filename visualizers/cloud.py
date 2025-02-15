import os
import re
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud, ImageColorGenerator
import logging
from visualizers.utils import save_fig, get_stopwords
from utils import load_data
from PIL import Image

log = logging.getLogger(__name__)


def main(args):
    # load data
    df = load_data(args)
    # load stopwords
    stopwords = get_stopwords(args.stopword_paths)
    if args.split_by_outgoing:
        # outgoing
        plt.subplot(121)
        plt.title('Outgoing')
        text_out = cleanup_text(df[df.outgoing == True]['text'], stopwords)
        render_wordcloud(args, text_out)
        # incoming
        plt.subplot(122)
        plt.title('Incoming')
        text_in = cleanup_text(df[df.outgoing == False]['text'], stopwords)
        render_wordcloud(args, text_in)
    else:
        # clean up data
        text = cleanup_text(df['text'], stopwords)
        # render word cloud
        render_wordcloud(args, text)
    # save fig
    plt.tight_layout()
    save_fig(plt.gcf(), 'cloud', dpi=args.dpi)


def render_wordcloud(args, text):
    log.info('Rendering word cloud...')
    mask = np.array(Image.open(os.path.join(args.mask_image)))
    wc = WordCloud(background_color="white", mask=mask, max_words=args.num_words, stopwords=None, normalize_plurals=False, collocations=True).generate(text)
    image_colors = ImageColorGenerator(mask)
    wc = wc.recolor(color_func=image_colors)
    plt.imshow(wc)
    plt.axis("off")


def cleanup_text(text, stopwords):
    # pre-compiled regex is faster than going through a list
    stopwords = '|'.join(stopwords)
    regex = re.compile(r'\b(' + stopwords + r')\b', re.UNICODE)
    log.info('Cleaning up data...')
    text = text \
        .replace(to_replace='None', value=np.nan).dropna() \
        .str.lower() \
        .apply(lambda w: re.sub(r'^https?:\/\/.*[\r\n]*', '', w)) \
        .apply(lambda w: regex.sub('', w))
    text = ' '.join(text)
    return text
