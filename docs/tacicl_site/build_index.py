"""Wrap page.html (the Artifact source, which has no document skeleton) into a standalone index.html for GitHub Pages.
Usage: python docs/tacicl_site/build_index.py"""
from pathlib import Path

here = Path(__file__).parent
body = (here / 'page.html').read_text()
head = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        '<meta name="description" content="TacICL: in-context tactile insertion with a frozen vision-language model. '
        'Vision gets it close. Touch gets it in.">\n'
        '<style>:root{color-scheme:light dark}body{margin:0}img{max-width:100%}[hidden]{display:none!important}</style>\n')
(here / 'index.html').write_text(head + body.replace('<style>', '</head>\n<body>\n<style>', 1) + '\n</body>\n</html>\n')
