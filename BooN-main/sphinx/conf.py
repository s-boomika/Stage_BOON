# Configuration file for the Sphinx documentation builder.

# Configuration file for the Sphinx documentation builder.
 
import os
import sys
#sys.path.insert(0, os.path.abspath('C:\\Users\\franck.delaplace\\OneDrive - Universite Evry Val d\'Essonne\\BooN'))
sys.path.insert(0, os.path.abspath(r'C:\\Users\\boomi\\Documents\\Boomika_Files\\M1_GENIOMHE\\P3\\Modelisation_simulation\\Stage_BOON\\BooN-main'))

project = 'BooN'
copyright = '2024, Franck Delaplace'
author = 'Franck Delaplace'
release = '1.05'

# -- General configuration -------------------------------
# linkcode_url = '{{ repo }}/blob/master/{{ object }}#L{{ lineno }}'   #Git Hub theme

extensions = [
    'sphinx.ext.autodoc',
    'sphinx_rtd_theme',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
]
 
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# autodoc_param_description = [
#     'param',
#     'type',
#     'return',
#     'rtype'
#     'default'
# ]
