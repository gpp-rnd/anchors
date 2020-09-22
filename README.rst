=======
anchors
=======


.. image:: https://img.shields.io/pypi/v/anchors.svg
        :target: https://pypi.python.org/pypi/anchors

.. image:: https://img.shields.io/travis/gpp-rnd/anchors.svg
        :target: https://travis-ci.com/gpp-rnd/anchors

.. image:: https://readthedocs.org/projects/anchors/badge/?version=latest
        :target: https://anchors.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status




Python package for calculating scores from ancnchor or modifier screens


* Free software: MIT license
* Documentation: https://anchors.readthedocs.io.

Tutorial
--------
To install::

    $ pip install anchors

Basic Usage
^^^^^^^^^^^
.. code:: python

    import pandas as pd
    from anchors import get_guide_residuals, get_gene_residuals
    lfc_df = pd.read_csv('https://raw.githubusercontent.com/PeterDeWeirdt/anchor_screen_parp_lfcs/master/parp_example_lfcs.csv')
    refernce_condition_df = pd.read_csv('https://raw.githubusercontent.com/PeterDeWeirdt/anchor_screen_parp_lfcs/master/parp_example_mapping.csv')
    guide_residuals, model_info, model_fit_plots = get_guide_residuals(lfc_df, refernce_condition_df)
    guide_mapping_df = pd.read_csv('https://raw.githubusercontent.com/PeterDeWeirdt/anchor_screen_parp_lfcs/master/brunello_guide_map.csv')
    gene_residuals = get_gene_residuals(guide_residuals, guide_mapping_df)

Features
--------

* TODO

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
