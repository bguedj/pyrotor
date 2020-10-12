# !/usr/bin/env python
# -*- coding:utf-8 -*-

"""
Simple logging utility.
"""


def log(message, verbose):
    """Print a message depending on verbose state"""
    if verbose:
        print(message)
