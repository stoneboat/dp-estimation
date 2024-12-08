#!/usr/bin/env python
# encoding: utf-8
"""
Constants.py
"""
WORKERS = 24

SAMPLES_BASE = 2  # estimated convergence curves (if exists) will be generated for sequential powers of SAMPLES_BASE
BATCH_SAMPLES = int(SAMPLES_BASE ** 13)