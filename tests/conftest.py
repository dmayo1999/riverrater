"""Shared test fixtures and configuration for RiverRater test suite."""

import os

# Ensure Qt uses offscreen rendering in headless CI environments.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
