# Pothole Detection

This site documents a data pipeline and tooling for pothole detection. The
project ingests raw mobile sensor sessions (accelerometer, gyroscope, GPS, and
labels), merges and resamples them into clean session CSVs, and can generate
spectrogram samples for training and analysis. It includes a CLI for listing,
processing, plotting, and video generation, plus a Python API for building
custom workflows.

## Main sections
- Getting started: environment setup and a basic end-to-end flow.
- Data layout: how raw sessions are organized, the required input files, and
  the generated outputs (session CSVs and sample NPZs).
- CLI usage: command reference for loader, plotter, and video tools.
- API reference: module-level documentation for programmatic use.
