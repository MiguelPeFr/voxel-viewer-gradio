---
title: Voxel Model Viewer
emoji: 🎲
colorFrom: gray
colorTo: indigo
sdk: gradio
sdk_version: "5.13.1"
app_file: viewer.py
pinned: false
---

# Voxel Model Viewer

A web application built with Gradio that allows users to view and interact with 3D voxel models in .vox format.

## Features

- Upload and view .vox files in 3D
- Interactive 3D visualization with rotation and zoom
- Dark theme interface for better contrast
- Support for colored voxels using the model's palette

## Usage

1. Upload a .vox file using the file upload interface
2. The model will be displayed in an interactive 3D viewer
3. Use mouse controls to rotate and zoom the model

## Technical Details

This application uses:
- Gradio for the web interface
- PyVox for parsing .vox files
- Plotly for 3D visualization
- NumPy for array operations

## Example

An example .vox file (modelo_optimizado.vox) is included in the repository for testing purposes.