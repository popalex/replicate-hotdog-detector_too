# Hotdog Detector

A Cog-based machine learning model that uses Stable Diffusion XL to generate images from text prompts.

## Overview

This project wraps the Stable Diffusion XL base model in a Cog predictor, making it easy to deploy and run predictions via API.

## Requirements

- GPU-enabled machine
- Python 3.10
- Cog ([installation instructions](https://github.com/replicate/cog))

## Setup

1. Install Cog if you haven't already:
```bash
sudo curl -o /usr/local/bin/cog -L https://github.com/replicate/cog/releases/latest/download/cog_`uname -s`_`uname -m`
sudo chmod +x /usr/local/bin/cog
```

2. Build the Docker image:
```bash
cog build
```

## Usage

### Run a prediction

```bash
cog predict -i prompt="a photo of a hotdog" -i num_steps=30
```

### Start the API server

```bash
cog run -p 5000 python -m cog.server.http
```

Then make predictions via HTTP:

```bash
curl http://localhost:5000/predictions -X POST \
  -H 'Content-Type: application/json' \
  -d '{"input": {"prompt": "a photo of a hotdog", "num_steps": 30}}'
```

## Model Details

- **Base Model**: [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
- **Framework**: Diffusers
- **Precision**: FP16 for GPU efficiency

## Configuration

See `cog.yaml` for the full build configuration including Python packages and system dependencies.

## Output

Generated images are saved to `/tmp/out.png` and the file path is returned.
