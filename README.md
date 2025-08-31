# StreaMAX - a JAX-accelerated stream generator

This package serves as a fast and efficient simulator of stellar streams. The goal is to make the modeling fast enough for Bayesian inference. 

## Models

StreaMAX offers the following 4 modeling frameworks all JAX-compiled:

    - Spray: from spray import generate_stream_spray
    - Streak: from streak import generate_stream_streak
    - 1st order: to come

## Quick Start

Look at the quick_start.ipynb to see how to define the parameters and generate a stream from the aformentioned methods. 

## Installation

```bash
# Clone the repository
git clone https://github.com/David-Chemaly/StreaMAX.git
cd StreaMAX

# Install dependencies
pip install -r requirements.txt
```