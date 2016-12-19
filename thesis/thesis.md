---
# this document is pandoc 1.18 flavored markdown
title: Backchannel Prediction for Conversational Speech
author: Robin
---

# Introduction

Motivation, Goals

# Methodology and Implementation

using Janus, Theano, etc.

# Extraction

## Backchannel selection

- list of 200 most common BCs; with silence before
- clean up `[noise]` and `[laughter]`

## Prediction outputs

- expect `1` shortly before BC, `0` 2 seconds earlier
- bell curve

# Training

## Feed forward

- softmax with categorical crossentropy for categorical output (1=BC, 0=NBC)
- sigmoid with mean squared error for bell curve output


validate error functions etc.

## RNN / LSTM

# Postprocessing

low-pass filter, trigger BC at maxima

use random BC sample from training data

# Evaluation

# Future Work

predict specific BCs / categories
