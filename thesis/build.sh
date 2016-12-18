#!/bin/bash

# use `FMT=tex ./build-paper.sh` to get the .tex file

export TEXINPUTS=wissdoc:

pandoc \
	thesis.md \
	--filter pandoc-crossref \
	--filter pandoc-citeproc \
	--standalone \
	--template wissdoc/diplarb.tex \
	--top-level-division=chapter \
	-o thesis.${FMT:-pdf}
