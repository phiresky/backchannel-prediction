# Makefile for pictures
# $Id$
# $HeadURL$
# $@ target name
# $< first dependency
# $? all newer dependency
# $^ all dependencies separated by space
# $* stem with which an implicit rule matches

PS2EPS = ps2eps

PS2EPSOPTS = -q -f -l -s a2

BILDERSRC_DIR = src
BILDER_DIR = .

FIG2DEV = fig2dev

BILDERPRN = $(wildcard $(BILDERSRC_DIR)/*.prn)
BILDERPPT = $(wildcard $(BILDERSRC_DIR)/*.ppt)
BILDERFIG = $(wildcard $(BILDERSRC_DIR)/*.fig)
BILDERFIG2EPS = $(notdir $(subst .fig,.eps,$(BILDERFIG)))
BILDERFIG2PDF = $(notdir $(subst .fig,.pdf,$(BILDERFIG)))
BILDERPRN2EPS = $(notdir $(subst .prn,.eps,$(BILDERPRN)))
BILDERPPT2PRN = $(notdir $(subst .ppt,.prn,$(BILDERPPT)))


vpath %.fig src
vpath %.prn src
vpath %.ppt src

default: fig eps pdf

%.eps :: %.prn
	$(PS2EPS) $(PS2EPSOPTS) $?
	@mv -f $(BILDERSRC_DIR)/$@ .

%.prn :: %.ppt
	@echo "Need to re-print:" $?

%.eps :: %.fig
	$(FIG2DEV) -L eps $? > $(BILDER_DIR)/$@

%.pdf :: %.fig
	$(FIG2DEV) -L pdf $? > $(BILDER_DIR)/$@

fig: $(BILDERFIG2EPS)

pdf: $(BILDERFIG2PDF)

eps: $(BILDERPRN2EPS) $(BILDERPPT2PRN)

$(BILDERFIG2EPS) : %.eps: %.fig

$(BILDERFIG2PDF) : %.pdf: %.fig

$(BILDERPRN2EPS) : %.eps: %.prn

$(BILDERPPT2PRN) : %.prn: %.ppt

.PHONY: eps fig pdf
