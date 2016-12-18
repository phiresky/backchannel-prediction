#################################################################
# Makefile (use GNU-Make)
#
# $* file name part of the current dependent with suffix deleted
#    (only for suffix rules)
# $@ full target name
# $< out-of-date module (only for suffix rules)
# $? list of components to be rebuilt
#
#################################################################

.DEFAULT:
	  @echo "This target needs remaking: default = [$<]"
	  @echo "Do you really use GNU-Make?"

MAINFILE       := diplarb
TITLEPAGE      := titelseite
TEXSOURCES     := $(wildcard *.tex)
BIBSRCFILES    := $(wildcard *.bib) 
BBLFILES       := $(subst bib,bbl,$(BIBSRCFILES))
AUXFILES       := $(subst tex,aux,$(TEXSOURCES))
SCRIPTDIR      := scripts

ISO_8859_DIR    := iso-8859-1

SOURCES = $(TEXSOURCES) $(BIBSRCFILES)\
	  wissdoc.cls\
	  svnver.sty\
	  diplarb.ist\
	  itmalpha.bst\
	  itmabbrv.bst\
	  Makefile\
	  README

DISTFILES = $(SOURCES) $(SCRIPTDIR) logos images $(ISO_8859_DIR)

$(MAINFILE).pdf: $(MAINFILE).tex wissdoc.cls $(TEXSOURCES) $(BBLFILES)
	- pdflatex $(MAINFILE)

final: it bbls double index

it:
	@rm -f $(MAINFILE).pdf
	- pdflatex $(MAINFILE)

bbls: $(BBLFILES)

$(BBLFILES): %.bbl: %.bib
	- pdflatex $(MAINFILE)
	- bibtex $(MAINFILE)

double: $(MAINFILE).tex
	 - pdflatex $(MAINFILE)
	 - pdflatex $(MAINFILE)

index: $(MAINFILE).ind

#makeindex kann keine 8-bit Umlaute richtig sortieren! :-(
#deswegen setzen wir sie für makeindex einfach nochmal um
$(MAINFILE).ind: $(MAINFILE).idx $(MAINFILE).ist
	- mv $(MAINFILE).idx $(MAINFILE).idx.orig
	- $(SCRIPTDIR)/recodetexumlaut <$(MAINFILE).idx.orig >$(MAINFILE).idx
	- makeindex -c -g -s $(MAINFILE).ist $(MAINFILE)
	- mv $(MAINFILE).idx.orig $(MAINFILE).idx


finalpdf: pdf
	- thumbpdf $(MAINFILE)
	- pdflatex $(MAINFILE)

pdf:    $(MAINFILE).pdf  $(MAINFILE).ind

fullrun: $(MAINFILE).tex $(TEXSOURCES)
	- rm $(MAINFILE).aux $(MAINFILE).ind
	- pdflatex $(MAINFILE)
	- make bbls
	- pdflatex $(MAINFILE)
	- $(MAKE) index
	- pdflatex $(MAINFILE)

zipps: $(MAINFILE).pdf
	       gzip -p $(MAINFILE).pdf

clean:
	- find . | egrep ".*((\.(aux|idx|ind|ilg|log|blg|bbl|toc|lof|lot|dvi|tlf|tlt))|~)$$" | xargs rm
	- rm -f $(MAINFILE).out thumb???.png thumbdta.tex thumbpdf.*

archive:
	- @ echo "Making archive..."
	- gtar -czvf $(MAINFILE).tar.gz $(SOURCES)  images/*.pdf logos/*.pdf Makefile 
	- @ echo "Archive ready."

dist:
	- @ echo "Making distribution..."
	- tar --exclude .svn -czvf wissdoctemplate.tar.gz $(DISTFILES) 
	- mkdir wissdoc; cd wissdoc; tar -xzvf ../wissdoctemplate.tar.gz;\
	  cd ..; tar -czvf wissdoctemplate.tar.gz wissdoc; rm -rf wissdoc
	- @ echo "Archive ready."

latin1:
	- if [[ ! -d $(ISO_8859_DIR) ]] ; then mkdir $(ISO_8859_DIR); fi
	- for file in *.tex *.cls; do uconv -f utf-8 -t iso-8859-15 $$file -o $(ISO_8859_DIR)/$$file; done;
	- perl -pe 's/utf8/latin1/g;s/UTF-8/ISO 8859-1/g' <$(ISO_8859_DIR)/wissdoc.cls >$(ISO_8859_DIR)/wissdoc.cls.new && mv $(ISO_8859_DIR)/wissdoc.cls.new $(ISO_8859_DIR)/wissdoc.cls

.PHONY: final it bbls double index finalpdf pdf fullrun zipps clean archive dist latin1
#end of Makefile
