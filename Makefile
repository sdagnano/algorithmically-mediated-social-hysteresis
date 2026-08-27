.PHONY: all figures submission paper verify manifest release clean

PYTHON ?= python
PDFLATEX ?= pdflatex

all: release

figures:
	$(PYTHON) code/simulate.py
	$(PYTHON) code/model_i_stress.py
	$(PYTHON) code/adaptive_network_stress.py
	$(PYTHON) code/design_power.py

submission: figures
	$(PYTHON) code/prepare_jcn_submission.py

paper: submission
	cd submission-jcn && $(PDFLATEX) -interaction=nonstopmode -halt-on-error submission.tex
	cd submission-jcn && $(PDFLATEX) -interaction=nonstopmode -halt-on-error submission.tex
	cd submission-jcn && $(PDFLATEX) -interaction=nonstopmode -halt-on-error submission.tex

manifest: paper
	$(PYTHON) code/build_release_manifest.py

verify: manifest
	$(PYTHON) code/verify_archive.py

release: verify

clean:
	$(RM) submission-jcn/*.aux submission-jcn/*.log submission-jcn/*.out submission-jcn/*.synctex.gz submission-jcn/*.xdv submission-jcn/submission.pdf
