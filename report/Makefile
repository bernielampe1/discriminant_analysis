
.PHONY: all
all:
	pdflatex report.tex
	bibtex report
	pdflatex report.tex
	pdflatex report.tex

.PHONY: clean
clean:
	rm -f *.log *.aux *.bbl *.blg

