#!/bin/bash
# Build the preprint and package it for arXiv submission.
#
# Use the TeX Live 2020 binaries directly rather than `pdflatex` from PATH.
# The cluster's /share/singularity/bin/pdflatex wraps a singularity image whose
# pdflatex app emits DVI, not PDF -- it exits 0 and writes a .dvi, so the
# failure looks like success until you go looking for the PDF.
set -euo pipefail

export PATH=/opt/ohpc/pub/libs/ccs/texlive/2020/bin/x86_64-linux:$PATH
cd "$(dirname "$0")"
DOC=esm2_quantization

# Three passes: natbib citation labels need two, cross-references settle on the
# third once the figures have displaced the float positions.
for i in 1 2 3; do
    pdflatex -interaction=nonstopmode "$DOC.tex" >/dev/null
done

echo "overfull boxes: $(grep -c Overfull "$DOC.log" || true)"
grep -E "Warning.*(undefined|Citation)" "$DOC.log" | sort -u || echo "no undefined refs/citations"
grep "Output written" "$DOC.log"

# arXiv wants the SOURCE, not the PDF: a tarball of .tex plus figures. It runs
# its own pdflatex, so .aux/.log/.out must be excluded, but .bbl would be needed
# if this used BibTeX -- it does not, the bibliography is inline in the .tex.
tar --format=ustar -czf arxiv-submission.tar.gz "$DOC.tex" fig_tail.pdf fig_fidelity.pdf fig_pareto.pdf
echo "arxiv tarball: $(ls -lh arxiv-submission.tar.gz | awk '{print $5}')"
tar -tzf arxiv-submission.tar.gz | sed 's/^/   /'
