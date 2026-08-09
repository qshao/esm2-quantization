#!/bin/bash
# Build the quantization report.
#
# Use the TeX Live 2020 binaries directly rather than `pdflatex` from PATH.
# The cluster's /share/singularity/bin/pdflatex is a wrapper around a
# singularity image whose pdflatex app emits DVI, not PDF -- it exits 0 and
# writes report.dvi, so the failure looks like success until you go looking for
# the PDF. `module load ccs/latex/texlive-2020` does not shadow it either.
set -euo pipefail

export PATH=/opt/ohpc/pub/libs/ccs/texlive/2020/bin/x86_64-linux:$PATH
cd "$(dirname "$0")"

# Twice, so the table/section cross-references resolve.
pdflatex -interaction=nonstopmode quantization_report.tex >/dev/null
pdflatex -interaction=nonstopmode quantization_report.tex >/dev/null

rm -f quantization_report.aux quantization_report.out
grep -c Overfull quantization_report.log | xargs echo "overfull boxes:"
# Undefined refs are silent in the PDF -- they render as "??" and are easy to
# miss in a 17-page document.
grep -E "Warning.*(undefined|Citation)" quantization_report.log | sort -u || echo "no undefined refs"
grep "Output written" quantization_report.log
