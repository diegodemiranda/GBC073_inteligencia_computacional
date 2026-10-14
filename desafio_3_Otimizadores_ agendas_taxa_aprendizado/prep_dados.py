"""Uso: python prep_dados.py [caminho_do_csv_bruto]   → gera data/features.csv (sem vazamento).
Opcional: os outros scripts já chamam isso sozinhos se só houver o CSV bruto."""
import sys
from dados import gerar, localiza, PROC
if len(sys.argv) > 1: gerar(sys.argv[1], PROC)
else: print("OK:", localiza())
