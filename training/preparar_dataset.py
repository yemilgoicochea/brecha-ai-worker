"""Genera dataset_limpio.csv a partir del scraping nacional de proyectos.

El CSV de entrada (resultado_nacional.csv) proviene del scraper de
webscraping-brechas/get-detail-ssi-by-cui y NO está en este repo:
ajusta PATH_NACIONAL a donde lo tengas localmente.
"""
from pathlib import Path

import pandas as pd

# ── RUTAS ──────────────────────────────────────────────────────────────────────
PATH_NACIONAL = r"d:\UPC\PROJECT\Scripts\webscraping-brechas\get-detail-ssi-by-cui\resultado_nacional.csv"
BASE          = Path(__file__).parent
PATH_SECTORS  = BASE / "sectors.csv"
PATH_OUTPUT   = BASE / "dataset_limpio.csv"

# ── 1. CARGAR Y UNIFICAR ───────────────────────────────────────────────────────
print("Cargando CSV nacional...")
df = pd.read_csv(PATH_NACIONAL, encoding="utf-8-sig", low_memory=False)
print(f"  Total filas: {len(df):,}")

# ── 2. CARGAR SECTORES ─────────────────────────────────────────────────────────
sectors = pd.read_csv(PATH_SECTORS, encoding="utf-8-sig")
# Mapeo: "11: SALUD" → "HEALTH"
transparency_map = dict(zip(
    sectors["transparency_name"].str.strip().str.upper(),
    sectors["code"]
))
print(f"  Sectores cargados: {len(sectors)}")

# ── 3. LIMPIAR ─────────────────────────────────────────────────────────────────
col_nombre = "Nombre de la inversion(proyecto)"
col_sector = "Sector"
col_cui    = "CUI"

df = df[[col_cui, col_nombre, col_sector]].copy()
df = df[df[col_nombre].notna() & (df[col_nombre].str.strip() != "")]
df = df[df[col_sector].notna() & (df[col_sector].str.strip() != "")]
df = df.drop_duplicates(subset=[col_cui])
print(f"  Filas únicas con nombre y sector: {len(df):,}")

# ── 4. MAPEAR SECTOR → CODE ────────────────────────────────────────────────────
df["sector_upper"] = df[col_sector].str.strip().str.upper()
df["sector_code"]  = df["sector_upper"].map(transparency_map)

# Ver cuántos no mapearon
sin_map = df[df["sector_code"].isna()]
print(f"\n  Sin mapeo ({len(sin_map):,}):")
print(sin_map["sector_upper"].value_counts().head(10))

# Filtrar solo los que mapearon
df = df[df["sector_code"].notna()].copy()
print(f"\n  Con sector mapeado: {len(df):,}")

# ── 5. GUARDAR ─────────────────────────────────────────────────────────────────
df = df[[col_cui, col_nombre, "sector_code"]].copy()
df.columns = ["cui", "nombre_proyecto", "sector_code"]
df.to_csv(PATH_OUTPUT, index=False, encoding="utf-8-sig")

print(f"\nDataset guardado en: {PATH_OUTPUT}")
print(f"\nDistribución por sector:")
print(df["sector_code"].value_counts())
