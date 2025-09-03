## suisca-utils

Utilidades y scripts para extracción de tablas en PDF, comparación de sugerencias en Excel y utilidades de correo/IMAP.

### Requisitos
- Python 3.10+
- Instalar dependencias: `pip install -r requirements.txt`
- Para extracción de tablas con Camelot (flavor lattice) puede requerirse Ghostscript o backend PDFium según la configuración.

---

## Scripts principales

### 1) `app.py` — Camelot PDF Table Extractor (básico)
- Descripción: aplicación Streamlit para extraer tablas desde PDFs usando Camelot con parámetros esenciales (flavors: `stream` o `lattice`).
- Ejecutar: `streamlit run app.py`
- Uso: sube un PDF, ajusta parámetros en la barra lateral y descarga las tablas como CSV.

### 2) `app_complete.py` — Camelot (avanzado: stream/lattice/network/hybrid)
- Descripción: aplicación Streamlit más completa con parámetros avanzados (áreas, columnas, layout_kwargs, y combinación de métodos `hybrid`).
- Ejecutar: `streamlit run app_complete.py`
- Notas: algunos modos pueden requerir dependencias del sistema (p. ej., Ghostscript) o el backend `pdfium`.

### 3) `compare.py` — Comparador de sugerencias (Excel)
- Descripción: aplicación Streamlit para comparar columnas de un Excel (cliente vs. usuario vs. IA) y calcular similitudes.
- Métricas soportadas:
  - Ratio secuencial (difflib) — siempre disponible.
  - TF-IDF char 3–5 — requiere `scikit-learn` (si no está, avisa y omite).
  - Token Set Ratio — usa RapidFuzz si está disponible; incluye fallback aproximado.
  - Embeddings (coseno) — opcional con `sentence-transformers`.
- Ejecutar: `streamlit run compare.py`
- Uso: sube un `.xlsx`/`.xls`, elige hoja y columnas, y visualiza las puntuaciones por fila.

### 4) `extract_atachments.py` — Extraer adjuntos de .msg
- Descripción: recorre carpetas numeradas `01`–`10` dentro de `input_dir` y extrae adjuntos de correos Outlook `.msg`.
- Configuración: edita la variable `input_dir` dentro del script para apuntar a tu ruta base.
- Ejecutar: `python extract_atachments.py`
- Salida: crea la carpeta `adjuntos_extraidos` dentro de `input_dir` con todos los adjuntos.

### 5) `seen_imap.py` y `unseen_imap.py` — Marcar correos (IMAP)
- `seen_imap.py`: marca todos los correos de `INBOX` como vistos.
- `unseen_imap.py`: marca un rango de UIDs como NO vistos.
- Seguridad: no guardes credenciales reales en el código. Usa variables de entorno o un gestor de secretos.
  - Ejemplo (recomendado):
    - Exportar antes de ejecutar: `export IMAP_SERVER=imap.servidor.net IMAP_EMAIL=usuario@dominio IMAP_PASSWORD='tu-pass'`
    - Y adaptar el script para leer `os.environ[...]` en lugar de literales.

---

## Instalación rápida
- Crear entorno virtual (opcional): `python -m venv .venv && source .venv/bin/activate`
- Instalar dependencias: `pip install -r requirements.txt`

## Notas
- No se debe versionar entornos locales. La carpeta `table_det/` está ignorada vía `.gitignore`.
- Si falta alguna dependencia opcional, las apps muestran avisos y siguen funcionando con degradación elegante.

