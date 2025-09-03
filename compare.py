# app.py
import io
import re
import pandas as pd
from difflib import SequenceMatcher
import streamlit as st
from scorers import SCORER_REGISTRY
try:
    import scorers.embed_cosine  # noqa: F401
    _EMBED_AVAILABLE = True
except Exception:
    _EMBED_AVAILABLE = False

st.set_page_config(page_title="Comparador de sugerencias", layout="wide")
st.title("📒 Comparador de sugerencias")

st.write("Sube un archivo .xlsx o .xls para visualizarlo.")

with st.sidebar:
    try:
        st.write("Scorers cargados:", list(SCORER_REGISTRY.keys()))
    except Exception:
        pass
    try:
        _tok_usr_nonnull = int(df_out["score_cliente_vs_usuario_tok"].notna().sum()) if "score_cliente_vs_usuario_tok" in df_out.columns else 0
        _tok_ia_nonnull = int(df_out["score_cliente_vs_IA_tok"].notna().sum()) if "score_cliente_vs_IA_tok" in df_out.columns else 0
        st.write("Token Set filas calculadas:", {"PROPUESTA USUARIO": _tok_usr_nonnull, "PROPUESTA IA": _tok_ia_nonnull})
    except Exception:
        pass

archivo = st.file_uploader("Elegir archivo Excel", type=["xlsx", "xls"])

@st.cache_data(show_spinner=False)
def obtener_hojas(bytes_data: bytes):
    # Devuelve solo los nombres de hojas (tipos serializables)
    xls = pd.ExcelFile(io.BytesIO(bytes_data))
    return xls.sheet_names

@st.cache_data(show_spinner=False)
def leer_hoja(bytes_data: bytes, hoja: str, header_row: int) -> pd.DataFrame:
    # Lee y devuelve un DataFrame (serializable por pickle)
    return pd.read_excel(
        io.BytesIO(bytes_data), sheet_name=hoja, header=header_row, engine="openpyxl"
    )

# Alias con el nombre usado en algunos entornos/llamadas previas
# Importante: devolver SOLO tipos serializables por pickle (p.ej., DataFrame)
@st.cache_data(show_spinner=False)
def leer_excel(bytes_data: bytes, hoja: str, header_row: int) -> pd.DataFrame:
    return pd.read_excel(
        io.BytesIO(bytes_data), sheet_name=hoja, header=header_row, engine="openpyxl"
    )

if archivo is not None:
    try:
        bytes_data = archivo.getvalue()
        hojas = obtener_hojas(bytes_data)
        hoja = st.selectbox("Selecciona la hoja", hojas, index=0)

        # Opciones de visualización
        col1, col2, col3 = st.columns([1,1,2])
        with col1:
            max_filas = st.number_input("Filas a mostrar", min_value=5, max_value=10000, value=100, step=5)
        with col2:
            header_row = st.number_input("Fila de encabezado (0-index)", min_value=0, value=0, step=1)
        with col3:
            dtype_preview = st.checkbox("Mostrar tipos de dato", value=False)

        # Leemos la hoja seleccionada con el header que elija el usuario (en caché)
        # Usamos leer_excel (alias) para evitar cualquier conflicto con caché previa
        df = leer_excel(bytes_data, hoja, int(header_row))

        st.caption(f"Hoja **{hoja}** — {df.shape[0]:,} filas × {df.shape[1]:,} columnas")

        if dtype_preview:
            st.write("**Tipos de columna:**")
            st.dataframe(pd.DataFrame(df.dtypes, columns=["dtype"]))

        st.dataframe(df.head(int(max_filas)), use_container_width=True)

        st.subheader("Comparar columnas por similitud (0–1)")
        cols = list(df.columns.astype(str))
        c1, c2, c3 = st.columns([1.2, 1.2, 1.6])
        with c1:
            col_cliente = st.selectbox("Solicitud del cliente (1)", options=cols, index=0 if cols else None, key="sel_cliente")
        with c2:
            col_usuario = st.selectbox("PROPUESTA USUARIO (1)", options=cols, index=min(1, len(cols)-1) if len(cols) > 1 else 0, key="sel_usuario")
        with c3:
            cols_ia = st.multiselect("PROPUESTA IA (1+)", options=cols, default=[cols[0]] if cols else [], key="sel_ia")
        # Todos los métodos se calculan y muestran de forma independiente; sin selector

        def _norm(x):
            if pd.isna(x):
                return ""
            s = str(x).strip().lower()
            return s

        def _sim(a, b) -> float:
            a2, b2 = _norm(a), _norm(b)
            if not a2 and not b2:
                return 1.0
            if not a2 or not b2:
                return 0.0
            return float(SequenceMatcher(None, a2, b2).ratio())

        def _sim_tfidf(a, b):
            a2, b2 = _norm(a), _norm(b)
            if not a2 and not b2:
                return 1.0
            if not a2 or not b2:
                return 0.0
            tfidf_fn = SCORER_REGISTRY.get("tfidf_char_cosine")
            if tfidf_fn is None:
                if not st.session_state.get("_tfidf_warned", False):
                    st.warning("No se pudo usar TF-IDF (char 3–5). Instale scikit-learn para habilitarlo.")
                    st.session_state["_tfidf_warned"] = True
                return None
            try:
                return float(tfidf_fn(a2, b2))
            except Exception as e:
                if not st.session_state.get("_tfidf_warned", False):
                    st.warning(f"No se pudo usar TF-IDF (char 3–5). Detalle: {e}")
                    st.session_state["_tfidf_warned"] = True
                return None

        def _tokenize_ws(s: str):
            # Palabras alfanuméricas (incluye acentos unicode)
            return re.findall(r"\w+", s, flags=re.UNICODE)

        def _fallback_token_set_ratio(a: str, b: str) -> float:
            ta = set(_tokenize_ws(a))
            tb = set(_tokenize_ws(b))
            if not ta and not tb:
                return 1.0
            if not ta or not tb:
                return 0.0
            inter = len(ta & tb)
            denom = len(ta) + len(tb)
            if denom == 0:
                return 0.0
            return (2.0 * inter) / float(denom)

        def _sim_token_set(a, b):
            a2, b2 = _norm(a), _norm(b)
            if not a2 and not b2:
                return 1.0
            if not a2 or not b2:
                return 0.0
            tok_fn = SCORER_REGISTRY.get("token_set_ratio")
            if tok_fn is None:
                # Fallback si RapidFuzz no está disponible
                if not st.session_state.get("_token_warned", False):
                    st.warning("RapidFuzz no disponible; usando fallback de conjunto de tokens (aprox.).")
                    st.session_state["_token_warned"] = True
                st.session_state["_token_fallback_used"] = True
                return float(_fallback_token_set_ratio(a2, b2))
            try:
                return float(tok_fn(a2, b2))
            except Exception as e:
                # Fallback ante error de import o ejecución
                if not st.session_state.get("_token_warned", False):
                    st.warning(f"No se pudo usar RapidFuzz token_set_ratio. Usando fallback (aprox). Detalle: {e}")
                    st.session_state["_token_warned"] = True
                st.session_state["_token_fallback_used"] = True
                return float(_fallback_token_set_ratio(a2, b2))

        def _sim_embed(a, b):
            a2, b2 = _norm(a), _norm(b)
            if not a2 and not b2:
                return 1.0
            if not a2 or not b2:
                return 0.0
            fn = SCORER_REGISTRY.get("embed_cosine")
            if fn is None:
                if not st.session_state.get("_embed_warned", False):
                    st.warning("No se pudo usar Sentence-Transformers (embed_cosine). Instale sentence-transformers para habilitarlo.")
                    st.session_state["_embed_warned"] = True
                return None
            try:
                return float(fn(a2, b2))
            except Exception as e:
                if not st.session_state.get("_embed_warned", False):
                    st.warning(f"No se pudo usar Sentence-Transformers (embed_cosine). Detalle: {e}")
                    st.session_state["_embed_warned"] = True
                return None

        if cols:
            df_out = df.copy()

            if col_cliente and col_usuario:
                df_out["score_cliente_vs_usuario"] = [
                    _sim(a, b) for a, b in zip(df_out[col_cliente], df_out[col_usuario])
                ]
                # TF-IDF paralelo (crear columna siempre; puede contener NA)
                tfidf_scores_usr = [_sim_tfidf(a, b) for a, b in zip(df_out[col_cliente], df_out[col_usuario])]
                df_out["score_cliente_vs_usuario_tfidf"] = tfidf_scores_usr
                # Token Set paralelo (columna estándar "_tok") — siempre crear columna
                tok_scores_usr = [_sim_token_set(a, b) for a, b in zip(df_out[col_cliente], df_out[col_usuario])]
                df_out["score_cliente_vs_usuario_tok"] = tok_scores_usr
                # Embed cosine paralelo (crear columna siempre; puede contener NA)
                try:
                    embed_scores_usr = [_sim_embed(a, b) for a, b in zip(df_out[col_cliente], df_out[col_usuario])]
                except Exception:
                    embed_scores_usr = [None] * len(df_out)
                df_out["score_cliente_vs_usuario_embed"] = embed_scores_usr
                # (se eliminan variantes duplicadas *_token_set; usar sufijo _tok)

            if cols_ia:
                best_scores_cli = []
                best_cols_cli = []
                best_scores_usr = []
                best_cols_usr = []
                # TF-IDF parallel
                best_scores_cli_tfidf = []
                best_scores_usr_tfidf = []
                # Token-set parallel (columna estándar "_tok")
                best_scores_cli_tok = []
                best_scores_usr_tok = []
                # Embed parallel
                best_scores_cli_embed = []
                best_scores_usr_embed = []

                for _, row in df_out.iterrows():
                    val_cli = row[col_cliente] if col_cliente else None
                    val_usr = row[col_usuario] if col_usuario else None
                    scores_cli = [(_sim(val_cli, row[c]), c) for c in cols_ia]
                    scores_usr = [(_sim(val_usr, row[c]), c) for c in cols_ia]
                    # TF-IDF scores (may be None)
                    scores_cli_tfidf = [(_sim_tfidf(val_cli, row[c]), c) for c in cols_ia]
                    scores_usr_tfidf = [(_sim_tfidf(val_usr, row[c]), c) for c in cols_ia]
                    # Token-set scores (may be None) – estándar "_tok"
                    scores_cli_tok = [(_sim_token_set(val_cli, row[c]), c) for c in cols_ia]
                    scores_usr_tok = [(_sim_token_set(val_usr, row[c]), c) for c in cols_ia]
                    # Embed scores (may be None)
                    try:
                        scores_cli_embed = [(_sim_embed(val_cli, row[c]), c) for c in cols_ia]
                        scores_usr_embed = [(_sim_embed(val_usr, row[c]), c) for c in cols_ia]
                    except Exception:
                        scores_cli_embed = []
                        scores_usr_embed = []
                    sc_cli, bc_cli = max(scores_cli, key=lambda t: t[0]) if scores_cli else (None, None)
                    sc_usr, bc_usr = max(scores_usr, key=lambda t: t[0]) if scores_usr else (None, None)
                    best_scores_cli.append(sc_cli)
                    best_cols_cli.append(bc_cli)
                    best_scores_usr.append(sc_usr)
                    best_cols_usr.append(bc_usr)

                    # For TF-IDF, ignore None when taking max
                    sc_cli_tfidf = None
                    if scores_cli_tfidf:
                        vals = [t[0] for t in scores_cli_tfidf if t[0] is not None]
                        sc_cli_tfidf = max(vals) if vals else None
                    sc_usr_tfidf = None
                    if scores_usr_tfidf:
                        vals = [t[0] for t in scores_usr_tfidf if t[0] is not None]
                        sc_usr_tfidf = max(vals) if vals else None
                    best_scores_cli_tfidf.append(sc_cli_tfidf)
                    best_scores_usr_tfidf.append(sc_usr_tfidf)

                    # Token-set (estándar _tok), ignore None when taking max
                    sc_cli_tok = None
                    if scores_cli_tok:
                        vals = [t[0] for t in scores_cli_tok if t[0] is not None]
                        sc_cli_tok = max(vals) if vals else None
                    sc_usr_tok = None
                    if scores_usr_tok:
                        vals = [t[0] for t in scores_usr_tok if t[0] is not None]
                        sc_usr_tok = max(vals) if vals else None
                    best_scores_cli_tok.append(sc_cli_tok)
                    best_scores_usr_tok.append(sc_usr_tok)
                    # Embed, ignore None when taking max
                    sc_cli_embed = None
                    if scores_cli_embed:
                        vals = [t[0] for t in scores_cli_embed if t[0] is not None]
                        sc_cli_embed = max(vals) if vals else None
                    sc_usr_embed = None
                    if scores_usr_embed:
                        vals = [t[0] for t in scores_usr_embed if t[0] is not None]
                        sc_usr_embed = max(vals) if vals else None
                    best_scores_cli_embed.append(sc_cli_embed)
                    best_scores_usr_embed.append(sc_usr_embed)

                df_out["score_cliente_vs_IA"] = best_scores_cli
                df_out["score_usuario_vs_IA"] = best_scores_usr
                df_out["mejor_col_IA_para_usuario"] = best_cols_usr
                # TF-IDF columns (crear siempre; pueden contener NA)
                df_out["score_cliente_vs_IA_tfidf"] = best_scores_cli_tfidf
                df_out["score_usuario_vs_IA_tfidf"] = best_scores_usr_tfidf
                # Token-set columns (estándar _tok) — siempre crear columnas
                df_out["score_cliente_vs_IA_tok"] = best_scores_cli_tok
                df_out["score_usuario_vs_IA_tok"] = best_scores_usr_tok
                # Embed columns (crear siempre; pueden contener NA)
                df_out["score_cliente_vs_IA_embed"] = best_scores_cli_embed
                df_out["score_usuario_vs_IA_embed"] = best_scores_usr_embed
            else:
                best_cols_cli = []

            # Solo mostrar columnas seleccionadas + scores respecto al CLIENTE
            # Construcción de listas por método
            score_cols_difflib_display = [c for c in ["score_cliente_vs_usuario", "score_cliente_vs_IA"] if c in df_out.columns]
            score_cols_tfidf_display = [c for c in ["score_cliente_vs_usuario_tfidf", "score_cliente_vs_IA_tfidf"] if c in df_out.columns]
            score_cols_tok_display = [c for c in ["score_cliente_vs_usuario_tok", "score_cliente_vs_IA_tok"] if c in df_out.columns]
            score_cols_embed_display = [c for c in ["score_cliente_vs_usuario_embed", "score_cliente_vs_IA_embed"] if c in df_out.columns]

            # Selección automática: prioriza Token Set si existe; si no, Difflib.
            has_tok_nonnull = False
            if score_cols_tok_display:
                try:
                    has_tok_nonnull = any(pd.to_numeric(df_out[c], errors="coerce").notna().any() for c in score_cols_tok_display)
                except Exception:
                    has_tok_nonnull = False
            if has_tok_nonnull:
                score_cols_display = score_cols_tok_display
                primary_usr_col, primary_ia_col = "score_cliente_vs_usuario_tok", "score_cliente_vs_IA_tok"
            else:
                score_cols_display = score_cols_difflib_display
                primary_usr_col, primary_ia_col = "score_cliente_vs_usuario", "score_cliente_vs_IA"

            # Score final por fila (método primario para resaltado)
            if score_cols_display:
                df_out["score_fila_cliente"] = (
                    df_out[score_cols_display].apply(pd.to_numeric, errors="coerce").max(axis=1)
                )
            # Scores por fila independientes por método
            cols_diff = [c for c in ["score_cliente_vs_usuario", "score_cliente_vs_IA"] if c in df_out.columns]
            if cols_diff:
                df_out["score_fila_cliente_diff"] = (
                    df_out[cols_diff].apply(pd.to_numeric, errors="coerce").max(axis=1)
                )
            cols_tfidf = [c for c in ["score_cliente_vs_usuario_tfidf", "score_cliente_vs_IA_tfidf"] if c in df_out.columns]
            if cols_tfidf:
                df_out["score_fila_cliente_tfidf"] = (
                    df_out[cols_tfidf].apply(pd.to_numeric, errors="coerce").max(axis=1)
                )
            cols_tok = [c for c in ["score_cliente_vs_usuario_tok", "score_cliente_vs_IA_tok"] if c in df_out.columns]
            if cols_tok:
                df_out["score_fila_cliente_tok"] = (
                    df_out[cols_tok].apply(pd.to_numeric, errors="coerce").max(axis=1)
                )
            cols_embed = [c for c in ["score_cliente_vs_usuario_embed", "score_cliente_vs_IA_embed"] if c in df_out.columns]
            if cols_embed:
                df_out["score_fila_cliente_embed"] = (
                    df_out[cols_embed].apply(pd.to_numeric, errors="coerce").max(axis=1)
                )

            # Calcular ganador_columna y bandera de empate sin crear columna 'decision'
            eps = 1e-9
            # Usar columnas primarias según selección de método
            if primary_ia_col or primary_usr_col:
                s_ia = pd.to_numeric(df_out.get(primary_ia_col), errors="coerce") if primary_ia_col in df_out.columns else pd.Series([pd.NA]*len(df_out), index=df_out.index)
                s_usr = pd.to_numeric(df_out.get(primary_usr_col), errors="coerce") if primary_usr_col in df_out.columns else pd.Series([pd.NA]*len(df_out), index=df_out.index)
                # serie con mejor columna IA por fila (si existe), sin exponer 'mejor_col_IA_para_cliente'
                best_col_cli_series = pd.Series(best_cols_cli, index=df_out.index) if best_cols_cli else pd.Series([pd.NA]*len(df_out), index=df_out.index)

                ganador = []
                empate_flags = []
                for idx in df_out.index:
                    ia = s_ia.loc[idx]
                    us = s_usr.loc[idx]
                    if pd.notna(ia) and pd.notna(us):
                        if abs(float(ia) - float(us)) <= eps:
                            ganador.append(best_col_cli_series.loc[idx])  # empate: asignamos IA
                            empate_flags.append(True)
                        elif float(ia) > float(us):
                            ganador.append(best_col_cli_series.loc[idx])
                            empate_flags.append(False)
                        else:
                            ganador.append("PROPUESTA USUARIO")
                            empate_flags.append(False)
                    elif pd.notna(ia):
                        ganador.append(best_col_cli_series.loc[idx])
                        empate_flags.append(False)
                    elif pd.notna(us):
                        ganador.append("PROPUESTA USUARIO")
                        empate_flags.append(False)
                    else:
                        ganador.append(pd.NA)
                        empate_flags.append(False)

                df_out["ganador_columna"] = ganador
                df_out["empate"] = empate_flags

            # Ganadores por método (usuario / IA / empate) — para depuración por fila
            def _winner_series(s_usr: pd.Series, s_ia: pd.Series, eps_local: float = 1e-9) -> list:
                out = []
                for u, i in zip(s_usr, s_ia):
                    u_na = pd.isna(u)
                    i_na = pd.isna(i)
                    if not u_na and not i_na:
                        du = float(u)
                        di = float(i)
                        if abs(di - du) <= eps_local:
                            out.append("IGUALES")
                        elif di > du:
                            out.append("PROPUESTA IA")
                        else:
                            out.append("PROPUESTA USUARIO")
                    elif not i_na:
                        out.append("PROPUESTA IA")
                    elif not u_na:
                        out.append("PROPUESTA USUARIO")
                    else:
                        out.append(pd.NA)
                return out

            # Difflib
            if ("score_cliente_vs_usuario" in df_out.columns) or ("score_cliente_vs_IA" in df_out.columns):
                s_usr_d = pd.to_numeric(df_out.get("score_cliente_vs_usuario"), errors="coerce") if "score_cliente_vs_usuario" in df_out.columns else pd.Series([pd.NA]*len(df_out), index=df_out.index)
                s_ia_d = pd.to_numeric(df_out.get("score_cliente_vs_IA"), errors="coerce") if "score_cliente_vs_IA" in df_out.columns else pd.Series([pd.NA]*len(df_out), index=df_out.index)
                df_out["ganador_diff"] = _winner_series(s_usr_d, s_ia_d, eps)

            # TF-IDF
            if ("score_cliente_vs_usuario_tfidf" in df_out.columns) or ("score_cliente_vs_IA_tfidf" in df_out.columns):
                s_usr_t = pd.to_numeric(df_out.get("score_cliente_vs_usuario_tfidf"), errors="coerce") if "score_cliente_vs_usuario_tfidf" in df_out.columns else pd.Series([pd.NA]*len(df_out), index=df_out.index)
                s_ia_t = pd.to_numeric(df_out.get("score_cliente_vs_IA_tfidf"), errors="coerce") if "score_cliente_vs_IA_tfidf" in df_out.columns else pd.Series([pd.NA]*len(df_out), index=df_out.index)
                df_out["ganador_tfidf"] = _winner_series(s_usr_t, s_ia_t, eps)

            # Token Set
            if ("score_cliente_vs_usuario_tok" in df_out.columns) or ("score_cliente_vs_IA_tok" in df_out.columns):
                s_usr_k = pd.to_numeric(df_out.get("score_cliente_vs_usuario_tok"), errors="coerce") if "score_cliente_vs_usuario_tok" in df_out.columns else pd.Series([pd.NA]*len(df_out), index=df_out.index)
                s_ia_k = pd.to_numeric(df_out.get("score_cliente_vs_IA_tok"), errors="coerce") if "score_cliente_vs_IA_tok" in df_out.columns else pd.Series([pd.NA]*len(df_out), index=df_out.index)
                df_out["ganador_tok"] = _winner_series(s_usr_k, s_ia_k, eps)

            # Embed Cosine
            if ("score_cliente_vs_usuario_embed" in df_out.columns) or ("score_cliente_vs_IA_embed" in df_out.columns):
                s_usr_e = pd.to_numeric(df_out.get("score_cliente_vs_usuario_embed"), errors="coerce") if "score_cliente_vs_usuario_embed" in df_out.columns else pd.Series([pd.NA]*len(df_out), index=df_out.index)
                s_ia_e = pd.to_numeric(df_out.get("score_cliente_vs_IA_embed"), errors="coerce") if "score_cliente_vs_IA_embed" in df_out.columns else pd.Series([pd.NA]*len(df_out), index=df_out.index)
                df_out["ganador_embed"] = _winner_series(s_usr_e, s_ia_e, eps)

            # Armar el set de columnas a mostrar: seleccionadas + mejores IA + scores
            view_cols = []
            for csel in [col_cliente, col_usuario]:
                if csel and csel not in view_cols:
                    view_cols.append(csel)
            for cia in cols_ia:
                if cia not in view_cols:
                    view_cols.append(cia)
            # Metadatos útiles (se removió 'mejor_col_IA_para_cliente')
            # Scores
            view_cols.extend(score_cols_display)
            # Añadir ganador del método primario (si procede)
            if score_cols_display == score_cols_tok_display and "ganador_tok" in df_out.columns:
                view_cols.append("ganador_tok")
            elif score_cols_display == score_cols_difflib_display and "ganador_diff" in df_out.columns:
                view_cols.append("ganador_diff")
            # Ocultar agregados por fila en la vista (se mantienen para estadísticas)
            # Add TF-IDF scores to the view if present
            view_cols.extend(score_cols_tfidf_display)
            if score_cols_tfidf_display and "ganador_tfidf" in df_out.columns:
                view_cols.append("ganador_tfidf")
            # Add Token-set scores to the view if present
            view_cols.extend(score_cols_tok_display)
            if score_cols_tok_display and "ganador_tok" in df_out.columns:
                view_cols.append("ganador_tok")
            # Add Embed scores to the view if present
            view_cols.extend(score_cols_embed_display)
            if score_cols_embed_display and "ganador_embed" in df_out.columns:
                view_cols.append("ganador_embed")
            # Add Difflib scores if not already included
            for c in score_cols_difflib_display:
                if c not in view_cols:
                    view_cols.append(c)
            if score_cols_difflib_display and "ganador_diff" in df_out.columns:
                view_cols.append("ganador_diff")
            # No se muestra 'decision'
            if "ganador_columna" in df_out.columns:
                view_cols.append("ganador_columna")

            # Quitar columnas duplicadas manteniendo el orden
            if view_cols:
                _seen = set()
                _unique_cols = []
                for c in view_cols:
                    if c not in _seen:
                        _unique_cols.append(c)
                        _seen.add(c)
                view_cols = _unique_cols
            df_view = df_out[view_cols] if view_cols else df_out

            # Resaltado por fila: en scores, mayor en verde; IGUALES en amarillo
            def _highlight_row_best(row: pd.Series):
                styles = pd.Series("", index=row.index)
                eps = 1e-9
                pairs = [
                    ("score_cliente_vs_usuario_tok", "score_cliente_vs_IA_tok"),
                    ("score_cliente_vs_usuario_tfidf", "score_cliente_vs_IA_tfidf"),
                    ("score_cliente_vs_usuario_embed", "score_cliente_vs_IA_embed"),
                    ("score_cliente_vs_usuario", "score_cliente_vs_IA"),
                ]
                for ucol, icol in pairs:
                    if ucol in row.index or icol in row.index:
                        u = pd.to_numeric(row.get(ucol), errors="coerce") if ucol in row.index else pd.NA
                        i = pd.to_numeric(row.get(icol), errors="coerce") if icol in row.index else pd.NA
                        if pd.notna(u) and pd.notna(i):
                            if abs(float(i) - float(u)) <= eps:
                                if icol in row.index:
                                    styles[icol] = "background-color: #fff3cd"  # amarillo IGUALES
                                if ucol in row.index:
                                    styles[ucol] = "background-color: #fff3cd"
                            elif float(i) > float(u):
                                if icol in row.index:
                                    styles[icol] = "background-color: #d4edda"  # verde mayor
                            else:
                                if ucol in row.index:
                                    styles[ucol] = "background-color: #d4edda"  # verde mayor
                        elif pd.notna(i):
                            if icol in row.index:
                                styles[icol] = "background-color: #d4edda"  # verde único
                        elif pd.notna(u):
                            if ucol in row.index:
                                styles[ucol] = "background-color: #d4edda"  # verde único
                return styles

            # Resaltado para columnas de "ganador_*" y "ganador_columna"
            def _highlight_winner_cells(row: pd.Series):
                styles = pd.Series("", index=row.index)
                winners = [
                    "ganador_tok",
                    "ganador_tfidf",
                    "ganador_embed",
                    "ganador_diff",
                    "ganador_columna",
                ]
                for wcol in winners:
                    if wcol in row.index:
                        val = row.get(wcol)
                        if pd.isna(val):
                            continue
                        try:
                            sval = str(val)
                        except Exception:
                            sval = ""
                        if sval == "PROPUESTA USUARIO":
                            styles[wcol] = "background-color: #cfe2ff"  # azul claro
                        elif sval == "PROPUESTA IA":
                            styles[wcol] = "background-color: #ffd8a8"  # naranja claro
                        elif sval == "IGUALES":
                            styles[wcol] = "background-color: #fff3cd"  # amarillo claro
                return styles

            st.caption("Vista con resaltado por fila (mejor score en verde, por método)")
            try:
                st.dataframe(
                    df_view.head(int(max_filas)).
                        style.apply(_highlight_row_best, axis=1).
                        apply(_highlight_winner_cells, axis=1),
                    use_container_width=True,
                )
            except Exception:
                st.dataframe(df_view.head(int(max_filas)), use_container_width=True)

            # Estadísticas de conteo de "buenas" (verde) por columna (sin 'decision')
            if ("score_cliente_vs_usuario" in df_out.columns) or ("score_cliente_vs_IA" in df_out.columns):
                st.subheader("Estadísticas — Mejores y IGUALES")
                s_ia = pd.to_numeric(df_out.get("score_cliente_vs_IA"), errors="coerce") if "score_cliente_vs_IA" in df_out.columns else pd.Series([pd.NA]*len(df_out), index=df_out.index)
                s_usr = pd.to_numeric(df_out.get("score_cliente_vs_usuario"), errors="coerce") if "score_cliente_vs_usuario" in df_out.columns else pd.Series([pd.NA]*len(df_out), index=df_out.index)
                ia_wins = (s_ia > s_usr)
                usr_wins = (s_usr > s_ia)
                empate_mask = s_ia.notna() & s_usr.notna() & ((s_ia - s_usr).abs() <= eps)

                total_usuario = int(usr_wins.sum())
                total_ia = int(ia_wins.sum())
                total_empates = int(empate_mask.sum())

                cA, cB, cC = st.columns(3)
                with cA:
                    st.metric("Mejores PROPUESTA USUARIO", f"{total_usuario}")
                with cB:
                    st.metric("Mejores PROPUESTA IA", f"{total_ia}")
                with cC:
                    st.metric("IGUALES", f"{total_empates}")
                # Mostrar método (difflib)
                st.caption("Método de similitud usado: Difflib (SequenceMatcher)")

            # TF-IDF stats block if both columns exist
            if ("score_cliente_vs_usuario_tfidf" in df_out.columns) or ("score_cliente_vs_IA_tfidf" in df_out.columns):
                st.subheader("Estadísticas — Mejores y IGUALES (TF-IDF)")
                s_ia_t = pd.to_numeric(df_out.get("score_cliente_vs_IA_tfidf"), errors="coerce") if "score_cliente_vs_IA_tfidf" in df_out.columns else pd.Series([pd.NA]*len(df_out), index=df_out.index)
                s_usr_t = pd.to_numeric(df_out.get("score_cliente_vs_usuario_tfidf"), errors="coerce") if "score_cliente_vs_usuario_tfidf" in df_out.columns else pd.Series([pd.NA]*len(df_out), index=df_out.index)
                ia_wins_t = (s_ia_t > s_usr_t)
                usr_wins_t = (s_usr_t > s_ia_t)
                empate_mask_t = s_ia_t.notna() & s_usr_t.notna() & ((s_ia_t - s_usr_t).abs() <= eps)

                total_usuario_t = int(usr_wins_t.sum())
                total_ia_t = int(ia_wins_t.sum())
                total_empates_t = int(empate_mask_t.sum())

                cA, cB, cC = st.columns(3)
                with cA:
                    st.metric("Mejores PROPUESTA USUARIO (TF-IDF)", f"{total_usuario_t}")
                with cB:
                    st.metric("Mejores PROPUESTA IA (TF-IDF)", f"{total_ia_t}")
                with cC:
                    st.metric("IGUALES (TF-IDF)", f"{total_empates_t}")
                st.caption("Método de similitud usado: TF-IDF (char 3–5) + coseno")

            # Token-set stats block (estándar con sufijo _tok)
            if ("score_cliente_vs_usuario_tok" in df_out.columns) or ("score_cliente_vs_IA_tok" in df_out.columns):
                st.subheader("Estadísticas — Mejores y IGUALES (Token Set)")
                s_ia_tk = pd.to_numeric(df_out.get("score_cliente_vs_IA_tok"), errors="coerce") if "score_cliente_vs_IA_tok" in df_out.columns else pd.Series([pd.NA]*len(df_out), index=df_out.index)
                s_usr_tk = pd.to_numeric(df_out.get("score_cliente_vs_usuario_tok"), errors="coerce") if "score_cliente_vs_usuario_tok" in df_out.columns else pd.Series([pd.NA]*len(df_out), index=df_out.index)
                valid_tk = s_ia_tk.notna() & s_usr_tk.notna()
                ia_wins_tk = valid_tk & (s_ia_tk > s_usr_tk)
                usr_wins_tk = valid_tk & (s_usr_tk > s_ia_tk)
                empate_mask_tk = valid_tk & ((s_ia_tk - s_usr_tk).abs() <= eps)

                cA, cB, cC = st.columns(3)
                with cA:
                    st.metric("Mejores PROPUESTA USUARIO (Token Set)", f"{int(usr_wins_tk.sum())}")
                with cB:
                    st.metric("Mejores PROPUESTA IA (Token Set)", f"{int(ia_wins_tk.sum())}")
                with cC:
                    st.metric("IGUALES (Token Set)", f"{int(empate_mask_tk.sum())}")
                if st.session_state.get("_token_fallback_used", False):
                    st.caption("Método de similitud usado: Token Set (Fallback aproximado; instale rapidfuzz para el oficial)")
                else:
                    st.caption("Método de similitud usado: RapidFuzz token_set_ratio")

            # Embed cosine stats block (Sentence-Transformers)
            if ("score_cliente_vs_usuario_embed" in df_out.columns) or ("score_cliente_vs_IA_embed" in df_out.columns):
                st.subheader("Estadísticas — Mejores y IGUALES (Embed Cosine)")
                s_ia_e = pd.to_numeric(df_out.get("score_cliente_vs_IA_embed"), errors="coerce") if "score_cliente_vs_IA_embed" in df_out.columns else pd.Series([pd.NA]*len(df_out), index=df_out.index)
                s_usr_e = pd.to_numeric(df_out.get("score_cliente_vs_usuario_embed"), errors="coerce") if "score_cliente_vs_usuario_embed" in df_out.columns else pd.Series([pd.NA]*len(df_out), index=df_out.index)
                valid_e = s_ia_e.notna() & s_usr_e.notna()
                ia_wins_e = valid_e & (s_ia_e > s_usr_e)
                usr_wins_e = valid_e & (s_usr_e > s_ia_e)
                empate_e = valid_e & ((s_ia_e - s_usr_e).abs() <= eps)

                cA, cB, cC = st.columns(3)
                with cA:
                    st.metric("Mejores PROPUESTA USUARIO (Embed)", f"{int(usr_wins_e.sum())}")
                with cB:
                    st.metric("Mejores PROPUESTA IA (Embed)", f"{int(ia_wins_e.sum())}")
                with cC:
                    st.metric("IGUALES (Embed)", f"{int(empate_e.sum())}")
                st.caption("Método de similitud usado: Sentence-Transformers embed_cosine")

            # (Se removió la sección "Métricas" y el control de Umbral a solicitud del usuario)

            # Descarga deshabilitada por solicitud del usuario

        # Descarga deshabilitada por solicitud del usuario
            # Resumen general (todas las métricas)
            try:
                import pandas as _pd  # local alias to avoid shadowing
                resumen_rows = []
                def _compute_counts(col_usr: str, col_ia: str):
                    eps_local = 1e-9
                    s_ia_loc = _pd.to_numeric(df_out.get(col_ia), errors="coerce") if col_ia in df_out.columns else _pd.Series([_pd.NA]*len(df_out), index=df_out.index)
                    s_usr_loc = _pd.to_numeric(df_out.get(col_usr), errors="coerce") if col_usr in df_out.columns else _pd.Series([_pd.NA]*len(df_out), index=df_out.index)
                    valid = s_ia_loc.notna() & s_usr_loc.notna()
                    ia_w = valid & (s_ia_loc > s_usr_loc)
                    usr_w = valid & (s_usr_loc > s_ia_loc)
                    emp = valid & ((s_ia_loc - s_usr_loc).abs() <= eps_local)
                    return int(_pd.to_numeric(usr_w, errors="coerce").fillna(0).sum()), int(_pd.to_numeric(ia_w, errors="coerce").fillna(0).sum()), int(_pd.to_numeric(emp, errors="coerce").fillna(0).sum())

                # Difflib (base)
                if ("score_cliente_vs_usuario" in df_out.columns) or ("score_cliente_vs_IA" in df_out.columns):
                    u,i,e = _compute_counts("score_cliente_vs_usuario", "score_cliente_vs_IA")
                    resumen_rows.append({"Método": "Difflib", "Mejores PROPUESTA USUARIO": u, "Mejores PROPUESTA IA": i, "IGUALES": e})

                # TF-IDF
                if ("score_cliente_vs_usuario_tfidf" in df_out.columns) or ("score_cliente_vs_IA_tfidf" in df_out.columns):
                    u,i,e = _compute_counts("score_cliente_vs_usuario_tfidf", "score_cliente_vs_IA_tfidf")
                    resumen_rows.append({"Método": "TF-IDF (char 3–5)", "Mejores PROPUESTA USUARIO": u, "Mejores PROPUESTA IA": i, "IGUALES": e})

                # Token Set (RapidFuzz)
                if ("score_cliente_vs_usuario_tok" in df_out.columns) or ("score_cliente_vs_IA_tok" in df_out.columns):
                    u,i,e = _compute_counts("score_cliente_vs_usuario_tok", "score_cliente_vs_IA_tok")
                    resumen_rows.append({"Método": "Token Set (RapidFuzz)", "Mejores PROPUESTA USUARIO": u, "Mejores PROPUESTA IA": i, "IGUALES": e})

                # Embed Cosine (Sentence-Transformers)
                if ("score_cliente_vs_usuario_embed" in df_out.columns) or ("score_cliente_vs_IA_embed" in df_out.columns):
                    u,i,e = _compute_counts("score_cliente_vs_usuario_embed", "score_cliente_vs_IA_embed")
                    resumen_rows.append({"Método": "Embed Cosine (ST)", "Mejores PROPUESTA USUARIO": u, "Mejores PROPUESTA IA": i, "IGUALES": e})

                if resumen_rows:
                    _file_label = getattr(archivo, "name", None)
                    if _file_label:
                        st.subheader(f"Resumen general de métodos — Archivo: {_file_label}")
                    else:
                        st.subheader("Resumen general de métodos")
                    df_resumen = _pd.DataFrame(resumen_rows)
                    try:
                        # Resaltar en verde el mayor entre 'Mejores PROPUESTA USUARIO' y 'Mejores PROPUESTA IA' por fila
                        if all(c in df_resumen.columns for c in ["Mejores PROPUESTA USUARIO", "Mejores PROPUESTA IA"]):
                            sty = df_resumen.style.highlight_max(subset=["Mejores PROPUESTA USUARIO", "Mejores PROPUESTA IA"], axis=1, color="#d4edda")
                            st.dataframe(sty, use_container_width=True)
                        else:
                            st.table(df_resumen)
                    except Exception:
                        st.table(df_resumen)
            except Exception:
                pass
    except Exception as e:
        st.error(f"Ocurrió un error leyendo el archivo: {e}")
else:
    st.info("👆 Arrastra o selecciona un archivo para comenzar.")
