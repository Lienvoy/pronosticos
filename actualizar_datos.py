"""
Pronósticos de fútbol: descarga resultados, ajusta un modelo Dixon-Coles por liga,
hace un backtest fuera de muestra y escribe datos.js para la página index.html.

Uso:
    pip install pandas numpy scipy requests lxml
    python actualizar_datos.py                 # fuente automática (football-data.co.uk y si falla openfootball)
    python actualizar_datos.py --fuente openfootball

Fuentes (todas gratuitas):
  - football-data.co.uk : resultados + tiros, córners, tarjetas y CUOTAS (Pinnacle/promedio). Ligas europeas y Liga MX.
                          fixtures.csv trae próximos partidos con cuotas. Se actualiza ~2 veces por semana.
  - openfootball (GitHub): resultados y calendario de ligas europeas (respaldo, sin cuotas ni estadísticas).
  - ClubElo (api.clubelo.com): rating Elo de todos los clubes europeos, se usa para Champions y Europa League,
                          donde se enfrentan equipos de ligas distintas.
"""
import argparse
import io
import json
import math
import sys
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import requests
from scipy.optimize import minimize
from scipy.stats import poisson

import valores

# ----------------------------------------------------------------------------- configuración
TEMPORADAS_FD = ["2425", "2526", "2627"]                # códigos de temporada en football-data.co.uk
TEMPORADAS_OF = ["2024-25", "2025-26", "2026-27"]       # la última es la actual
OF_ACTUAL = TEMPORADAS_OF[-1]

LIGAS = {
    # clave: (nombre, código football-data, código openfootball)
    "premier":    ("Premier League", "E0",  "en.1"),
    "laliga":     ("LaLiga",         "SP1", "es.1"),
    "seriea":     ("Serie A",        "I1",  "it.1"),
    "ligue1":     ("Ligue 1",        "F1",  "fr.1"),
    # Si "League 1" era la EFL League One inglesa, descomenta esta línea:
    # "leagueone": ("EFL League One", "E2",  "en.3"),
    "eredivisie": ("Eredivisie",     "N1",  "nl.1"),
    "ligamx":     ("Liga MX",        "MEX", None),
}
COPAS = {"ucl": "Champions League", "uel": "Europa League"}
PAIS_ELO = {"premier": "ENG", "laliga": "ESP", "seriea": "ITA", "ligue1": "FRA", "eredivisie": "NED"}

XI = 0.005            # decaimiento temporal por día (vida media ≈ 140 días)
RIDGE = 0.6           # regularización de fuerzas de ataque/defensa
PRIOR_ASCENDIDO = (-0.25, 0.25)   # equipos sin historia en la liga arrancan más débiles
MAX_GOLES = 10
HTTP = requests.Session()
HTTP.headers["User-Agent"] = "pronosticos-personal/1.0"


def get(url, timeout=30, intentos=3):
    """GET con reintentos: football-data.co.uk a veces corta la conexión."""
    import time
    for i in range(intentos):
        try:
            r = HTTP.get(url, timeout=timeout)
            r.raise_for_status()
            return r
        except requests.RequestException:
            if i == intentos - 1:
                raise
            time.sleep(2 * (i + 1))


# ----------------------------------------------------------------------------- fuentes
FD_COLS = {"HS": "tiros_l", "AS": "tiros_v", "HST": "arco_l", "AST": "arco_v",
           "HC": "corners_l", "AC": "corners_v", "HY": "amarillas_l", "AY": "amarillas_v",
           "HR": "rojas_l", "AR": "rojas_v"}


def _cuotas_fd(df):
    """Cuotas 1X2 y +/-2.5: preferimos cierre de Pinnacle, luego Pinnacle, luego promedio de mercado."""
    out = pd.DataFrame(index=df.index)
    for dest, opciones in {
        "c1": ["PSCH", "PSH", "AvgCH", "AvgH", "B365H"],
        "cx": ["PSCD", "PSD", "AvgCD", "AvgD", "B365D"],
        "c2": ["PSCA", "PSA", "AvgCA", "AvgA", "B365A"],
        "co25": ["PC>2.5", "P>2.5", "AvgC>2.5", "Avg>2.5"],
        "cu25": ["PC<2.5", "P<2.5", "AvgC<2.5", "Avg<2.5"],
    }.items():
        serie = pd.Series(np.nan, index=df.index)
        for c in opciones:
            if c in df.columns:
                serie = serie.fillna(pd.to_numeric(df[c], errors="coerce"))
        out[dest] = serie
    return out


def _normaliza_fd(df, liga_mx=False):
    if liga_mx:
        df = df.rename(columns={"Home": "HomeTeam", "Away": "AwayTeam", "HG": "FTHG", "AG": "FTAG"})
    df = df.dropna(subset=["HomeTeam", "AwayTeam"])
    out = pd.DataFrame({
        "fecha": pd.to_datetime(df["Date"], dayfirst=True, errors="coerce").dt.date,
        "local": df["HomeTeam"].str.strip(), "visita": df["AwayTeam"].str.strip(),
        "gl": pd.to_numeric(df.get("FTHG"), errors="coerce"),
        "gv": pd.to_numeric(df.get("FTAG"), errors="coerce"),
        "gl_mt": pd.to_numeric(df.get("HTHG"), errors="coerce") if "HTHG" in df else np.nan,
        "gv_mt": pd.to_numeric(df.get("HTAG"), errors="coerce") if "HTAG" in df else np.nan,
    })
    for c, n in FD_COLS.items():
        out[n] = pd.to_numeric(df[c], errors="coerce") if c in df else np.nan
    out = pd.concat([out, _cuotas_fd(df)], axis=1)
    return out.dropna(subset=["fecha"])


def carga_football_data(clave, codigo):
    if codigo == "MEX":
        df = pd.read_csv(io.StringIO(get("https://www.football-data.co.uk/new/MEX.csv").text))
        df = _normaliza_fd(df, liga_mx=True)
        df = df[df["fecha"] >= date.today() - timedelta(days=800)]
        jugados, futuros = df.dropna(subset=["gl"]), df[df["gl"].isna()]
    else:
        partes = []
        for t in TEMPORADAS_FD:
            txt = get(f"https://www.football-data.co.uk/mmz4281/{t}/{codigo}.csv").text
            partes.append(_normaliza_fd(pd.read_csv(io.StringIO(txt), encoding_errors="ignore")))
        df = pd.concat(partes, ignore_index=True)
        jugados, futuros = df.dropna(subset=["gl"]), df.iloc[0:0]
    return jugados.reset_index(drop=True), futuros.reset_index(drop=True)


_FIXTURES_FD = None


def proximos_football_data(codigo):
    global _FIXTURES_FD
    if _FIXTURES_FD is None:
        try:
            # el archivo trae BOM: se decodifica con utf-8-sig para que la primera columna sea "Div"
            txt = get("https://www.football-data.co.uk/fixtures.csv").content.decode("utf-8-sig", errors="ignore")
            _FIXTURES_FD = pd.read_csv(io.StringIO(txt))
        except Exception as e:  # noqa: BLE001
            print("  fixtures.csv no disponible:", e)
            _FIXTURES_FD = pd.DataFrame()
    if _FIXTURES_FD.empty or "Div" not in _FIXTURES_FD:
        return pd.DataFrame()
    return _normaliza_fd(_FIXTURES_FD[_FIXTURES_FD["Div"] == codigo])


def carga_openfootball(codigo):
    filas = []
    for t in TEMPORADAS_OF:
        d = get(f"https://raw.githubusercontent.com/openfootball/football.json/master/{t}/{codigo}.json").json()
        for m in d["matches"]:
            s = m.get("score")
            ft = s if isinstance(s, list) else (s or {}).get("ft")
            ht = (s or {}).get("ht") if isinstance(s, dict) else None
            filas.append({"fecha": date.fromisoformat(m["date"]), "local": m["team1"], "visita": m["team2"],
                          "gl": ft[0] if ft else np.nan, "gv": ft[1] if ft else np.nan,
                          "gl_mt": ht[0] if ht else np.nan, "gv_mt": ht[1] if ht else np.nan,
                          "temporada": t})
    df = pd.DataFrame(filas)
    for n in list(FD_COLS.values()) + ["c1", "cx", "c2", "co25", "cu25"]:
        df[n] = np.nan
    jugados = df.dropna(subset=["gl"])
    futuros = df[df["gl"].isna() & (df["temporada"] == OF_ACTUAL)]
    return jugados.drop(columns="temporada").reset_index(drop=True), futuros.drop(columns="temporada").reset_index(drop=True)


# ----------------------------------------------------------------------------- complemento ESPN
# football-data.co.uk se actualiza ~2 veces por semana; ESPN tiene el marcador minutos después del partido.
# Con ESPN completamos los partidos recientes que football-data aún no publica y el calendario próximo.
ESPN_SB = "https://site.api.espn.com/apis/site/v2/sports/soccer"
ESPN_LIGA = {"premier": "eng.1", "laliga": "esp.1", "seriea": "ita.1", "ligue1": "fra.1",
             "eredivisie": "ned.1", "ligamx": "mex.1"}
MAPAS, MODELOS = {}, {}   # nombre ESPN -> football-data y modelo ajustado de cada liga (para las copas)
EVENTOS_ESPN = {}         # partidos de ESPN por liga (con ids de equipo y de partido)
PENDIENTES_SIM = []       # partidos próximos: se simulan al final, cuando ya se conocen las alineaciones
ESPN_STATS = {"totalShots": "tiros", "shotsOnTarget": "arco", "wonCorners": "corners",
              "yellowCards": "amarillas", "redCards": "rojas"}


def _decimal(ml):
    """Momio americano (+380, -160) a cuota decimal."""
    try:
        ml = float(ml)
    except (TypeError, ValueError):
        return np.nan
    if ml == 0:
        return np.nan
    return round(1 + (ml / 100 if ml > 0 else 100 / -ml), 3)


def _cuotas_espn(o):
    """Acepta los dos formatos de ESPN: el plano del resumen del partido (homeTeamOdds.moneyLine) y el
    anidado del calendario (moneyline.home.close.odds). Preferimos la cuota de cierre."""
    o = o or {}

    def anidada(grupo, lado):
        g = ((o.get(grupo) or {}).get(lado) or {})
        return (g.get("close") or {}).get("odds") or (g.get("open") or {}).get("odds")

    out = {"c1": _decimal(anidada("moneyline", "home") or (o.get("homeTeamOdds") or {}).get("moneyLine")),
           "cx": _decimal(anidada("moneyline", "draw") or (o.get("drawOdds") or {}).get("moneyLine")),
           "c2": _decimal(anidada("moneyline", "away") or (o.get("awayTeamOdds") or {}).get("moneyLine")),
           "co25": np.nan, "cu25": np.nan}
    if o.get("overUnder") == 2.5:
        out["co25"] = _decimal(anidada("total", "over") or o.get("overOdds"))
        out["cu25"] = _decimal(anidada("total", "under") or o.get("underOdds"))
    return out


def _eventos_espn(codigo, desde, hasta):
    """Partidos de ESPN día por día (el filtro por rango de fechas de ESPN no es confiable)."""
    eventos, d = {}, desde
    while d <= hasta:
        try:
            js = get(f"{ESPN_SB}/{codigo}/scoreboard?dates={d:%Y%m%d}").json()
        except Exception:  # noqa: BLE001
            js = {}
        for ev in js.get("events", []):
            c = ev["competitions"][0]
            eq = {x.get("homeAway"): x for x in c.get("competitors", [])}
            if "home" not in eq or "away" not in eq:
                continue
            st = ev["status"]["type"]
            fin = bool(st.get("completed"))
            eventos[ev["id"]] = {"id": ev["id"], "fecha": d, "estado": st.get("state"),
                                 "local": eq["home"]["team"]["displayName"], "visita": eq["away"]["team"]["displayName"],
                                 "id_l": str(eq["home"]["team"]["id"]), "id_v": str(eq["away"]["team"]["id"]),
                                 "gl": int(eq["home"]["score"]) if fin else None,
                                 "gv": int(eq["away"]["score"]) if fin else None,
                                 "odds": (c.get("odds") or [None])[0]}
        d += timedelta(days=1)
    return list(eventos.values())


def _mapa_equipos(eventos, jug, actuales):
    """Nombre ESPN -> nombre football-data. Se vota con partidos que ambas fuentes tienen (misma fecha ±1 y
    mismo marcador); los que no alcanzan votos se emparejan por parecido de nombre."""
    import difflib
    from collections import Counter
    votos = Counter()
    for e in eventos:
        if e["gl"] is None:
            continue
        cerca = jug["fecha"].map(lambda f: abs((f - e["fecha"]).days) <= 1)
        for r in jug[cerca & (jug["gl"] == e["gl"]) & (jug["gv"] == e["gv"])].itertuples():
            votos[(e["local"], r.local)] += 1
            votos[(e["visita"], r.visita)] += 1
    mapa, usados = {}, set()
    for (a, b), n in votos.most_common():
        if n >= 2 and a not in mapa and b not in usados:
            mapa[a] = b
            usados.add(b)
    libres = sorted(set(actuales) - usados)
    for nombre in sorted({e["local"] for e in eventos} | {e["visita"] for e in eventos}):
        if nombre not in mapa and libres:
            m = difflib.get_close_matches(nombre, libres, n=1, cutoff=0.5)
            if m:
                mapa[nombre] = m[0]
                libres.remove(m[0])
    return mapa


def complemento_espn(clave, jug, fut):
    """Devuelve (partidos jugados que faltan en jug, próximos partidos que faltan en fut)."""
    codigo = ESPN_LIGA[clave]
    hoy, ultimo = date.today(), max(jug["fecha"])
    desde = ultimo - timedelta(days=21)          # tres semanas en común para emparejar nombres
    eventos = _eventos_espn(codigo, desde, hoy + timedelta(days=7))
    recientes = jug[jug["fecha"] >= desde - timedelta(days=2)]
    actuales = set(jug[jug["fecha"] >= ultimo - timedelta(days=90)]["local"])
    mapa = _mapa_equipos(eventos, recientes, actuales)

    def ya_esta(df, f, l):
        return bool(len(df)) and ((df["local"] == l) & df["fecha"].map(lambda x: abs((x - f).days) <= 1)).any()

    cols = ["fecha", "local", "visita", "gl", "gv", "gl_mt", "gv_mt", *FD_COLS.values(), "c1", "cx", "c2", "co25", "cu25"]
    jugados, proximos = [], []
    for e in eventos:
        l, v = mapa.get(e["local"]), mapa.get(e["visita"])
        if not l or not v:
            continue
        fila = dict.fromkeys(cols, np.nan)
        fila.update(fecha=e["fecha"], local=l, visita=v)
        if e["gl"] is not None and e["fecha"] >= ultimo - timedelta(days=1) and not ya_esta(jug, e["fecha"], l):
            fila.update(gl=e["gl"], gv=e["gv"])
            try:
                s = get(f"{ESPN_SB}/{codigo}/summary?event={e['id']}").json()
                for t in (s.get("boxscore") or {}).get("teams", []):
                    lado = "_l" if t.get("homeAway") == "home" else "_v"
                    for st in t.get("statistics", []):
                        if st.get("name") in ESPN_STATS:
                            fila[ESPN_STATS[st["name"]] + lado] = pd.to_numeric(st.get("displayValue"), errors="coerce")
                fila.update(_cuotas_espn((s.get("pickcenter") or [None])[0] or e["odds"]))
            except Exception:  # noqa: BLE001
                fila.update(_cuotas_espn(e["odds"]))
            jugados.append(fila)
        elif e["estado"] == "pre" and e["fecha"] >= hoy and not ya_esta(fut, e["fecha"], l):
            fila.update(_cuotas_espn(e["odds"]))
            proximos.append(fila)
    sin_mapa = sorted({n for e in eventos for n in (e["local"], e["visita"]) if n not in mapa})
    if sin_mapa:
        print("    ESPN sin equivalencia:", ", ".join(sin_mapa))
    MAPAS[clave] = mapa          # lo reutiliza el modelo de copas europeas
    EVENTOS_ESPN[clave] = eventos   # y las alineaciones (último partido de cada equipo)
    return pd.DataFrame(jugados, columns=cols), pd.DataFrame(proximos, columns=cols)


def carga_clubelo():
    hoy = date.today().isoformat()
    df = pd.read_csv(io.StringIO(get(f"http://api.clubelo.com/{hoy}").text))
    df = df[df["Level"] == 1].sort_values("Elo", ascending=False)
    return [{"club": r.Club, "pais": r.Country, "elo": round(float(r.Elo), 1)} for r in df.itertuples()]


# ----------------------------------------------------------------------------- modelo Dixon-Coles
def _tau(x, y, l, m, rho):
    t = np.ones_like(l)
    t = np.where((x == 0) & (y == 0), 1 - l * m * rho, t)
    t = np.where((x == 0) & (y == 1), 1 + l * rho, t)
    t = np.where((x == 1) & (y == 0), 1 + m * rho, t)
    t = np.where((x == 1) & (y == 1), 1 - rho, t)
    return np.maximum(t, 1e-10)


def ajusta_dixon_coles(jug, hasta, inicios, con_cov=False):
    """Ajusta el modelo con los partidos anteriores a `hasta`, ponderados por antigüedad.
    Los equipos sin partidos antes del inicio de la temporada en curso (ascendidos) usan un prior más débil.
    Con con_cov=True también devuelve el vector de parámetros y su covarianza aproximada (para Montecarlo)."""
    d = jug[jug["fecha"] < hasta]
    ini = max([s for s in inicios if s < hasta], default=d["fecha"].min())
    equipos_nuevos = set(d["local"]) - set(d[d["fecha"] < ini]["local"])
    equipos = sorted(set(d["local"]) | set(d["visita"]))
    idx = {e: i for i, e in enumerate(equipos)}
    n = len(equipos)
    hi = d["local"].map(idx).to_numpy()
    ai = d["visita"].map(idx).to_numpy()
    x = d["gl"].to_numpy(int)
    y = d["gv"].to_numpy(int)
    dias = np.array([(hasta - f).days for f in d["fecha"]])
    w = np.exp(-XI * dias)
    pa = np.array([PRIOR_ASCENDIDO[0] if e in equipos_nuevos else 0 for e in equipos])
    pd_ = np.array([PRIOR_ASCENDIDO[1] if e in equipos_nuevos else 0 for e in equipos])
    lgx = np.array([math.lgamma(v + 1) for v in x])
    lgy = np.array([math.lgamma(v + 1) for v in y])

    def nll(p):
        mu, h, rho = p[0], p[1], p[2]
        a, df_ = p[3:3 + n], p[3 + n:]
        l = np.exp(mu + h + a[hi] + df_[ai])
        m = np.exp(mu + a[ai] + df_[hi])
        ll = x * np.log(l) - l - lgx + y * np.log(m) - m - lgy + np.log(_tau(x, y, l, m, rho))
        return -(w * ll).sum() + RIDGE * (((a - pa) ** 2).sum() + ((df_ - pd_) ** 2).sum())

    p0 = np.concatenate([[0.2, 0.25, -0.05], pa, pd_])
    bounds = [(None, None), (-1, 1), (-0.3, 0.3)] + [(-3, 3)] * (2 * n)
    r = minimize(nll, p0, method="L-BFGS-B", bounds=bounds)
    p = r.x
    out = {"mu": float(p[0]), "home": float(p[1]), "rho": float(p[2]),
           "ataque": {e: float(p[3 + i]) for i, e in enumerate(equipos)},
           "defensa": {e: float(p[3 + n + i]) for i, e in enumerate(equipos)}}
    if con_cov:
        out.update(theta=p.copy(), cov=_covarianza(p, nll, hi, ai, w, n), orden=equipos)
        se = np.sqrt(np.diag(out["cov"]))
        out["se"] = {e: (float(se[3 + i]), float(se[3 + n + i])) for i, e in enumerate(equipos)}
    return out


def _covarianza(p, nll, hi, ai, w, n):
    """Inversa del Hessiano de la log-verosimilitud penalizada (aproximación de Laplace).
    Para la parte Poisson con enlace log, el Hessiano es exacto y cerrado: Σ w·λ·x·x' (x = fila de diseño) + ridge.
    El término de Dixon-Coles solo toca a rho, cuya segunda derivada se calcula numéricamente aparte."""
    P = 3 + 2 * n
    k = len(hi)
    a, df_ = p[3:3 + n], p[3 + n:]
    l = np.exp(p[0] + p[1] + a[hi] + df_[ai])
    m = np.exp(p[0] + a[ai] + df_[hi])
    Xl = np.zeros((k, P)); Xm = np.zeros((k, P))
    filas = np.arange(k)
    Xl[:, 0] = 1; Xl[:, 1] = 1; Xl[filas, 3 + hi] = 1; Xl[filas, 3 + n + ai] = 1
    Xm[:, 0] = 1; Xm[filas, 3 + ai] = 1; Xm[filas, 3 + n + hi] = 1
    H = (Xl * (w * l)[:, None]).T @ Xl + (Xm * (w * m)[:, None]).T @ Xm
    H[np.arange(3, P), np.arange(3, P)] += 2 * RIDGE
    eps = 1e-3
    pr = [p.copy() for _ in range(3)]
    pr[0][2] -= eps; pr[2][2] += eps
    h_rho = (nll(pr[0]) - 2 * nll(pr[1]) + nll(pr[2])) / eps ** 2
    H[2, 2] = max(h_rho, 1.0)
    try:
        np.linalg.cholesky(H)
    except np.linalg.LinAlgError:
        H += np.eye(P) * 1e-6 * np.trace(H) / P
    return np.linalg.inv(H)


def lambdas(mod, local, visita):
    a, d = mod["ataque"], mod["defensa"]
    pa, pd_ = PRIOR_ASCENDIDO
    al, dl = a.get(local, pa), d.get(local, pd_)
    av, dv = a.get(visita, pa), d.get(visita, pd_)
    return math.exp(mod["mu"] + mod["home"] + al + dv), math.exp(mod["mu"] + av + dl)


def matriz(l, m, rho):
    g = np.arange(MAX_GOLES + 1)
    M = np.outer(poisson.pmf(g, l), poisson.pmf(g, m))
    M[0, 0] *= 1 - l * m * rho
    M[0, 1] *= 1 + l * rho
    M[1, 0] *= 1 + m * rho
    M[1, 1] *= 1 - rho
    return M / M.sum()


def prob_1x2(mod, local, visita):
    l, m = lambdas(mod, local, visita)
    M = matriz(l, m, mod["rho"])
    return float(np.tril(M, -1).sum()), float(np.trace(M)), float(np.triu(M, 1).sum()), M


# ----------------------------------------------------------------------------- backtest
def backtest(jug, inicios):
    """Reajusta cada semana con datos previos y predice la semana siguiente (fuera de muestra).
    Cubre la temporada pasada completa y lo que va de la actual."""
    if len(inicios) < 2:
        return [], {}
    desde = inicios[-2]
    act = jug[jug["fecha"] >= desde].sort_values("fecha")
    corte = desde + timedelta(days=21)   # dejamos 3 semanas de arranque
    fin = act["fecha"].iloc[-1]
    preds = []
    while corte <= fin:
        sem = act[(act["fecha"] >= corte) & (act["fecha"] < corte + timedelta(days=7))]
        if len(sem):
            mod = ajusta_dixon_coles(jug, corte, inicios)
            for r in sem.itertuples():
                p1, px, p2, M = prob_1x2(mod, r.local, r.visita)
                pover = float(sum(M[i, j] for i in range(MAX_GOLES + 1) for j in range(MAX_GOLES + 1) if i + j > 2))
                preds.append({"fecha": r.fecha.isoformat(), "local": r.local, "visita": r.visita,
                              "gl": int(r.gl), "gv": int(r.gv), "p": [round(p1, 4), round(px, 4), round(p2, 4)],
                              "pover": round(pover, 4),
                              "c": [None if pd.isna(v) else float(v) for v in (r.c1, r.cx, r.c2)]})
        corte += timedelta(days=7)
    return preds, metricas(preds, jug[jug["fecha"] < desde])


def metricas(preds, hist):
    if not preds:
        return {}
    res = np.array([0 if p["gl"] > p["gv"] else 1 if p["gl"] == p["gv"] else 2 for p in preds])
    P = np.array([p["p"] for p in preds])
    Y = np.eye(3)[res]
    base = np.array([(hist["gl"] > hist["gv"]).mean(), (hist["gl"] == hist["gv"]).mean(),
                     (hist["gl"] < hist["gv"]).mean()]) if len(hist) else np.array([.45, .27, .28])
    out = {"n": len(preds),
           "logloss": float(-np.mean(np.log(P[np.arange(len(P)), res]))),
           "logloss_base": float(-np.mean(np.log(base[res]))),
           "brier": float(np.mean(((P - Y) ** 2).sum(1))),
           "brier_base": float(np.mean(((base - Y) ** 2).sum(1))),
           "acierto": float(np.mean(P.argmax(1) == res))}
    # calibración: probabilidad pronosticada vs frecuencia observada
    bins = np.linspace(0, 1, 6)
    cal = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = (P >= lo) & (P < hi)
        if sel.sum() >= 5:
            cal.append({"rango": f"{int(lo*100)}–{int(hi*100)}%", "n": int(sel.sum()),
                        "pred": float(P[sel].mean()), "obs": float(Y[sel].mean())})
    out["calibracion"] = cal
    # si hay cuotas: comparar contra el mercado y simular apuestas de valor a stake fijo
    C = np.array([[np.nan if v is None else v for v in p["c"]] for p in preds])
    ok = ~np.isnan(C).any(1)
    if ok.sum() >= 20:
        imp = 1 / C[ok]
        imp = imp / imp.sum(1, keepdims=True)
        out["logloss_mercado"] = float(-np.mean(np.log(imp[np.arange(ok.sum()), res[ok]])))
        ev = P[ok] * C[ok] - 1
        apuestas = ev > 0.05
        ganancia = np.where(Y[ok] == 1, C[ok] - 1, -1)
        out["apuestas_valor"] = int(apuestas.sum())
        out["roi_valor"] = float(ganancia[apuestas].mean()) if apuestas.any() else None
    return out


# ----------------------------------------------------------------------------- simulación Montecarlo
# Ver METODOLOGIA_SIMULACIONES.md. Semilla fija: con los mismos datos se obtienen los mismos números.
S_PARAM = 2000            # juegos de parámetros muestreados de la normal (θ, Σ)
SIMS_POR_JUEGO = 10       # partidos simulados por juego de parámetros -> 20,000 por partido
S_TEMP = 1000             # juegos de parámetros usados en la simulación de temporada
TEMP_POR_JUEGO = 10       # temporadas por juego de parámetros -> 10,000 temporadas
SEMILLA = 20261025
ZONAS = {   # clave: (etiqueta del 1º, plazas de arriba, etiqueta, plazas de descenso, ida y vuelta)
    "premier": ("Campeón", 4, "Top 4", 3, True), "laliga": ("Campeón", 4, "Top 4", 3, True),
    "seriea": ("Campeón", 4, "Top 4", 3, True), "ligue1": ("Campeón", 4, "Top 4", 2, True),
    "eredivisie": ("Campeón", 4, "Top 4", 2, True), "ligamx": ("Superlíder", 6, "Liguilla directa (top 6)", 0, False),
    # copas: fase de liga de 36 equipos; 1-8 a octavos, 9-24 a playoff, 25-36 eliminados
    "ucl": ("1º de la fase", 8, "Top 8 (octavos directos)", 12, True),
    "uel": ("1º de la fase", 8, "Top 8 (octavos directos)", 12, True),
}


def _rng(clave, extra=""):
    return np.random.default_rng([SEMILLA, *map(ord, clave + extra)])


def muestrea_parametros(mod, rng):
    """S_PARAM vectores θ_s ~ Normal(θ̂, Σ̂), respetando los límites del ajuste."""
    L = np.linalg.cholesky(mod["cov"])
    th = mod["theta"][None, :] + rng.standard_normal((S_PARAM, len(mod["theta"]))) @ L.T
    th[:, 1] = th[:, 1].clip(-1, 1)
    th[:, 2] = th[:, 2].clip(-0.3, 0.3)
    th[:, 3:] = th[:, 3:].clip(-3, 3)
    return th


def _lambdas_s(mod, th, local, visita):
    """λ local y λ visita para cada juego de parámetros (vectores de largo S)."""
    idx, n = {e: i for i, e in enumerate(mod["orden"])}, len(mod["orden"])
    pa, pd_ = PRIOR_ASCENDIDO
    al = th[:, 3 + idx[local]] if local in idx else pa
    dl = th[:, 3 + n + idx[local]] if local in idx else pd_
    av = th[:, 3 + idx[visita]] if visita in idx else pa
    dv = th[:, 3 + n + idx[visita]] if visita in idx else pd_
    return np.exp(th[:, 0] + th[:, 1] + al + dv), np.exp(th[:, 0] + av + dl), th[:, 2]


def _matrices_dc(l, m, rho):
    """Matriz de marcadores Dixon-Coles para cada juego de parámetros: forma (S, 11, 11)."""
    g = np.arange(MAX_GOLES + 1)
    fact = np.cumprod(np.r_[1, g[1:]]).astype(float)
    pl = np.exp(-l)[:, None] * l[:, None] ** g / fact          # Poisson directo: mucho más rápido que scipy aquí
    pm = np.exp(-m)[:, None] * m[:, None] ** g / fact
    M = pl[:, :, None] * pm[:, None, :]
    M[:, 0, 0] *= 1 - l * m * rho
    M[:, 0, 1] *= 1 + l * rho
    M[:, 1, 0] *= 1 + m * rho
    M[:, 1, 1] *= 1 - rho
    np.maximum(M, 0, out=M)
    M /= M.sum((1, 2), keepdims=True)
    return M


def _muestrea_marcadores(M, k, rng):
    """k marcadores por juego de parámetros, por transformada inversa sobre la matriz aplanada."""
    S, C = len(M), (MAX_GOLES + 1) ** 2
    cum = M.reshape(S, -1).cumsum(1)
    cum /= cum[:, -1:]
    fila = np.arange(S)[:, None]
    # una sola búsqueda binaria para todas las filas: a la fila s se le suma s para que queden ordenadas
    idx = np.searchsorted((cum + fila).ravel(), (rng.random((S, k)) + fila).ravel(), side="right")
    idx = np.minimum(idx.reshape(S, k) - fila * C, C - 1)
    return idx // (MAX_GOLES + 1), idx % (MAX_GOLES + 1)


def simula_partido(mod, th, local, visita, rng, mult=(1.0, 1.0)):
    """mult: multiplicadores de los goles esperados de local y visita por la alineación titular."""
    l, m, rho = _lambdas_s(mod, th, local, visita)
    l, m = l * mult[0], m * mult[1]
    M = _matrices_dc(l, m, rho)
    gl, gv = _muestrea_marcadores(M, SIMS_POR_JUEGO, rng)          # (S, 10)
    N = gl.size
    res = {"1": gl > gv, "X": gl == gv, "2": gl < gv}
    # error estándar Montecarlo agrupado por juego de parámetros (las 10 simulaciones de un juego no son independientes)
    ee = {k: float(v.mean(1).std(ddof=1) / math.sqrt(len(v))) for k, v in res.items()}
    tri = np.tril(np.ones((MAX_GOLES + 1,) * 2, bool), -1)
    p_s = {"1": M[:, tri].sum(1), "X": np.trace(M, axis1=1, axis2=2), "2": M[:, tri.T].sum(1)}
    l0, m0 = lambdas(mod, local, visita)
    Ma = matriz(l0 * mult[0], m0 * mult[1], mod["rho"])
    p1, px, p2 = float(np.tril(Ma, -1).sum()), float(np.trace(Ma)), float(np.triu(Ma, 1).sum())
    tot = (gl + gv).ravel()
    marc, cuenta = np.unique(np.stack([gl.ravel(), gv.ravel()], 1), axis=0, return_counts=True)
    top = np.argsort(-cuenta)[:10]
    hist = np.bincount(np.minimum(tot, 7), minlength=8) / N
    return {"n": int(N), "juegos": len(th), "lm": [round(l0 * mult[0], 3), round(m0 * mult[1], 3), round(mod["rho"], 4)],
            "p": [round(float(res[k].mean()), 4) for k in "1X2"],
            "ee": [round(ee[k], 4) for k in "1X2"],
            "int90": [[round(float(np.percentile(p_s[k], 5)), 4), round(float(np.percentile(p_s[k], 95)), 4)] for k in "1X2"],
            "analitica": [round(p1, 4), round(px, 4), round(p2, 4)],
            "marcadores": [[int(marc[i, 0]), int(marc[i, 1]), round(float(cuenta[i] / N), 4)] for i in top],
            "goles": [round(float(gl.mean()), 2), round(float(gv.mean()), 2)],
            "total_p5_p95": [int(np.percentile(tot, 5)), int(np.percentile(tot, 95))],
            "hist_total": [round(float(x), 4) for x in hist],
            "over25": round(float((tot > 2).mean()), 4),
            "ambos": round(float(((gl > 0) & (gv > 0)).mean()), 4)}


def simula_temporada(clave, mod, th, act, fut, rng, pendientes=None, equipos=None):
    """10,000 temporadas: se juega el calendario restante y se suman a la tabla actual.
    Si se da `pendientes` (lista de (local, visita, True)), se usa ese calendario en vez de generarlo."""
    etiqueta1, n_arriba, etiqueta_arriba, n_desc, ida_vuelta = ZONAS[clave]
    equipos = equipos or sorted(set(act["local"]) | set(act["visita"]))
    ix = {e: i for i, e in enumerate(equipos)}
    th = th[:S_TEMP]
    E, T = len(equipos), S_TEMP * TEMP_POR_JUEGO
    pts0, gd0, gf0 = np.zeros(E), np.zeros(E), np.zeros(E)
    for r in act.itertuples():
        i, j = ix[r.local], ix[r.visita]
        gf0[i] += r.gl; gf0[j] += r.gv; gd0[i] += r.gl - r.gv; gd0[j] += r.gv - r.gl
        pts0[i] += 3 if r.gl > r.gv else 1 if r.gl == r.gv else 0
        pts0[j] += 3 if r.gv > r.gl else 1 if r.gl == r.gv else 0
    jugados = set(zip(act["local"], act["visita"]))
    if pendientes is not None:
        pass
    elif ida_vuelta:
        pendientes = [(a, b, True) for a in equipos for b in equipos if a != b and (a, b) not in jugados]
    else:   # Liga MX: una vuelta; la localía sale del calendario conocido o, si no, a la suerte
        conocidos = {frozenset((r.local, r.visita)): (r.local, r.visita) for r in fut.itertuples()} if len(fut) else {}
        pendientes = []
        for x in range(E):
            for y in range(x + 1, E):
                a, b = equipos[x], equipos[y]
                if (a, b) in jugados or (b, a) in jugados:
                    continue
                par = conocidos.get(frozenset((a, b)))
                pendientes.append((*par, True) if par else (a, b, False))
    pts = np.repeat(pts0[None, :], T, 0); gd = np.repeat(gd0[None, :], T, 0); gf = np.repeat(gf0[None, :], T, 0)
    filas = np.arange(T)
    for a, b, fijo in pendientes:
        gl, gv = (x.ravel() for x in _muestrea_marcadores(_matrices_dc(*_lambdas_s(mod, th, a, b)), TEMP_POR_JUEGO, rng))
        loc = np.full(T, ix[a]); vis = np.full(T, ix[b])
        if not fijo:
            gl2, gv2 = (x.ravel() for x in _muestrea_marcadores(_matrices_dc(*_lambdas_s(mod, th, b, a)), TEMP_POR_JUEGO, rng))
            voltea = rng.random(T) < 0.5
            loc, vis = np.where(voltea, ix[b], loc), np.where(voltea, ix[a], vis)
            gl, gv = np.where(voltea, gl2, gl), np.where(voltea, gv2, gv)
        # local y visita son columnas distintas en cada fila, así que la suma con índices no tiene choques
        pts[filas, loc] += np.where(gl > gv, 3, np.where(gl == gv, 1, 0))
        pts[filas, vis] += np.where(gv > gl, 3, np.where(gl == gv, 1, 0))
        gd[filas, loc] += gl - gv; gd[filas, vis] += gv - gl
        gf[filas, loc] += gl; gf[filas, vis] += gv
    # desempate: puntos, diferencia de goles, goles a favor y sorteo
    clave_orden = pts * 1e6 + (gd + 1000) * 1e2 + gf + rng.random((T, E)) * 0.1
    pos = np.empty((T, E), int)
    pos[filas[:, None], np.argsort(-clave_orden, 1)] = np.arange(1, E + 1)
    pj0 = np.bincount([ix[e] for e in act["local"]] + [ix[e] for e in act["visita"]], minlength=E)
    tabla = []
    for e in equipos:
        i = ix[e]
        tabla.append({"equipo": e, "pts": int(pts0[i]), "pj": int(pj0[i]),
                      "pts_esp": round(float(pts[:, i].mean()), 1),
                      "pts_p5_p95": [int(np.percentile(pts[:, i], 5)), int(np.percentile(pts[:, i], 95))],
                      "p1": round(float((pos[:, i] == 1).mean()), 4),
                      "parriba": round(float((pos[:, i] <= n_arriba).mean()), 4),
                      "pdesc": round(float((pos[:, i] > E - n_desc).mean()), 4) if n_desc else None,
                      "pos_media": round(float(pos[:, i].mean()), 1),
                      "pos_dist": [round(float(x), 4) for x in np.bincount(pos[:, i] - 1, minlength=E) / T]})
    tabla.sort(key=lambda r: -r["pts_esp"])
    return {"temporadas": T, "pendientes": len(pendientes), "etiqueta1": etiqueta1,
            "etiqueta_arriba": etiqueta_arriba, "n_desc": n_desc, "tabla": tabla}


# ----------------------------------------------------------------------------- copas europeas (modelo propio)
# Respaldo que no depende de ClubElo. Fuerza de un club en Europa:
#   ataque = ataque en su liga doméstica + ajuste de esa liga + desviación propia (solo clubes de ligas que no seguimos)
# Los ajustes por liga, la localía y las fuerzas de clubes de otras ligas se estiman con partidos de Champions y
# Europa League de ESPN (temporada anterior y actual), con decaimiento temporal y regularización.
ESPN_COPAS = {"ucl": ("uefa.champions", "Champions League"), "uel": ("uefa.europa", "Europa League")}
GRUPOS = list(ESPN_LIGA) + ["otras"]
RIDGE_LIGA, RIDGE_OTROS = 0.5, 2.0       # la desviación de clubes de otras ligas se encoge fuerte hacia la media
COPA_DESDE = date(2025, 7, 1)
COLS_FILA = ["fecha", "local", "visita", "gl", "gv", "gl_mt", "gv_mt", *FD_COLS.values(), "c1", "cx", "c2", "co25", "cu25"]


def eventos_copa(codigo):
    """Partidos de la copa por mes (ESPN acepta dates=AAAAMM) y ventana de la fase de liga actual."""
    base = get(f"{ESPN_SB}/{codigo}/scoreboard").json()
    fase = next((e for c in base["leagues"][0].get("calendar", []) for e in c.get("entries", [])
                 if "league" in e.get("label", "").lower()), None)
    hoy = date.today()
    ini_fase = date.fromisoformat(fase["startDate"][:10]) if fase else hoy - timedelta(days=60)
    fin_fase = date.fromisoformat(fase["endDate"][:10]) if fase else hoy + timedelta(days=100)
    hasta, m, eventos = max(fin_fase, hoy + timedelta(days=30)), COPA_DESDE, {}
    while m <= hasta:
        try:
            js = get(f"{ESPN_SB}/{codigo}/scoreboard?dates={m:%Y%m}&limit=500").json()
        except Exception:  # noqa: BLE001
            js = {}
        for ev in js.get("events", []):
            c = ev["competitions"][0]
            eq = {x.get("homeAway"): x for x in c.get("competitors", [])}
            if "home" not in eq or "away" not in eq:
                continue
            st = ev["status"]["type"]
            try:
                gl, gv = (int(eq["home"]["score"]), int(eq["away"]["score"])) if st.get("completed") else (None, None)
            except (TypeError, ValueError):
                continue
            eventos[ev["id"]] = {"id": ev["id"], "fecha": date.fromisoformat(ev["date"][:10]), "estado": st.get("state"),
                                 "local": eq["home"]["team"]["displayName"], "visita": eq["away"]["team"]["displayName"],
                                 "id_l": str(eq["home"]["team"]["id"]), "id_v": str(eq["away"]["team"]["id"]),
                                 "gl": gl, "gv": gv, "odds": (c.get("odds") or [None])[0]}
        m = (m + timedelta(days=32)).replace(day=1)
    return sorted(eventos.values(), key=lambda e: e["fecha"]), ini_fase, fin_fase


_EUROPA = {}


def partidos_europeos():
    """Todos los partidos jugados de Champions, Europa y Conference League: se ajustan juntos porque los clubes se
    mueven entre copas y así cada club tiene más partidos."""
    if "todos" not in _EUROPA:
        todos = {}
        for codigo in [c for c, _ in ESPN_COPAS.values()] + ["uefa.europa.conf"]:
            try:
                evs = _EUROPA[codigo] = eventos_copa(codigo)
            except Exception as e:  # noqa: BLE001
                print(f"  ESPN {codigo} no disponible: {e}")
                continue
            for ev in evs[0]:
                if ev["gl"] is not None:
                    todos[(ev["fecha"], ev["id_l"], ev["id_v"])] = ev
        _EUROPA["todos"] = sorted(todos.values(), key=lambda e: e["fecha"])
    return _EUROPA["todos"]


ESPN_EQUIPOS = {}         # liga -> {id ESPN: nombre ESPN}
TM_LIGA = {"premier": "GB1", "laliga": "ES1", "seriea": "IT1", "ligue1": "FR1", "eredivisie": "NL1", "ligamx": "MEX1"}
CACHE_V = {}              # caché de valores.py (Transfermarkt y alineaciones)
COPA_MOD = {}             # modelo de cada copa en formato de liga (para estimar el efecto de la alineación)
LESIONES = {}             # id ESPN del equipo -> nombres normalizados de lesionados/suspendidos
TM_CLUB = {}              # id ESPN del equipo -> (id Transfermarkt, nombre, valor de plantilla)


def equipos_espn_liga(clave):
    if clave not in ESPN_EQUIPOS:
        d = get(f"{ESPN_SB}/{ESPN_LIGA[clave]}/teams").json()
        ESPN_EQUIPOS[clave] = {str(t["id"]): t["displayName"]
                               for t in (x.get("team", x) for x in d["sports"][0]["leagues"][0]["teams"])}
    return ESPN_EQUIPOS[clave]


def bases_domesticas():
    """id ESPN del club -> (grupo, ataque, defensa) domésticos, centrados en la media de su liga para que sean
    comparables entre ligas (el nivel de cada liga lo pone el ajuste por liga y el valor de mercado)."""
    out = {}
    for clave in ESPN_LIGA:
        mod, mapa = MODELOS.get(clave), MAPAS.get(clave, {})
        if not mod:
            continue
        try:
            eqs = equipos_espn_liga(clave)
        except Exception:  # noqa: BLE001
            continue
        fds = {i: mapa.get(n) for i, n in eqs.items() if mapa.get(n) in mod["ataque"]}
        if not fds:
            continue
        ma = np.mean([mod["ataque"][f] for f in fds.values()])
        md = np.mean([mod["defensa"][f] for f in fds.values()])
        for i, f in fds.items():
            out[i] = (GRUPOS.index(clave), mod["ataque"][f] - ma, mod["defensa"][f] - md)
    return out


def asigna_clubes_tm(ids_nombres, bases):
    """Une cada club de ESPN con su club de Transfermarkt: los de nuestras ligas se buscan en su liga y el resto
    en las demás ligas europeas. Llena TM_CLUB."""
    ligas = valores.valores_clubes(CACHE_V)
    otros = [c for cod, cl in ligas.items() if cod not in TM_LIGA.values() for c in cl]
    for i, nombre in ids_nombres.items():
        if i in TM_CLUB:
            continue
        cand = ligas.get(TM_LIGA[GRUPOS[bases[i][0]]], []) if i in bases else otros
        cl = valores.asigna_club(nombre, cand) or (valores.asigna_club(nombre, otros + [c for cl in ligas.values() for c in cl], 0.75)
                                                  if i not in bases else None)
        if cl:
            TM_CLUB[i] = cl


def log_valores(ids):
    """id ESPN -> log(valor de plantilla) centrado. Clubes sin valor: percentil 10 (casi siempre son clubes chicos)."""
    v = {i: math.log(TM_CLUB[i][2]) for i in ids if i in TM_CLUB and TM_CLUB[i][2]}
    if not v:
        return {}
    bajo, media = float(np.percentile(list(v.values()), 10)), float(np.mean(list(v.values())))
    return {i: v.get(i, bajo) - media for i in ids}


def ajusta_copa(eventos, bases, hasta, valores_log=None):
    """Poisson ponderado y penalizado con gradiente analítico; ρ se fija en el promedio de las ligas domésticas.
    Sin valores_log (versión 1): ataque = doméstico + ajuste de liga (+ desviación propia si es de otra liga).
    Con valores_log (versión 2): ataque = ajuste de liga + δ·doméstico + γ·log(valor de plantilla) + desviación,
    con localía con prior 1.18. q = [c, localía, ajustes ataque (G), ajustes defensa (G), (δA, δD, γA, γD), desv. (2K)]."""
    d = [e for e in eventos if e["gl"] is not None and e["fecha"] < hasta]
    otros = sorted({i for e in d for i in (e["id_l"], e["id_v"]) if i not in bases})
    ko = {i: k for k, i in enumerate(otros)}
    nuevo = valores_log is not None
    G, K, E = len(GRUPOS), len(otros), (4 if nuevo else 0)
    V0 = 2 + 2 * G
    P = V0 + E + 2 * K

    def club(i):
        return bases[i] if i in bases else (G - 1, 0.0, 0.0)

    def fila(atacante, defensor, local):
        x = np.zeros(P)
        x[0], x[1] = 1, local
        ga, aa, _ = club(atacante)
        gd, _, dd = club(defensor)
        x[2 + ga] = 1
        x[2 + G + gd] = 1
        if atacante in ko:
            x[V0 + E + ko[atacante]] = 1
        if defensor in ko:
            x[V0 + E + K + ko[defensor]] = 1
        if not nuevo:
            return x, aa + dd
        x[V0], x[V0 + 1] = aa, dd
        x[V0 + 2], x[V0 + 3] = valores_log.get(atacante, 0.0), -valores_log.get(defensor, 0.0)
        return x, 0.0

    Xl, bl = zip(*[fila(e["id_l"], e["id_v"], 1) for e in d]) if d else ((), ())
    Xm, bm = zip(*[fila(e["id_v"], e["id_l"], 0) for e in d]) if d else ((), ())
    X, base = np.array(Xl + Xm), np.array(bl + bm)
    goles = np.array([e["gl"] for e in d] + [e["gv"] for e in d], float)
    w = np.tile(np.exp(-XI * np.array([(hasta - e["fecha"]).days for e in d])), 2)
    if nuevo:
        ridge = np.r_[0, 5.0, [1.0] * 2 * G, 2.0, 2.0, 0.5, 0.5, [RIDGE_OTROS] * 2 * K]
        prior = np.r_[0, math.log(1.18), np.zeros(2 * G), 1.0, 1.0, 0.0, 0.0, np.zeros(2 * K)]
    else:
        ridge = np.r_[0, 0, [RIDGE_LIGA] * 2 * G, [RIDGE_OTROS] * 2 * K]
        prior = np.zeros(P)

    def f(q):
        eta = X @ q + base
        lam = np.exp(eta)
        val = -(w * (goles * eta - lam)).sum() + (ridge * (q - prior) ** 2).sum()
        grad = -X.T @ (w * (goles - lam)) + 2 * ridge * (q - prior)
        return val, grad

    bounds = [(None, None), (-1, 1)] + [(-3, 3)] * (P - 2)
    q = minimize(f, np.r_[0.1, 0.2, prior[2:]], jac=True, method="L-BFGS-B", bounds=bounds).x
    lam = np.exp(X @ q + base)
    H = (X * (w * lam)[:, None]).T @ X + 2 * np.diag(ridge)
    rho = float(np.mean([m["rho"] for m in MODELOS.values()])) if MODELOS else -0.05
    return {"q": q, "cov": np.linalg.inv(H + np.eye(P) * 1e-9), "rho": rho, "ko": ko, "G": G, "K": K, "E": E,
            "n": len(d), "valores": valores_log, "apariciones": {g: int((X[:, 2 + i] > 0).sum()) for i, g in enumerate(GRUPOS)}}


def modelo_copa_estandar(ajuste, bases, equipos):
    """Lleva el modelo de copa al formato de las ligas (mu, home, rho, ataque/defensa por club) para reutilizar
    las mismas funciones de probabilidad y Montecarlo. equipos: {nombre: id ESPN}.
    Devuelve el modelo y una función que transforma muestras de q en muestras θ en ese formato."""
    G, K, E, ko, vl = ajuste["G"], ajuste["K"], ajuste["E"], ajuste["ko"], ajuste["valores"]
    V0 = 2 + 2 * G
    nombres = sorted(equipos)
    n, Pq = len(nombres), len(ajuste["q"])
    T = np.zeros((3 + 2 * n, Pq))
    b = np.zeros(3 + 2 * n)
    T[0, 0] = 1
    T[1, 1] = 1
    b[2] = ajuste["rho"]
    for j, nom in enumerate(nombres):
        i = equipos[nom]
        g, a, dd = bases[i] if i in bases else (G - 1, 0.0, 0.0)
        T[3 + j, 2 + g] = 1
        T[3 + n + j, 2 + G + g] = 1
        if E:
            T[3 + j, V0], T[3 + j, V0 + 2] = a, vl.get(i, 0.0)
            T[3 + n + j, V0 + 1], T[3 + n + j, V0 + 3] = dd, -vl.get(i, 0.0)
        else:
            b[3 + j], b[3 + n + j] = a, dd
        if i in ko:
            T[3 + j, V0 + E + ko[i]] = 1
            T[3 + n + j, V0 + E + K + ko[i]] = 1
    theta = T @ ajuste["q"] + b
    mod = {"mu": float(theta[0]), "home": float(theta[1]), "rho": float(theta[2]), "orden": nombres,
           "ataque": {e: float(theta[3 + j]) for j, e in enumerate(nombres)},
           "defensa": {e: float(theta[3 + n + j]) for j, e in enumerate(nombres)}}

    def muestras(rng):
        L = np.linalg.cholesky(ajuste["cov"])
        qs = ajuste["q"][None, :] + rng.standard_normal((S_PARAM, Pq)) @ L.T
        th = qs @ T.T + b
        th[:, 1] = th[:, 1].clip(-1, 1)
        return th
    return mod, muestras


def _df_eventos(evs):
    df = pd.DataFrame([{**dict.fromkeys(COLS_FILA, np.nan), "fecha": e["fecha"], "local": e["local"],
                        "visita": e["visita"], "gl": e["gl"], "gv": e["gv"], **_cuotas_espn(e["odds"])} for e in evs],
                      columns=COLS_FILA)
    return df


def _cuotas_resumen(codigo, e):
    """Cuotas del mercado de un partido jugado: vienen en el resumen del partido (se guarda en caché)."""
    valores.alineaciones(CACHE_V, codigo, e["id"], True)
    o = CACHE_V.get("alineaciones", {}).get(e["id"], {}).get("odds")
    c = _cuotas_espn(o)
    return [None if np.isnan(c[k]) else c[k] for k in ("c1", "cx", "c2")]


def _valida_copa(ajuste_val, bases, equipos, actuales, codigo):
    mod_v, _ = modelo_copa_estandar(ajuste_val, bases, equipos)
    preds = []
    for e in actuales:
        if e["gl"] is not None:
            p1, px, p2, _ = prob_1x2(mod_v, e["local"], e["visita"])
            preds.append({"fecha": e["fecha"].isoformat(), "local": e["local"], "visita": e["visita"], "gl": e["gl"],
                          "gv": e["gv"], "p": [round(p1, 4), round(px, 4), round(p2, 4)], "c": _cuotas_resumen(codigo, e)})
    return preds


def _difiere(preds, umbral=0.15):
    n = 0
    for p in preds:
        if None in p["c"]:
            continue
        imp = np.array([1 / c for c in p["c"]])
        imp /= imp.sum()
        n += bool(np.abs(np.array(p["p"]) - imp).max() > umbral)
    return n


def procesa_copa(clave):
    codigo, nombre = ESPN_COPAS[clave]
    europa = partidos_europeos()
    if codigo not in _EUROPA:
        return None
    eventos, ini_fase, fin_fase = _EUROPA[codigo]
    hoy = date.today()
    ini_temp = date(hoy.year if hoy.month >= 7 else hoy.year - 1, 7, 1)
    jugados = [e for e in eventos if e["gl"] is not None]
    if len(europa) < 50:
        print(f"  {nombre}: solo {len(europa)} partidos europeos en ESPN, se omite")
        return None
    bases = bases_domesticas()
    nombres_id = {e[i]: e[k] for e in europa + eventos for k, i in (("local", "id_l"), ("visita", "id_v"))}
    try:
        asigna_clubes_tm(nombres_id, bases)
    except Exception as e:  # noqa: BLE001
        print("  Transfermarkt no disponible para valores de plantilla:", e)
    vlog = log_valores(set(nombres_id)) or None
    actuales = [e for e in eventos if e["fecha"] >= ini_temp]
    equipos = {e[k]: e[i] for e in actuales for k, i in (("local", "id_l"), ("visita", "id_v"))}

    # validación fuera de muestra (ajuste con la temporada pasada, pronóstico de la actual): versión 1 vs 2 vs mercado
    hist = _df_eventos([e for e in jugados if e["fecha"] < ini_temp])
    val = {}
    for version, vl in (("v1", None), ("v2", vlog)):
        if version == "v2" and vl is None:
            continue
        aj = ajusta_copa(europa, bases, ini_temp, vl)
        if aj["n"] >= 50:
            preds = _valida_copa(aj, bases, equipos, actuales, codigo)
            val[version] = (preds, metricas(preds, hist))
    elegida = min(val, key=lambda k: val[k][1].get("logloss", 9)) if val else ("v2" if vlog else "v1")
    ajuste = ajusta_copa(europa, bases, hoy + timedelta(days=1), vlog if elegida == "v2" else None)
    mod, muestras = modelo_copa_estandar(ajuste, bases, equipos)
    COPA_MOD[clave] = mod
    preds, met = val.get(elegida, ([], {}))
    if met:
        met["difiere15"] = _difiere(preds)
        met["versiones"] = {k: {"logloss": round(v[1]["logloss"], 4), "mercado": round(v[1].get("logloss_mercado", float("nan")), 4),
                                "difiere15": _difiere(v[0]), "n": v[1]["n"]} for k, v in val.items()}
    n_dom = sum(1 for i in set(equipos.values()) if i in bases)
    n_val = sum(1 for i in set(equipos.values()) if i in TM_CLUB)
    q = ajuste["q"]
    print(f"  {nombre}: {ajuste['n']} partidos europeos para ajustar, {len(equipos)} clubes esta temporada "
          f"({n_dom} con fuerza doméstica, {n_val} con valor de mercado); versión elegida {elegida}, localía ×{math.exp(q[1]):.2f}")
    for k, (p, m) in val.items():
        print(f"    validación {k}: {m['n']} partidos, log-loss {m['logloss']:.3f} (base {m['logloss_base']:.3f}"
              + (f", mercado {m['logloss_mercado']:.3f}" if "logloss_mercado" in m else "") + f"), difieren >15 pts del mercado: {_difiere(p)}")
    predmap = {(p["fecha"], p["local"]): p for p in preds}

    th = muestras(_rng(clave))
    prox_evs = sorted([e for e in eventos if e["estado"] == "pre" and hoy <= e["fecha"] <= hoy + timedelta(days=30)],
                      key=lambda e: e["fecha"])[:40]
    fut = _df_eventos(prox_evs)
    fase = [e for e in eventos if ini_fase <= e["fecha"] <= fin_fase]
    act_fase = _df_eventos([e for e in fase if e["gl"] is not None])
    equipos_fase = sorted({e["local"] for e in fase} | {e["visita"] for e in fase})
    temporada = None
    if 30 <= len(equipos_fase) <= 40:
        pend = [(e["local"], e["visita"], True) for e in fase if e["gl"] is None]
        temporada = simula_temporada(clave, mod, th, act_fase, act_fase.iloc[0:0], _rng(clave, "fase"),
                                     pendientes=pend, equipos=equipos_fase)

    def filas(df):
        out = []
        for r in df[COLS_FILA].to_dict("records"):
            r = {k: limpio(v) for k, v in r.items()}
            r["fecha"] = r["fecha"].isoformat()
            if (r["fecha"], r["local"]) in predmap:
                r["p_prev"] = predmap[(r["fecha"], r["local"])]["p"]
            out.append(r)
        return out
    prox = filas(fut)
    PENDIENTES_SIM.append({"clave": clave, "codigo": codigo, "mod": mod, "th": th, "filas": prox,
                           "ids": [(e["id"], e["id_l"], e["id_v"]) for e in prox_evs]})
    recientes = _df_eventos([e for e in actuales if e["gl"] is not None]).sort_values("fecha", ascending=False)
    fuente = ("valor de mercado (Transfermarkt) + fuerzas domésticas + " if elegida == "v2" else "fuerzas domésticas + ")
    return {"nombre": nombre, "origen": f"modelo propio: {fuente}{ajuste['n']} partidos europeos de ESPN",
            "actualizado": max(e["fecha"] for e in jugados).isoformat(), "inicio": ini_temp.isoformat(),
            "temporada_sim": temporada,
            "modelo": {"mu": mod["mu"], "home": mod["home"], "rho": mod["rho"],
                       "equipos": {e: [round(mod["ataque"][e], 4), round(mod["defensa"][e], 4)] for e in mod["orden"]}},
            "tabla": estadisticas(act_fase, ini_fase) if len(act_fase) else [], "proximos": prox,
            "recientes": filas(recientes), "backtest": met}


# ----------------------------------------------------------------------------- alineación titular
def _ultimo_partido():
    """id ESPN del equipo -> (fecha, código de competición, id de partido) de su último partido jugado."""
    out = {}
    fuentes = [(ESPN_LIGA[c], evs) for c, evs in EVENTOS_ESPN.items()] + [(cod, v[0]) for cod, v in _EUROPA.items() if cod != "todos"]
    for codigo, evs in fuentes:
        for e in evs:
            if e.get("gl") is None or "id_l" not in e:
                continue
            for i in (e["id_l"], e["id_v"]):
                if i not in out or e["fecha"] > out[i][0]:
                    out[i] = (e["fecha"], codigo, e["id"])
    return out


def _xi_info(xi, squad, fuera):
    jug = valores.valora_xi(xi, squad, fuera)
    of, de = valores.lineas(jug)
    of_t, de_t = valores.lineas([{"grupo": j[1], "valor": j[2]} for j in valores.mejor_xi(squad)])
    return jug, of / of_t, de / de_t


def estima_beta(log=print):
    """β_a y β_d con los partidos jugados cuya alineación conocemos: goles ~ Poisson(λ_modelo · r_ataque^βa · r_defensa_rival^-βd).
    r = valor de las líneas del XI que jugó / valor del mejor XI de la plantilla."""
    filas = []
    fuentes = [(c, ESPN_LIGA[c], evs) for c, evs in EVENTOS_ESPN.items()] + \
              [(c, cod, _EUROPA[cod][0]) for c, (cod, _) in ESPN_COPAS.items() if cod in _EUROPA]
    inv = {c: {v: k for k, v in MAPAS.get(c, {}).items()} for c in ESPN_LIGA}
    hoy = date.today()
    ini_temp = date(hoy.year if hoy.month >= 7 else hoy.year - 1, 7, 1)
    for clave, codigo, evs in fuentes:
        mod = MODELOS.get(clave) or COPA_MOD.get(clave)
        if not mod:
            continue
        for e in evs:
            if e.get("gl") is None or "id_l" not in e or e["fecha"] < ini_temp:
                continue
            nl, nv = (MAPAS[clave].get(e["local"]), MAPAS[clave].get(e["visita"])) if clave in ESPN_LIGA else (e["local"], e["visita"])
            if not nl or not nv or (clave in ESPN_LIGA and (nl not in mod["ataque"] or nv not in mod["ataque"])):
                continue
            xi = valores.alineaciones(CACHE_V, codigo, e["id"], True)
            if e["id_l"] not in xi or e["id_v"] not in xi:
                continue
            sq = valores.plantillas(CACHE_V, [TM_CLUB[i][0] for i in (e["id_l"], e["id_v"]) if i in TM_CLUB], log)
            if not all(i in TM_CLUB and TM_CLUB[i][0] in sq for i in (e["id_l"], e["id_v"])):
                continue
            _, ral, rdl = _xi_info(xi[e["id_l"]], sq[TM_CLUB[e["id_l"]][0]], set())
            _, rav, rdv = _xi_info(xi[e["id_v"]], sq[TM_CLUB[e["id_v"]][0]], set())
            l, m = lambdas(mod, nl, nv)
            filas.append((l, m, mod["rho"], math.log(ral), math.log(rdl), math.log(rav), math.log(rdv), e["gl"], e["gv"]))
    if len(filas) < 60:
        log(f"  Alineaciones: solo {len(filas)} partidos con XI y valores; se usa β = 0.3 fijo")
        return (0.3, 0.3), {"n": len(filas)}
    F = np.array(filas)
    l, m, rho, ral, rdl, rav, rdv, gl, gv = F.T

    def nll(b):
        el = np.log(l) + b[0] * ral - b[1] * rdv
        em = np.log(m) + b[0] * rav - b[1] * rdl
        return -(gl * el - np.exp(el)).sum() - (gv * em - np.exp(em)).sum() + 2.0 * (b ** 2).sum()
    b = minimize(nll, [0.3, 0.3], method="L-BFGS-B", bounds=[(0, 1.5), (0, 1.5)]).x

    def logloss(bb):
        tot = 0.0
        for li, mi, ri, a1, d1, a2, d2, x, y in filas:
            M = matriz(li * math.exp(bb[0] * a1 - bb[1] * d2), mi * math.exp(bb[0] * a2 - bb[1] * d1), ri)
            p = [np.tril(M, -1).sum(), np.trace(M), np.triu(M, 1).sum()][0 if x > y else 1 if x == y else 2]
            tot -= math.log(max(p, 1e-9))
        return tot / len(filas)
    info = {"n": len(filas), "beta": [round(float(b[0]), 3), round(float(b[1]), 3)],
            "logloss_con": round(logloss(b), 4), "logloss_sin": round(logloss((0, 0)), 4)}
    log(f"  Alineaciones: β_ataque {b[0]:.2f}, β_defensa {b[1]:.2f} con {len(filas)} partidos; log-loss "
        f"{info['logloss_con']:.4f} con alineación vs {info['logloss_sin']:.4f} sin ella (dentro de muestra)")
    if info["logloss_con"] >= info["logloss_sin"]:
        b = np.array([0.1, 0.1])
        log("  La alineación no mejoró el log-loss: se usa β = 0.1 (efecto pequeño)")
    return (float(b[0]), float(b[1])), info


def simula_proximos(log=print):
    """Simula todos los partidos próximos con el XI esperado de cada equipo."""
    ultimo = _ultimo_partido()
    inv = {c: {v: k for k, v in MAPAS.get(c, {}).items()} for c in ESPN_LIGA}
    ids_eq = {}
    for p in PENDIENTES_SIM:          # ligas: de nombre football-data a id ESPN y partido ESPN
        if p["clave"] in ESPN_LIGA:
            eqs = {n: i for i, n in equipos_espn_liga(p["clave"]).items()}
            evs = EVENTOS_ESPN.get(p["clave"], [])
            p["ids"] = []
            for r in p["filas"]:
                il, iv = eqs.get(inv[p["clave"]].get(r["local"])), eqs.get(inv[p["clave"]].get(r["visita"]))
                ev = next((e for e in evs if e.get("id_l") == il and e.get("id_v") == iv), None)
                p["ids"].append((ev["id"] if ev else None, il, iv))
        for _, il, iv in p["ids"]:
            for i in (il, iv):
                if i:
                    ids_eq[i] = True
    # clubes de Transfermarkt para todos los equipos que juegan
    bases = bases_domesticas()
    nombres = {}
    for c in ESPN_LIGA:
        try:
            nombres.update(equipos_espn_liga(c))
        except Exception:  # noqa: BLE001
            pass
    for cod, v in _EUROPA.items():
        if cod != "todos":
            for e in v[0]:
                nombres[e["id_l"]], nombres[e["id_v"]] = e["local"], e["visita"]
    try:
        asigna_clubes_tm({i: nombres[i] for i in nombres}, bases)
        valores.plantillas(CACHE_V, sorted({TM_CLUB[i][0] for i in nombres if i in TM_CLUB}), log)
    except Exception as e:  # noqa: BLE001
        log("  Transfermarkt no disponible, se simula sin alineaciones:", e)
    beta, info_beta = estima_beta(log)
    squads = {k: v["jugadores"] for k, v in CACHE_V.get("plantillas", {}).items()}
    hoy = date.today()
    n_conf = n_ult = 0
    for p in PENDIENTES_SIM:
        for r, (ev_id, il, iv) in zip(p["filas"], p["ids"]):
            mult, info = [1.0, 1.0], {"beta": [round(beta[0], 3), round(beta[1], 3)]}
            confirmada = {}
            if ev_id and date.fromisoformat(r["fecha"]) <= hoy + timedelta(days=1):
                confirmada = valores.alineaciones(CACHE_V, p["codigo"], ev_id, False)
            rs = {}
            for lado, i in (("local", il), ("visita", iv)):
                if not i or i not in TM_CLUB or TM_CLUB[i][0] not in squads:
                    continue
                if i in confirmada:
                    xi, fuente = confirmada[i], "alineación confirmada por ESPN"
                    n_conf += 1
                elif i in ultimo:
                    f, cod, eid = ultimo[i]
                    xi = valores.alineaciones(CACHE_V, cod, eid, True).get(i)
                    fuente = f"XI del último partido ({f.isoformat()})"
                    n_ult += 1
                else:
                    continue
                if not xi:
                    continue
                jug, ra, rd = _xi_info(xi, squads[TM_CLUB[i][0]], LESIONES.get(i, set()) if i not in confirmada else set())
                rs[lado] = (ra, rd)
                info[lado] = {"fuente": fuente, "club_tm": TM_CLUB[i][1], "valor_plantilla": TM_CLUB[i][2],
                              "r_ataque": round(ra, 3), "r_defensa": round(rd, 3),
                              "jugadores": [{"nombre": j["nombre"], "grupo": j["grupo"], "valor": j["valor"],
                                             **({"reemplazo_de": j["reemplazo_de"]} if j.get("reemplazo_de") else {})} for j in jug]}
            if "local" in rs and "visita" in rs:
                mult = [rs["local"][0] ** beta[0] * rs["visita"][1] ** -beta[1],
                        rs["visita"][0] ** beta[0] * rs["local"][1] ** -beta[1]]
                info["mult"] = [round(mult[0], 3), round(mult[1], 3)]
            r["sim"] = simula_partido(p["mod"], p["th"], r["local"], r["visita"],
                                      _rng(p["clave"], f"{r['fecha']}{r['local']}{r['visita']}"), tuple(mult))
            r["xi"] = info
    log(f"  Simulados {sum(len(p['filas']) for p in PENDIENTES_SIM)} partidos próximos; alineaciones: "
        f"{n_conf} confirmadas, {n_ult} del último partido")
    return info_beta


# ----------------------------------------------------------------------------- estadísticas
def estadisticas(jug, inicio_temp):
    act = jug[jug["fecha"] >= inicio_temp].sort_values("fecha")
    equipos = sorted(set(act["local"]) | set(act["visita"]))
    tabla = []
    for e in equipos:
        L = act[act["local"] == e]
        V = act[act["visita"] == e]
        gf = L["gl"].sum() + V["gv"].sum()
        gc = L["gv"].sum() + V["gl"].sum()
        g = (L["gl"] > L["gv"]).sum() + (V["gv"] > V["gl"]).sum()
        emp = (L["gl"] == L["gv"]).sum() + (V["gl"] == V["gv"]).sum()
        pj = len(L) + len(V)
        todos = pd.concat([L.assign(gf=L["gl"], gc=L["gv"]), V.assign(gf=V["gv"], gc=V["gl"])]).sort_values("fecha")
        forma = "".join("G" if r.gf > r.gc else "E" if r.gf == r.gc else "P" for r in todos.tail(5).itertuples())
        fila = {"equipo": e, "pj": int(pj), "g": int(g), "e": int(emp), "p": int(pj - g - emp),
                "gf": int(gf), "gc": int(gc), "pts": int(3 * g + emp), "forma": forma,
                "over25": float(((todos["gf"] + todos["gc"]) > 2.5).mean()) if pj else 0,
                "ambos": float(((todos["gf"] > 0) & (todos["gc"] > 0)).mean()) if pj else 0,
                "porteria0": float((todos["gc"] == 0).mean()) if pj else 0,
                "pts_local": int(3 * (L["gl"] > L["gv"]).sum() + (L["gl"] == L["gv"]).sum()),
                "pts_visita": int(3 * (V["gv"] > V["gl"]).sum() + (V["gl"] == V["gv"]).sum())}
        for k in ("tiros", "arco", "corners", "amarillas"):
            v = pd.concat([L[f"{k}_l"], V[f"{k}_v"]])
            fila[k] = None if v.isna().all() else round(float(v.mean()), 2)
        tabla.append(fila)
    tabla.sort(key=lambda r: (-r["pts"], -(r["gf"] - r["gc"]), -r["gf"]))
    return tabla


def limpio(v):
    if isinstance(v, float) and math.isnan(v):
        return None
    if isinstance(v, (np.floating,)):
        return None if np.isnan(v) else float(v)
    if isinstance(v, np.integer):
        return int(v)
    return v


def procesa_liga(clave, fuente):
    nombre, cod_fd, cod_of = LIGAS[clave]
    origen = None
    if fuente in ("auto", "football-data"):
        try:
            jug, fut = carga_football_data(clave, cod_fd)
            origen = "football-data.co.uk"
            if cod_fd != "MEX":
                fut = proximos_football_data(cod_fd)
        except Exception as e:  # noqa: BLE001
            print(f"  football-data.co.uk falló para {nombre}: {e}")
    if origen is None and cod_of and fuente in ("auto", "openfootball"):
        jug, fut = carga_openfootball(cod_of)
        origen = "openfootball (GitHub)"
    if origen is None:
        print(f"  {nombre}: sin fuente disponible, se omite")
        return None
    if clave in ESPN_LIGA:
        try:
            extra, prox = complemento_espn(clave, jug, fut)
            if len(extra):
                jug = pd.concat([jug, extra], ignore_index=True)
                # lo que ESPN ya da por jugado deja de ser "próximo"
                fut = fut[~fut.apply(lambda r: ((extra["local"] == r["local"]) & extra["fecha"].map(
                    lambda f: abs((f - r["fecha"]).days) <= 1)).any(), axis=1)] if len(fut) else fut
                origen += f" + ESPN ({len(extra)} recientes)"
            if len(prox):
                fut = pd.concat([fut, prox], ignore_index=True)
        except Exception as e:  # noqa: BLE001
            print(f"  complemento ESPN falló para {nombre}: {e}")

    hoy = date.today()
    ultimo = max(jug["fecha"])
    # inicios de temporada (o de torneo corto en Liga MX): fechas tras un hueco largo sin partidos
    fechas = sorted(set(jug["fecha"]))
    hueco = 17 if cod_fd == "MEX" else 30   # Apertura y Clausura se separan por ~3 semanas
    inicios = [fechas[0]] + [fechas[i + 1] for i in range(len(fechas) - 1) if (fechas[i + 1] - fechas[i]).days > hueco]
    inicio = inicios[-1]
    act = jug[jug["fecha"] >= inicio]

    print(f"  {nombre}: {len(jug)} partidos ({origen}), temporada desde {inicio}, último {ultimo}")
    mod = ajusta_dixon_coles(jug, ultimo + timedelta(days=1), inicios, con_cov=True)
    MODELOS[clave] = mod
    preds, met = backtest(jug, inicios)

    proximos = fut[fut["fecha"] >= hoy].sort_values("fecha").head(40)
    t0 = datetime.now()
    th = muestrea_parametros(mod, _rng(clave))
    try:
        temporada = simula_temporada(clave, mod, th, act, fut, _rng(clave, "temporada"))
    except Exception as e:  # noqa: BLE001
        print("  simulación de temporada falló:", e)
        temporada = None
    print(f"  Montecarlo: {len(proximos)} partidos próximos (se simulan al final, con alineaciones)"
          + (f", {temporada['temporadas']:,} temporadas ({temporada['pendientes']} partidos pendientes)" if temporada else "")
          + f" en {(datetime.now() - t0).total_seconds():.1f} s")
    recientes = act.sort_values("fecha", ascending=False)      # toda la temporada, para filtrar por equipo
    predmap = {(p["fecha"], p["local"]): p for p in preds}
    cols = ["fecha", "local", "visita", "gl", "gv", "gl_mt", "gv_mt", *FD_COLS.values(), "c1", "cx", "c2", "co25", "cu25"]
    def filas(df):
        out = []
        for r in df[cols].to_dict("records"):
            r = {k: limpio(v) for k, v in r.items()}
            r["fecha"] = r["fecha"].isoformat()
            pp = predmap.get((r["fecha"], r["local"]))
            if pp:
                r["p_prev"] = pp["p"]
            out.append(r)
        return out

    prox_filas = filas(proximos)
    PENDIENTES_SIM.append({"clave": clave, "codigo": ESPN_LIGA.get(clave), "mod": mod, "th": th, "filas": prox_filas, "ids": []})
    return {"nombre": nombre, "origen": origen, "actualizado": ultimo.isoformat(), "inicio": inicio.isoformat(),
            "temporada_sim": temporada, "espn": MAPAS.get(clave, {}),     # nombre ESPN -> nombre del modelo (pestaña En vivo)
            "modelo": {"mu": mod["mu"], "home": mod["home"], "rho": mod["rho"],
                       "equipos": {e: [round(mod["ataque"][e], 4), round(mod["defensa"][e], 4)] for e in mod["ataque"]
                                   if e in set(act["local"]) | set(act["visita"])},
                "se": {e: [round(v[0], 4), round(v[1], 4)] for e, v in mod["se"].items()
                       if e in set(act["local"]) | set(act["visita"])}},
            "tabla": estadisticas(jug, inicio), "proximos": prox_filas, "recientes": filas(recientes),
            "backtest": met}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fuente", default="auto", choices=["auto", "football-data", "openfootball"])
    ap.add_argument("--salida", default="index.html")
    ap.add_argument("--sin-jugadores", action="store_true", help="omite plantillas, lesiones y noticias (más rápido)")
    args = ap.parse_args()
    data = {"generado": datetime.now().isoformat(timespec="minutes"), "ligas": {}, "copas": {}, "elo": None}
    CACHE_V.update(valores.carga_cache())
    # primero jugadores: las lesiones y suspensiones se usan para armar el XI esperado de cada partido
    if not args.sin_jugadores:
        print("Jugadores (ESPN, Fantasy PL, Transfermarkt); tarda 3 a 5 minutos")
        try:
            from jugadores import recopila
            data["jugadores"] = recopila()
            for comp in data["jugadores"]["competiciones"].values():
                for j in comp["jugadores"]:
                    if (j.get("lesion") or {}).get("estado") in ("Lesionado", "Suspendido", "Baja"):
                        LESIONES.setdefault(j["equipo_id"], set()).add(valores.normaliza(j["nombre"]))
        except Exception as e:  # noqa: BLE001
            print("  jugadores no disponibles:", e)
    for clave in LIGAS:
        print("Procesando", LIGAS[clave][0])
        try:
            r = procesa_liga(clave, args.fuente)
        except Exception as e:  # noqa: BLE001
            print("  error:", e)
            r = None
        if r:
            data["ligas"][clave] = r
    for clave, (_, nombre) in ESPN_COPAS.items():
        print("Procesando", nombre)
        try:
            r = procesa_copa(clave)
            if r:
                data["ligas"][clave] = r
        except Exception as e:  # noqa: BLE001
            print("  error:", e)
    try:
        data["elo"] = carga_clubelo()
        print(f"ClubElo: {len(data['elo'])} clubes")
    except Exception as e:  # noqa: BLE001
        print("ClubElo no disponible (no importa: Champions/Europa usan el modelo propio):", e)
    data["copas"] = COPAS
    print("Alineaciones (ESPN) y valores de mercado (Transfermarkt); la primera vez tarda unos minutos más")
    t0 = datetime.now()
    try:
        data["alineacion_beta"] = simula_proximos()
    except Exception as e:  # noqa: BLE001
        print("  alineaciones no disponibles, se simula sin ellas:", e)
        for p in PENDIENTES_SIM:
            for r in p["filas"]:
                r["sim"] = simula_partido(p["mod"], p["th"], r["local"], r["visita"], _rng(p["clave"], f"{r['fecha']}{r['local']}{r['visita']}"))
    valores.guarda_cache(CACHE_V)
    print(f"  listo en {(datetime.now() - t0).total_seconds():.0f} s")
    escribe_pagina(data, args.salida)


def escribe_pagina(data, salida):
    """Inserta los datos en plantilla.html y escribe una página que se abre con doble clic."""
    import os
    carpeta = os.path.dirname(os.path.abspath(__file__))
    plantilla = open(os.path.join(carpeta, "plantilla.html"), encoding="utf-8").read()
    js = "window.DATOS = " + json.dumps(data, ensure_ascii=False, default=limpio).replace("</", "<\\/") + ";"
    cuerpo = plantilla.replace("/*DATOS*/", js)
    if salida.endswith("index.html"):
        cuerpo = ('<!doctype html>\n<html lang="es">\n<head>\n<meta charset="utf-8">\n'
                  '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
                  + cuerpo.replace("<div class=\"wrap\">", "</head>\n<body>\n<div class=\"wrap\">", 1) + "\n</body>\n</html>\n")
    with open(salida, "w", encoding="utf-8") as f:
        f.write(cuerpo)
    print("Escrito", salida)


if __name__ == "__main__":
    sys.exit(main())
