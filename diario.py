"""
Diario de pronósticos: guarda lo que la página decía ANTES de cada partido y lo califica con el resultado real.

El backtest de la pestaña Fiabilidad reconstruye pronósticos del pasado; el diario es la prueba honesta: el
pronóstico queda congelado al empezar el partido (con la alineación y las cuotas de ese momento) y después solo
se le agrega el resultado. Se guarda en diario.json; en la nube vive en la rama "diario" del repositorio.
"""
import json
import os
from datetime import date, datetime, timedelta, timezone

import numpy as np

RUTA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diario.json")
DIAS_SIN_RESULTADO = 10     # pasado este plazo sin marcador (aplazado, nombre distinto) deja de esperarse


def carga(ruta=RUTA):
    try:
        with open(ruta, encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d.get("pronosticos"), dict) else {"pronosticos": {}}
    except (OSError, ValueError):
        return {"pronosticos": {}}


def guarda(diario, ruta=RUTA):
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(diario, f, ensure_ascii=False, separators=(",", ":"))


def _mercado(c):
    if not c or any(v is None or not v > 1 for v in c):
        return None
    imp = np.array([1 / v for v in c])
    return [round(float(x), 4) for x in imp / imp.sum()]


def _empezo(r, ahora):
    """¿El partido ya empezó? Con la hora de ESPN es exacto; sin ella se congela desde el día del partido."""
    if r.get("estado") and r["estado"] != "pre":
        return True
    if r.get("hora"):
        try:
            return datetime.fromisoformat(r["hora"].replace("Z", "+00:00")) <= ahora
        except ValueError:
            pass
    return date.fromisoformat(r["fecha"]) <= ahora.astimezone().date()


def registra(diario, ligas, mezcla_mercado):
    """Anota o actualiza el pronóstico de cada partido próximo que todavía no empieza."""
    ahora = datetime.now(timezone.utc)
    P = diario["pronosticos"]
    nuevos = actualizados = 0
    for comp, L in ligas.items():
        alpha = (L.get("mezcla") or {}).get("alpha", 1.0)
        for r in L.get("proximos", []):
            s = r.get("sim")
            if not s or _empezo(r, ahora):
                continue
            clave = f"{comp}|{r['fecha']}|{r['local']}|{r['visita']}"
            c = [r.get("c1"), r.get("cx"), r.get("c2")]
            pm, pk = s["analitica"], _mercado(c)
            ent = P.get(clave) or {"comp": comp, "fecha": r["fecha"], "local": r["local"], "visita": r["visita"],
                                   "primero": ahora.isoformat(timespec="minutes")}
            xi = r.get("xi") or {}
            nuevo = {"hora": r.get("hora"), "p": pm, "pk": pk, "c": c if pk else None, "alpha": alpha,
                     "pc": [round(float(x), 4) for x in mezcla_mercado(pm, c, alpha)] if pk else None, "over25": s.get("over25"),
                     "xi": "confirmada" if "confirmada" in str((xi.get("local") or {}).get("fuente", "")) else
                           ("ultimo" if xi.get("local") else None)}
            if all(ent.get(k) == v for k, v in nuevo.items()):
                continue                    # sin cambios: no se toca (así el archivo solo cambia cuando hay algo nuevo)
            nuevos += clave not in P
            actualizados += clave in P
            ent.update(nuevo, ts=ahora.isoformat(timespec="minutes"))
            P[clave] = ent
    return nuevos, actualizados


def califica(diario, ligas):
    """Pone el marcador final a los pronósticos cuyo partido ya está en los resultados."""
    P = diario["pronosticos"]
    pendientes = [e for e in P.values() if "gl" not in e and not e.get("sin_resultado")]
    if not pendientes:
        return 0
    res = {}
    for comp, L in ligas.items():
        for r in L.get("recientes", []):
            if r.get("gl") is not None and r.get("gv") is not None:
                res.setdefault((comp, r["local"], r["visita"]), []).append((date.fromisoformat(r["fecha"]), r))
    hoy, n = date.today(), 0
    for e in pendientes:
        f = date.fromisoformat(e["fecha"])
        m = [r for fr, r in res.get((e["comp"], e["local"], e["visita"]), []) if abs((fr - f).days) <= 2]
        if m:
            e["gl"], e["gv"] = int(m[0]["gl"]), int(m[0]["gv"])
            n += 1
        elif (hoy - f).days > DIAS_SIN_RESULTADO:
            e["sin_resultado"] = True
    return n


def _metricas(ents):
    res = np.array([0 if e["gl"] > e["gv"] else 1 if e["gl"] == e["gv"] else 2 for e in ents])
    idx = np.arange(len(ents))
    out = {"n": len(ents)}
    for k, nombre in (("p", "modelo"), ("pc", "combinado"), ("pk", "mercado")):
        sel = [i for i, e in enumerate(ents) if e.get(k)]     # mercado solo existe en partidos con cuotas
        if not sel:
            continue
        Pm = np.array([ents[i][k] for i in sel])
        Y = np.eye(3)[res[sel]]
        out[nombre] = {"n": len(sel),
                       "logloss": round(float(-np.mean(np.log(np.clip(Pm[np.arange(len(sel)), res[sel]], 1e-9, 1)))), 4),
                       "brier": round(float(np.mean(((Pm - Y) ** 2).sum(1))), 4),
                       "acierto": round(float(np.mean(Pm.argmax(1) == res[sel])), 4)}
    # modelo vs mercado en los mismos partidos (los que tienen cuotas)
    con = [i for i, e in enumerate(ents) if e.get("pk")]
    if con:
        Pm = np.array([ents[i]["p"] for i in con])
        out["modelo_con_cuotas"] = round(float(-np.mean(np.log(np.clip(Pm[np.arange(len(con)), res[con]], 1e-9, 1)))), 4)
    # calibración del modelo (las tres probabilidades de cada partido)
    P = np.array([e["p"] for e in ents])
    Y = np.eye(3)[res]
    cal = []
    for lo, hi in zip(np.linspace(0, 1, 6)[:-1], np.linspace(0, 1, 6)[1:]):
        s = (P >= lo) & (P < hi)
        if s.sum() >= 5:
            cal.append({"rango": f"{int(lo * 100)}–{int(hi * 100)}%", "n": int(s.sum()),
                        "pred": round(float(P[s].mean()), 4), "obs": round(float(Y[s].mean()), 4)})
    out["calibracion"] = cal
    # el favorito del modelo, por nivel de confianza (lo que usan los "picks más probables")
    fav = P.max(1)
    acierta = P.argmax(1) == res
    out["favorito"] = [{"rango": f"{int(lo * 100)}–{int(hi * 100)}%", "n": int(s.sum()),
                        "pred": round(float(fav[s].mean()), 4), "obs": round(float(acierta[s].mean()), 4)}
                       for lo, hi in ((.4, .5), (.5, .6), (.6, .7), (.7, 1.0)) for s in [(fav >= lo) & ((fav < hi) | (hi == 1.0))] if s.sum()]
    return out


def resumen(diario):
    """Métricas por competición y en total, más los últimos partidos calificados, para la página."""
    P = diario["pronosticos"]
    hechos = sorted([e for e in P.values() if "gl" in e], key=lambda e: (e["fecha"], e["comp"]))
    out = {"desde": min((e["primero"][:10] for e in P.values()), default=None),
           "pendientes": sum(1 for e in P.values() if "gl" not in e and not e.get("sin_resultado")),
           "total": _metricas(hechos) if hechos else {"n": 0}, "por_comp": {}}
    for comp in sorted({e["comp"] for e in hechos}):
        out["por_comp"][comp] = _metricas([e for e in hechos if e["comp"] == comp])
    out["ultimos"] = [{k: e.get(k) for k in ("comp", "fecha", "local", "visita", "gl", "gv", "p", "pc", "pk", "c", "xi")}
                      for e in hechos[-60:][::-1]]
    return out
