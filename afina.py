"""
Afina por liga la velocidad de olvido (XI) y el suavizado (RIDGE) del modelo y los guarda en parametros.json.

Para no engañarnos: los parámetros se eligen con los pronósticos de la temporada pasada y después se comprueba
si también mejoran los de la temporada actual, que no se usaron para elegir. Solo se guardan si no empeoran ahí.
Tarda (decenas de minutos): correrlo de vez en cuando, por ejemplo una vez al mes.

Uso:  python afina.py            (todas las ligas)
      python afina.py premier    (una sola)
"""
import json
import os
import sys

import numpy as np
import pandas as pd

import actualizar_datos as A

XIS = [0.0025, 0.004, 0.005, 0.0065, 0.008]     # vida media ≈ 280, 175, 140, 105 y 85 días
RIDGES = [0.3, 0.6, 1.0]
RUTA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "parametros.json")


def datos_liga(clave):
    nombre, cod_fd, _ = A.LIGAS[clave]
    jug, fut = A.carga_football_data(clave, cod_fd)
    if cod_fd != "MEX":
        fut = A.proximos_football_data(cod_fd)
    try:
        extra, _ = A.complemento_espn(clave, jug, fut)
        if len(extra):
            jug = pd.concat([jug, extra], ignore_index=True)
    except Exception as e:  # noqa: BLE001
        print("  complemento ESPN falló:", e)
    fechas = sorted(set(jug["fecha"]))
    hueco = 17 if cod_fd == "MEX" else 30
    inicios = [fechas[0]] + [fechas[i + 1] for i in range(len(fechas) - 1) if (fechas[i + 1] - fechas[i]).days > hueco]
    xg = A.xg_aproximado(jug)
    return (xg[0] if xg else jug), inicios, xg is not None


def evalua(jug, inicios, con_xg, xi, ridge):
    """Log-loss de la temporada pasada y de la actual con esos parámetros."""
    preds, _, w, _ = A.backtest(jug, inicios, con_xg=con_xg, xi=xi, ridge=ridge)
    corte = inicios[-1].isoformat()
    out = {}
    for nombre, sel in (("pasada", [p for p in preds if p["fecha"] < corte]), ("actual", [p for p in preds if p["fecha"] >= corte])):
        res = np.array([0 if p["gl"] > p["gv"] else 1 if p["gl"] == p["gv"] else 2 for p in sel])
        out[nombre] = A._logloss([p["p"] for p in sel], res) if sel else float("nan")
        out["n_" + nombre] = len(sel)
    out["peso_goles"] = w
    return out


def afina(clave):
    print(f"== {A.LIGAS[clave][0]}")
    jug, inicios, con_xg = datos_liga(clave)
    cache = {}

    def prueba(xi, ridge):
        if (xi, ridge) not in cache:
            cache[(xi, ridge)] = evalua(jug, inicios, con_xg, xi, ridge)
            r = cache[(xi, ridge)]
            print(f"  XI {xi:<7} ridge {ridge:<4} pasada {r['pasada']:.4f} ({r['n_pasada']})  actual {r['actual']:.4f} ({r['n_actual']})")
        return cache[(xi, ridge)]

    base = prueba(A.XI, A.RIDGE)
    mejor_xi = min(XIS, key=lambda x: prueba(x, A.RIDGE)["pasada"])          # primero la velocidad de olvido
    mejor_r = min(RIDGES, key=lambda r: prueba(mejor_xi, r)["pasada"])       # luego el suavizado
    elegido = prueba(mejor_xi, mejor_r)
    queda = elegido["actual"] < base["actual"]                                  # solo si mejora la temporada actual
    print(f"  -> XI {mejor_xi}, ridge {mejor_r}: temporada actual {base['actual']:.4f} -> {elegido['actual']:.4f}"
          f" {'(se guarda)' if queda else '(no mejora fuera de la elección: se dejan los generales)'}")
    return ({"xi": mejor_xi, "ridge": mejor_r} if queda else None), base, elegido


def main():
    ligas = sys.argv[1:] or list(A.LIGAS)
    try:
        with open(RUTA, encoding="utf-8") as f:
            params = json.load(f)
    except (OSError, ValueError):
        params = {}
    resumen = {}
    for clave in ligas:
        try:
            p, base, elegido = afina(clave)
        except Exception as e:  # noqa: BLE001
            print(f"  {clave}: error {e}")
            continue
        if p:
            params[clave] = p
        else:
            params.pop(clave, None)
        resumen[clave] = {"antes": round(base["actual"], 4), "despues": round((elegido if p else base)["actual"], 4), **(p or {})}
        with open(RUTA, "w", encoding="utf-8") as f:            # se guarda liga por liga por si se interrumpe
            json.dump(params, f, ensure_ascii=False, indent=1)
    print(json.dumps(resumen, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
