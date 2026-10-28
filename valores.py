"""
Valores de mercado (Transfermarkt) y alineaciones titulares (ESPN).

- Valor de cada club (tabla de la liga en Transfermarkt): sirve para comparar clubes de ligas distintas en las copas.
- Valor de cada jugador (página de plantilla del club): sirve para medir qué tan fuerte es el XI que sale a jugar.
- XI titular: el confirmado por ESPN si ya está publicado (aprox. 1 hora antes); si no, el del último partido del
  equipo, quitando lesionados/suspendidos conocidos y metiendo al siguiente de su posición.

Transfermarkt cambia poco, así que todo se guarda en cache_valores.json y se refresca cada 7 días.
Las alineaciones de partidos jugados no cambian nunca y se guardan para siempre.
"""
import difflib
import json
import os
import re
import time
import unicodedata
from datetime import date, datetime

import requests

CARPETA = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(CARPETA, "cache_valores.json")
DIAS_CACHE = 7
ESPN = "https://site.api.espn.com/apis/site/v2/sports/soccer"
TM = "https://www.transfermarkt.com"
# ligas de Transfermarkt para tener el valor de los clubes que juegan en Europa
LIGAS_TM = ["GB1", "ES1", "IT1", "FR1", "NL1", "MEX1", "L1", "PO1", "BE1", "TR1", "SC1", "A1", "C1", "GR1",
            "DK1", "NO1", "SE1", "TS1", "PL1", "UKR1", "SER1", "KR1", "RO1", "ZYP1", "ISR1", "UNG1", "SLO1",
            "SL1", "BU1", "AZ1", "KAS1", "IR1", "WER1", "FI1", "IS1"]
HTTP = requests.Session()
HTTP.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                                   "(KHTML, like Gecko) Chrome/128.0 Safari/537.36",
                     "Accept-Language": "en-US,en;q=0.9"})
GRUPO_TM = {"Goalkeeper": "POR", "Centre-Back": "DEF", "Left-Back": "DEF", "Right-Back": "DEF", "Defender": "DEF",
            "Sweeper": "DEF", "Defensive Midfield": "MED", "Central Midfield": "MED", "Attacking Midfield": "MED",
            "Left Midfield": "MED", "Right Midfield": "MED", "Midfield": "MED", "Left Winger": "DEL",
            "Right Winger": "DEL", "Centre-Forward": "DEL", "Second Striker": "DEL", "Attack": "DEL"}
FORMA_BASE = {"POR": 1, "DEF": 4, "MED": 3, "DEL": 3}      # XI de referencia: el mejor 1-4-3-3 por valor
GENERICOS = {"fc", "cf", "ac", "sc", "afc", "club", "de", "cd", "ud", "rc", "sd", "ca", "as", "ss", "us", "sk", "fk",
             "bk", "if", "ff", "nk", "hnk", "gnk", "pfc", "sv", "vfb", "vfl", "tsg", "rb", "1", "the", "and", "calcio",
             "football", "futbol", "cp", "sl", "sporting"}


def normaliza(s):
    s = unicodedata.normalize("NFKD", str(s or "")).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9 ]", " ", s).strip()


def _tokens(s):
    return [t for t in normaliza(s).split() if t not in GENERICOS and len(t) > 1]


def parecido(a, b):
    """0 a 1: coincidencia de nombres de club o jugador entre fuentes."""
    ta, tb = set(_tokens(a)), set(_tokens(b))
    if not ta or not tb:
        return difflib.SequenceMatcher(None, normaliza(a), normaliza(b)).ratio()
    comun = sum(1 for t in ta if any(t == u or (len(t) > 3 and (u.startswith(t) or t.startswith(u))) for u in tb))
    return max(comun / max(len(ta), len(tb)), 0.9 * difflib.SequenceMatcher(None, " ".join(sorted(ta)), " ".join(sorted(tb))).ratio())


def valor_eur(txt):
    m = re.search(r"€\s*([\d.,]+)\s*(bn|m|k|Th\.)?", txt or "")
    if not m:
        return None
    v = float(m.group(1).replace(",", ""))
    return v * {"bn": 1e9, "m": 1e6, "k": 1e3, "Th.": 1e3}.get(m.group(2), 1)


# ----------------------------------------------------------------------------- caché
def carga_cache():
    try:
        with open(CACHE, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def guarda_cache(c):
    with open(CACHE, "w", encoding="utf-8") as f:
        json.dump(c, f, ensure_ascii=False)


def _fresco(fecha):
    try:
        return (date.today() - date.fromisoformat(fecha)).days < DIAS_CACHE
    except (TypeError, ValueError):
        return False


_FALLAS_TM = [0]       # fallas seguidas; desde la nube Transfermarkt a veces bloquea y no hay que insistir


def _tm(url):
    if _FALLAS_TM[0] >= 3:
        raise RuntimeError("Transfermarkt bloqueado en esta corrida; se usan los datos guardados")
    for i in range(2):
        try:
            r = HTTP.get(TM + url, timeout=15)
            if r.status_code == 200:
                _FALLAS_TM[0] = 0
                time.sleep(1.2)
                return r.text
        except requests.RequestException:
            pass
        time.sleep(2)
    _FALLAS_TM[0] += 1
    raise RuntimeError(f"Transfermarkt no respondió: {url}")


# ----------------------------------------------------------------------------- Transfermarkt
def clubes_liga(codigo):
    """[(id, nombre, valor total de la plantilla)] de la tabla de clubes de una liga."""
    from lxml import html as lh
    doc = lh.fromstring(_tm(f"/x/startseite/wettbewerb/{codigo}"))
    out, vistos = [], set()
    for tr in doc.xpath("//table[contains(@class,'items')]/tbody/tr"):
        a = tr.xpath(".//td[contains(@class,'hauptlink')]//a[contains(@href,'/verein/')]")
        if not a:
            continue
        m = re.search(r"/verein/(\d+)", a[0].get("href"))
        if not m or m.group(1) in vistos:
            continue
        vistos.add(m.group(1))
        celdas = [" ".join(t.text_content().split()) for t in tr.xpath("./td")]
        out.append((m.group(1), a[0].get("title") or a[0].text_content().strip(), valor_eur(celdas[-1]) if celdas else None))
    return out


def plantilla(tm_id):
    """[(nombre, grupo de posición, valor €)] de la página de plantilla del club."""
    from lxml import html as lh
    doc = lh.fromstring(_tm(f"/x/kader/verein/{tm_id}/saison_id/{date.today().year if date.today().month >= 7 else date.today().year - 1}"))
    out = []
    for tr in doc.xpath("//table[contains(@class,'items')]/tbody/tr[contains(@class,'odd') or contains(@class,'even')]"):
        nombre = tr.xpath(".//td[contains(@class,'hauptlink')]//a/text()")
        if not nombre:
            continue
        pos = tr.xpath(".//table[contains(@class,'inline-table')]//tr[last()]/td/text()")
        valor = tr.xpath(".//td[contains(@class,'rechts') and contains(@class,'hauptlink')]//text()")
        out.append((nombre[0].strip(), GRUPO_TM.get((pos[0] if pos else "").strip(), "MED"),
                    valor_eur(" ".join(valor)) or 0.0))
    return out


def valores_clubes(cache, log=print):
    """{nombre normalizado TM: (id, nombre, valor)} de todas las ligas de LIGAS_TM, con caché semanal.
    Si Transfermarkt no responde, se usan los datos guardados aunque tengan más de 7 días."""
    c = cache.setdefault("ligas", {})
    for cod in LIGAS_TM:
        if cod in c and (_fresco(c[cod]["fecha"]) or _FALLAS_TM[0] >= 3):
            continue
        try:
            c[cod] = {"fecha": date.today().isoformat(), "clubes": clubes_liga(cod)}
        except Exception as e:  # noqa: BLE001
            log(f"    Transfermarkt {cod}: {e}")
    return {cod: v["clubes"] for cod, v in c.items()}


def asigna_club(nombre_espn, candidatos, minimo=0.6):
    """Club de Transfermarkt más parecido al nombre de ESPN entre los candidatos [(id, nombre, valor)]."""
    mejor, puntaje = None, 0
    for cl in candidatos:
        p = parecido(nombre_espn, cl[1])
        if p > puntaje:
            mejor, puntaje = cl, p
    return mejor if puntaje >= minimo else None


def plantillas(cache, tm_ids, log=print):
    c = cache.setdefault("plantillas", {})
    for i in tm_ids:
        if i in c and (_fresco(c[i]["fecha"]) or _FALLAS_TM[0] >= 3):
            continue
        if _FALLAS_TM[0] >= 3:
            continue
        try:
            c[i] = {"fecha": date.today().isoformat(), "jugadores": plantilla(i)}
        except Exception as e:  # noqa: BLE001
            log(f"    plantilla {i}: {e}")
    return {i: c[i]["jugadores"] for i in tm_ids if i in c}


# ----------------------------------------------------------------------------- ESPN: alineaciones
def grupo_espn(abrev):
    """Abreviatura de ESPN (G, CD-L, LB, CM-R, AM, LF, CF…) a línea."""
    a = (abrev or "").upper().split("-")[0]
    if a.startswith("G"):
        return "POR"
    if a in ("D", "CD", "CB", "LB", "RB", "LWB", "RWB", "SW"):
        return "DEF"
    if a in ("F", "CF", "LF", "RF", "LW", "RW", "S", "ST", "FW", "SS"):
        return "DEL"
    return "MED"


def alineaciones(cache, codigo, event_id, completado):
    """{id equipo ESPN: [(nombre, grupo)]} de los titulares. Los partidos jugados se guardan en caché."""
    c = cache.setdefault("alineaciones", {})
    if event_id in c:
        return c[event_id]["xi"]
    try:
        s = HTTP.get(f"{ESPN}/{codigo}/summary?event={event_id}", timeout=30).json()
    except Exception:  # noqa: BLE001
        return {}
    xi = {}
    for r in s.get("rosters", []):
        tit = [p for p in r.get("roster", []) if p.get("starter")]
        if len(tit) >= 10:
            xi[str(r["team"]["id"])] = [((p.get("athlete") or {}).get("displayName", ""), grupo_espn((p.get("position") or {}).get("abbreviation")))
                                        for p in tit]
    odds = (s.get("pickcenter") or [None])[0]
    if completado and xi:
        c[event_id] = {"xi": xi, "odds": odds}
    return xi


# ----------------------------------------------------------------------------- valor de un XI
def mejor_xi(jugadores, fuera=()):
    """El 1-4-3-3 de más valor con los jugadores disponibles; si falta alguien en una línea se completa con el resto."""
    disp = sorted([j for j in jugadores if normaliza(j[0]) not in fuera], key=lambda j: -j[2])
    xi, usados = [], set()
    for g, n in FORMA_BASE.items():
        for j in [j for j in disp if j[1] == g][:n]:
            xi.append(j)
            usados.add(j[0])
    for j in disp:
        if len(xi) >= 11:
            break
        if j[0] not in usados:
            xi.append(j)
            usados.add(j[0])
    return xi


def _busca_jugador(nombre, jugadores):
    mejor, p = None, 0
    for j in jugadores:
        q = parecido(nombre, j[0])
        if q > p:
            mejor, p = j, q
    return mejor if p >= 0.5 else None


def valora_xi(xi_espn, jugadores, fuera=()):
    """Une el XI de ESPN con los valores de Transfermarkt. Los que están en `fuera` (lesionados/suspendidos)
    se reemplazan por el jugador disponible de más valor de su misma línea.
    Devuelve [{nombre, grupo, valor, baja, reemplazo_de}]."""
    mediana = sorted(j[2] for j in jugadores)[len(jugadores) // 2] if jugadores else 1e6
    en_xi = set()
    out = []
    for nombre, g_espn in xi_espn:
        j = _busca_jugador(nombre, jugadores)
        grupo = j[1] if j else g_espn
        if j:
            en_xi.add(j[0])
        out.append({"nombre": nombre, "grupo": grupo, "valor": j[2] if j else mediana, "baja": normaliza(nombre) in fuera
                    or bool(j and normaliza(j[0]) in fuera), "tm": j[0] if j else None})
    final = []
    for p in out:
        if not p["baja"]:
            final.append(p)
            continue
        cand = sorted([j for j in jugadores if j[1] == p["grupo"] and j[0] not in en_xi and normaliza(j[0]) not in fuera],
                      key=lambda j: -j[2]) or sorted([j for j in jugadores if j[0] not in en_xi and normaliza(j[0]) not in fuera], key=lambda j: -j[2])
        if cand:
            en_xi.add(cand[0][0])
            final.append({"nombre": cand[0][0], "grupo": cand[0][1], "valor": cand[0][2], "baja": False, "reemplazo_de": p["nombre"]})
        else:
            final.append({**p, "baja": False})
    return final


def lineas(xi):
    """(valor ofensivo, valor defensivo): medios + delanteros, y portero + defensas."""
    of = sum(p["valor"] for p in xi if p["grupo"] in ("MED", "DEL"))
    de = sum(p["valor"] for p in xi if p["grupo"] in ("POR", "DEF"))
    return max(of, 1e5), max(de, 1e5)
