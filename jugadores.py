"""
Sección Jugadores: plantillas, estadísticas, lesiones y noticias de cada competición.

Fuentes (gratuitas, sin registro):
  - ESPN (site.api.espn.com): equipos, plantilla con estadísticas de la temporada en esa competición
    (partidos, goles, asistencias, tiros, tarjetas, atajadas) y noticias por equipo con jugadores etiquetados.
  - Fantasy Premier League: estado físico oficial de la Premier (lesionado, duda, suspendido, % de jugar).
  - Transfermarkt: lista de lesionados por liga con tipo de lesión y fecha estimada de regreso.
Cada fuente es opcional: si una falla, el resto sigue funcionando.
"""
import re
import time
import unicodedata
from datetime import datetime, timezone

import requests

ESPN = "https://site.api.espn.com/apis/site/v2/sports/soccer"
COMPETICIONES = {
    # clave de la página: (código ESPN, código Transfermarkt)
    "premier": ("eng.1", "GB1"),
    "laliga": ("esp.1", "ES1"),
    "seriea": ("ita.1", "IT1"),
    "ligue1": ("fra.1", "FR1"),
    "eredivisie": ("ned.1", "NL1"),
    "ligamx": ("mex.1", "MEX1"),
    "ucl": ("uefa.champions", None),
    "uel": ("uefa.europa", None),
}
NAVEGADOR = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                           "(KHTML, like Gecko) Chrome/128.0 Safari/537.36",
             "Accept-Language": "es-MX,es;q=0.9,en;q=0.8"}
HTTP = requests.Session()
HTTP.headers.update(NAVEGADOR)

# estadística de ESPN -> nombre corto en la página
STATS = {"appearances": "pj", "subIns": "sup", "totalGoals": "goles", "goalAssists": "asist",
         "totalShots": "tiros", "shotsOnTarget": "arco", "yellowCards": "amarillas", "redCards": "rojas",
         "foulsCommitted": "faltas", "foulsSuffered": "faltas_rec", "offsides": "fueras",
         "saves": "atajadas", "goalsConceded": "gc", "shotsFaced": "tiros_rec", "ownGoals": "autogoles"}
POSICION = {"G": "POR", "D": "DEF", "M": "MED", "F": "DEL"}


def _json(url, pausa=0.25):
    r = HTTP.get(url, timeout=30)
    r.raise_for_status()
    time.sleep(pausa)
    return r.json()


def normaliza(nombre):
    s = unicodedata.normalize("NFKD", nombre or "").encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z ]", "", s).strip()


def mismo_club(a, b):
    """'Liverpool FC' y 'Liverpool' son el mismo club; evita pegar la lesión a un homónimo de otro equipo."""
    ta = {t for t in normaliza(a).split() if len(t) > 2 and t not in ("club", "real", "city", "united")}
    tb = {t for t in normaliza(b).split() if len(t) > 2 and t not in ("club", "real", "city", "united")}
    return bool(ta & tb) or normaliza(a) in normaliza(b) or normaliza(b) in normaliza(a)


# ----------------------------------------------------------------------------- ESPN
def equipos_espn(codigo):
    d = _json(f"{ESPN}/{codigo}/teams")
    out = []
    for liga in d["sports"][0]["leagues"]:
        for t in liga["teams"]:
            t = t.get("team", t)
            out.append({"id": str(t["id"]), "nombre": t.get("displayName") or t.get("name"),
                        "abrev": t.get("abbreviation", "")})
    return out


def _estadisticas(atleta):
    st = atleta.get("statistics") or {}
    splits = st.get("splits") or {}
    if isinstance(splits, list):
        splits = splits[0] if splits else {}
    out = {}
    for cat in splits.get("categories", []):
        for s in cat.get("stats", []):
            k = STATS.get(s.get("name"))
            if k and s.get("value") is not None:
                out[k] = int(s["value"]) if float(s["value"]).is_integer() else round(float(s["value"]), 2)
    return out


def _lesion_espn(atleta):
    """ESPN rara vez llena `injuries` en fútbol, pero cuando lo hace lo usamos."""
    for inj in atleta.get("injuries") or []:
        tipo = inj.get("type")
        tipo = tipo.get("description") or tipo.get("name") if isinstance(tipo, dict) else tipo
        det = inj.get("details") or {}
        return {"estado": "Lesionado" if str(inj.get("status", "")).lower() in ("out", "injured", "") else "Duda",
                "detalle": det.get("type") or inj.get("shortComment") or tipo or "Lesión",
                "regreso": (det.get("returnDate") or "")[:10] or None, "fuente": "ESPN"}
    status = (atleta.get("status") or {}).get("name", "Active")
    if status and status.lower() not in ("active", "activo"):
        return {"estado": "Baja", "detalle": status, "regreso": None, "fuente": "ESPN"}
    return None


def plantilla_espn(codigo, equipo):
    d = _json(f"{ESPN}/{codigo}/teams/{equipo['id']}/roster")
    atletas = d.get("athletes", [])
    if atletas and isinstance(atletas[0], dict) and "items" in atletas[0]:   # formato agrupado por posición
        atletas = [a for g in atletas for a in g["items"]]
    out = []
    for a in atletas:
        pos = a.get("position") or {}
        j = {"id": str(a.get("id")), "nombre": a.get("displayName") or a.get("fullName"),
             "equipo": equipo["nombre"], "equipo_id": equipo["id"],
             "pos": POSICION.get(pos.get("abbreviation"), pos.get("abbreviation") or ""),
             "edad": a.get("age"), "nac": a.get("citizenship") or (a.get("citizenshipCountry") or {}).get("name"),
             "dorsal": a.get("jersey")}
        j.update(_estadisticas(a))
        les = _lesion_espn(a)
        if les:
            j["lesion"] = les
        out.append(j)
    return out


def noticias_espn(codigo, equipo, limite=12):
    d = _json(f"{ESPN}/{codigo}/news?team={equipo['id']}&limit={limite}")
    out = []
    for a in d.get("articles", []):
        cats = a.get("categories", [])
        atletas = [str(c.get("athleteId") or (c.get("athlete") or {}).get("id")) for c in cats if c.get("type") == "athlete"]
        equipos = [str(c.get("teamId") or (c.get("team") or {}).get("id")) for c in cats if c.get("type") == "team"]
        out.append({"id": str(a.get("id")), "titulo": a.get("headline"), "resumen": a.get("description"),
                    "fecha": (a.get("published") or "")[:10], "url": ((a.get("links") or {}).get("web") or {}).get("href"),
                    "atletas": atletas, "equipos": equipos})
    return out


# ----------------------------------------------------------------------------- lesiones de otras fuentes
def lesiones_fpl():
    """Estado oficial de la Fantasy Premier League: i=lesionado, d=duda, s=suspendido, u/n=no disponible."""
    d = HTTP.get("https://fantasy.premierleague.com/api/bootstrap-static/", timeout=30).json()
    estados = {"i": "Lesionado", "d": "Duda", "s": "Suspendido", "u": "Baja", "n": "Baja"}
    out = {}
    for e in d["elements"]:
        if e.get("status") in estados:
            info = {"estado": estados[e["status"]], "detalle": e.get("news") or "",
                    "regreso": None, "prob": e.get("chance_of_playing_next_round"), "fuente": "Fantasy PL"}
            m = re.search(r"Expected back (\d{1,2} \w{3})", e.get("news") or "")
            if m:
                info["regreso"] = m.group(1)
            for n in {f"{e.get('first_name', '')} {e.get('second_name', '')}", e.get("web_name", "")}:
                out[normaliza(n)] = info
    return out


def lesiones_transfermarkt(codigo):
    """Tabla de lesionados de Transfermarkt. Columnas: jugador/posición, club, lesión, desde, regreso, valor."""
    from lxml import html as lh
    r = HTTP.get(f"https://www.transfermarkt.com/liga/verletztespieler/wettbewerb/{codigo}", timeout=30)
    r.raise_for_status()
    time.sleep(1.0)
    doc = lh.fromstring(r.text)
    out = {}
    for tr in doc.xpath("//table[contains(@class,'items')]/tbody/tr"):
        tds = tr.xpath("./td")
        nombre = tr.xpath(".//td[contains(@class,'hauptlink')]//a/text()")
        if not nombre or len(tds) < 5:
            continue
        textos = [" ".join(t.text_content().split()) for t in tds]
        club = (tr.xpath(".//td//a/img/@title") or tr.xpath(".//td//img/@alt") or [""])[-1]
        # la celda del club solo trae el escudo; las siguientes con texto son lesión, desde y regreso
        resto = [t for t in textos[1:] if t] + ["", "", ""]
        out[normaliza(nombre[0].strip())] = {"estado": "Lesionado", "detalle": resto[0], "desde": resto[1],
                                             "regreso": resto[2] or None, "club": club, "fuente": "Transfermarkt"}
    return out


# ----------------------------------------------------------------------------- todo junto
def recopila(competiciones=None, con_noticias=True, log=print):
    competiciones = competiciones or list(COMPETICIONES)
    lesiones = {}
    try:
        lesiones.update(lesiones_fpl())
        log(f"  Fantasy PL: {len(lesiones)} jugadores con aviso")
    except Exception as e:  # noqa: BLE001
        log("  Fantasy PL no disponible:", e)
    for clave in competiciones:
        tm = COMPETICIONES[clave][1]
        if not tm:
            continue
        try:
            l = lesiones_transfermarkt(tm)
            for k, v in l.items():
                lesiones.setdefault(k, v)       # FPL tiene prioridad en la Premier
            log(f"  Transfermarkt {tm}: {len(l)} lesionados")
        except Exception as e:  # noqa: BLE001
            log(f"  Transfermarkt {tm} no disponible:", e)

    salida = {"generado": datetime.now(timezone.utc).isoformat(timespec="minutes"), "competiciones": {}}
    for clave in competiciones:
        codigo = COMPETICIONES[clave][0]
        try:
            equipos = equipos_espn(codigo)
        except Exception as e:  # noqa: BLE001
            log(f"  ESPN {codigo}: sin equipos ({e})")
            continue
        jugadores, noticias = [], {}
        for eq in equipos:
            try:
                jugadores += plantilla_espn(codigo, eq)
            except Exception as e:  # noqa: BLE001
                log(f"    plantilla {eq['nombre']}: {e}")
            if con_noticias:
                try:
                    for n in noticias_espn(codigo, eq):
                        noticias.setdefault(n["id"], n)
                except Exception as e:  # noqa: BLE001
                    log(f"    noticias {eq['nombre']}: {e}")
        for j in jugadores:
            if "lesion" not in j:
                les = lesiones.get(normaliza(j["nombre"]))
                if les and (not les.get("club") or mismo_club(les["club"], j["equipo"])):
                    j["lesion"] = les
        # noticias más recientes primero, máximo 300 por competición
        lista = sorted(noticias.values(), key=lambda n: n["fecha"], reverse=True)[:300]
        salida["competiciones"][clave] = {"equipos": equipos, "jugadores": jugadores, "noticias": lista}
        log(f"  {clave}: {len(equipos)} equipos, {len(jugadores)} jugadores, "
            f"{sum('lesion' in j for j in jugadores)} con baja, {len(lista)} noticias")
    return salida


if __name__ == "__main__":
    import json
    print(json.dumps(recopila(["premier"]), ensure_ascii=False)[:2000])
