# Tablero de Probabilidades

Página personal para ver estadísticas de partidos y estimar probabilidades en Liga MX, LaLiga, Premier League, Ligue 1, Serie A, Eredivisie, Champions League y Europa League, y compararlas con las cuotas de tu casa de apuestas.

## Cómo usarla

```bash
pip install pandas numpy scipy requests lxml
python actualizar_datos.py            # 4 a 6 minutos con jugadores, escribe index.html
python actualizar_datos.py --sin-jugadores   # solo pronósticos, ~1 minuto
```

Abre `index.html` con doble clic. No necesita servidor. Corre el script los martes y viernes (cuando football-data.co.uk actualiza) o prográmalo con el Programador de tareas de Windows / cron.

Archivos:

| Archivo | Qué hace |
|---|---|
| `actualizar_datos.py` | Descarga datos, ajusta el modelo, hace el backtest y genera `index.html` |
| `jugadores.py` | Plantillas, estadísticas por jugador, lesiones y noticias |
| `plantilla.html` | Diseño de la página (HTML/CSS/JS); el script le inserta los datos |
| `index.html` | La página lista para abrir |
| `.github/workflows/actualizar.yml` | Actualización automática en GitHub cada 2 horas y publicación en GitHub Pages |

### Actualización automática en GitHub

El flujo `.github/workflows/actualizar.yml` corre `actualizar_datos.py` en la nube cada 2 horas (y con cada `git push`) y publica la página en `https://<usuario>.github.io/<repositorio>/`. La página se recarga sola cada 30 minutos.

1. Sube esta carpeta a un repositorio de GitHub (rama `main`).
2. En el repositorio: **Settings > Pages > Build and deployment > Source = GitHub Actions**.
3. En **Actions > Actualizar tablero > Run workflow** se puede forzar una actualización.

En GitHub gratis, Pages solo funciona con repositorios públicos: cualquiera con el enlace puede ver la página. Transfermarkt a veces bloquea los servidores de GitHub; si pasa, la página sale igual, pero sin esas lesiones.

## Diseño

**Fuentes de datos (gratuitas)**

| Competición | Fuente | Qué trae |
|---|---|---|
| 5 ligas europeas | football-data.co.uk (`mmz4281/2627/E0.csv`, `SP1`, `I1`, `F1`, `N1`) | Resultados, medio tiempo, tiros, tiros a puerta, córners, tarjetas y cuotas (Pinnacle de cierre, promedio de mercado) |
| Próximos partidos | football-data.co.uk `fixtures.csv` | Calendario de la semana con cuotas |
| Liga MX | football-data.co.uk `new/MEX.csv` | Resultados y cuotas (sin tiros ni córners) |
| Complemento al momento (6 ligas) | ESPN `.../{liga}/scoreboard?dates=AAAAMMDD` y `.../summary?event={id}` | Partidos que football-data aún no publica (marcador, tiros, a puerta, córners, tarjetas, cuotas DraftKings) y calendario de los próximos 7 días. Los nombres de equipo se emparejan con los de football-data usando partidos en común |
| Champions / Europa | ClubElo (`api.clubelo.com/AAAA-MM-DD`) | Rating Elo de todos los clubes europeos |
| Respaldo europeo | openfootball en GitHub | Resultados y calendario, sin cuotas ni estadísticas |

"League 1" se interpretó como la Ligue 1 francesa. Si te referías a la EFL League One inglesa, descomenta la línea `leagueone` en `LIGAS`.

**Jugadores** (pestaña Jugadores en cada competición)

| Dato | Fuente |
|---|---|
| Plantillas y estadísticas de la temporada en esa competición (PJ, goles, asistencias, tiros, a puerta, tarjetas, atajadas) | ESPN `site.api.espn.com/apis/site/v2/sports/soccer/{liga}/teams/{id}/roster` |
| Noticias por equipo, con jugadores etiquetados | ESPN `.../{liga}/news?team={id}` |
| Lesiones con tipo y fecha estimada de regreso | Transfermarkt (página de lesionados de cada liga) |
| Parte médico oficial de la Premier (lesionado, duda, % de jugar) | Fantasy Premier League `bootstrap-static` |

Las bajas también aparecen bajo cada partido en Próximos. En Champions y Europa las estadísticas son solo de ese torneo; sus lesiones se cruzan con las de las ligas domésticas descargadas.

**Modelo**

- Ligas: Dixon-Coles. Cada equipo tiene una fuerza de ataque y una de defensa; los goles de cada lado siguen una Poisson con corrección para marcadores bajos (0-0, 1-0, 0-1, 1-1). Los partidos pesan menos conforme envejecen (vida media ≈ 140 días) y los ascendidos arrancan con un prior más débil. Se usan tres temporadas.
- Copas: el Elo de ClubElo da la expectativa de victoria (ventaja local de 65 puntos); se convierte a goles esperados con un total de 2.8 por partido para obtener todos los mercados.
- De la matriz de marcadores salen 1X2, doble oportunidad, más/menos 1.5/2.5/3.5, ambos anotan, hándicap −1.5 y marcador exacto.

**Valor esperado y banca**

- EV = probabilidad × cuota − 1. Solo vale la pena apostar con EV mayor a tu mínimo (por defecto 5%), para dejar margen al error del modelo.
- Monto = banca × fracción de Kelly (por defecto ¼) × (p·cuota − 1)/(cuota − 1), con tope de 3% de la banca por apuesta.

**Fiabilidad**

El script reajusta el modelo semana a semana solo con información previa y pronostica la semana siguiente (temporada pasada completa + la actual). Reporta log-loss y Brier contra una referencia de frecuencias históricas, una gráfica de calibración y, cuando hay cuotas, la comparación contra el mercado y el rendimiento de las apuestas con EV > 5%.

Resultado con openfootball (oct 2026, sin cuotas): el modelo mejora entre 5% y 9% el log-loss frente a frecuencias históricas en las cinco ligas y está bien calibrado entre 20% y 80%. Todavía no está probado contra el mercado; eso se ve al correr el script con football-data.co.uk.

## Advertencia

Ninguna apuesta es segura. Las cuotas de cierre de Pinnacle son muy difíciles de superar; un modelo público como este acierta la dirección pero rara vez le gana al mercado de forma sostenida. Úsalo para filtrar partidos, apuesta montos pequeños y registra cada apuesta para medir si tu rendimiento real es positivo.
