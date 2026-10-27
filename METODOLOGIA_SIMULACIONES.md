# Metodología de las simulaciones Montecarlo

Este documento explica cómo el tablero calcula las probabilidades de victoria, empate y derrota, los marcadores probables y las probabilidades de fin de temporada. Todo lo ejecuta `actualizar_datos.py` en cada actualización; la página solo muestra los resultados.

## 1. Modelo de goles

Para cada liga se estima un modelo **Dixon-Coles** (Dixon y Coles, 1997). Los goles de cada equipo en un partido siguen una distribución de Poisson:

- Goles del local ~ Poisson(λ), con log λ = μ + h + αₗ + δᵥ
- Goles del visitante ~ Poisson(ν), con log ν = μ + αᵥ + δₗ

donde μ es el nivel de goles de la liga, h la ventaja de local, α el ataque de cada equipo y δ su defensa (un valor más alto significa que recibe más goles). Dixon-Coles agrega un factor τ(x, y; ρ) que corrige la probabilidad de los marcadores 0-0, 1-0, 0-1 y 1-1, que el modelo Poisson independiente subestima o sobreestima. El parámetro ρ se estima junto con los demás.

## 2. Estimación

Se maximiza la log-verosimilitud **ponderada y penalizada**:

- **Datos:** todos los partidos de las últimas tres temporadas de football-data.co.uk, más los partidos recientes de ESPN que esa fuente todavía no publica.
- **Ponderación temporal:** cada partido pesa w = exp(−0.005 · días de antigüedad). Un partido de hace 140 días pesa la mitad que uno de hoy, y uno de hace un año, alrededor de 16%.
- **Penalización ridge** (0.6 · Σ α² + δ²): acerca las fuerzas a cero, es decir, al equipo promedio. Evita valores extremos con pocos partidos y resuelve la identificación entre μ y las fuerzas. Los equipos recién ascendidos se acercan a un valor más débil que el promedio.
- **Optimizador:** L-BFGS-B con límites (h en [−1, 1], ρ en [−0.3, 0.3], fuerzas en [−3, 3]).

## 3. Incertidumbre de los parámetros

Los parámetros estimados θ̂ no se conocen con certeza. Para llevar esa incertidumbre a las probabilidades se usa una **aproximación de Laplace**: θ ~ Normal(θ̂, Σ̂), donde Σ̂ es la inversa del Hessiano de la log-verosimilitud penalizada evaluado en θ̂.

- Para la parte Poisson con enlace logarítmico, el Hessiano tiene forma cerrada y exacta: H = Σ wₖ λₖ xₖ xₖ' + Σ wₖ νₖ zₖ zₖ' + 2 · 0.6 · I (sobre las fuerzas), donde x y z son las filas de diseño de cada partido.
- La segunda derivada respecto a ρ se calcula numéricamente (diferencias centrales). Su covarianza con el resto se toma como cero porque τ solo afecta a cuatro marcadores.
- El ridge garantiza que H sea definida positiva. Si no lo fuera por redondeo, se le suma una pequeña constante en la diagonal.

**Interpretación económica:** es el equivalente a usar la matriz de varianzas y covarianzas de un estimador de máxima verosimilitud penalizada, o la aproximación normal a la posterior bayesiana con una previa normal centrada en cero. La covarianza entre parámetros importa: por ejemplo, subir μ y bajar todos los ataques deja los goles esperados iguales, y la simulación respeta esa compensación porque muestrea el vector completo y no cada parámetro por separado.

**Ponderación:** como los partidos antiguos pesan menos, la información efectiva es menor que el número de partidos. Por eso los intervalos son anchos: reflejan que la fuerza actual de un equipo se conoce con pocos partidos recientes.

## 4. Simulación de un partido

Para cada partido próximo:

1. Se sacan **S = 2,000 juegos de parámetros** θₛ ~ Normal(θ̂, Σ̂) mediante la factorización de Cholesky de Σ̂, y se recortan a los límites del modelo.
2. Con cada juego se calculan λₛ y νₛ, y la matriz de marcadores de 0 a 10 goles por lado con la corrección τ, normalizada para sumar 1.
3. De cada matriz se sacan **10 marcadores** por transformada inversa (número uniforme contra la distribución acumulada de los 121 marcadores posibles).
4. En total son **N = 20,000 partidos simulados**. Con semilla fija, los mismos datos producen siempre los mismos números.

Para cada partido se reporta:

| Resultado | Cómo se calcula |
|---|---|
| % gana local / empate / gana visita | Frecuencia en las 20,000 simulaciones |
| ± error | 1.96 × error estándar Montecarlo. Se calcula por grupos: la desviación estándar de la frecuencia dentro de cada juego de parámetros, dividida entre √2,000, porque las 10 simulaciones de un mismo juego no son independientes. Suele ser de ±0.6 a ±0.9 puntos porcentuales |
| Rango 90% | Percentiles 5 y 95 de la probabilidad exacta de cada resultado entre los 2,000 juegos de parámetros. Indica cuánto podría moverse la probabilidad si las fuerzas fueran algo distintas a las estimadas |
| Fórmula | Probabilidad exacta de Dixon-Coles con θ̂, sin incertidumbre (es lo que muestra la calculadora) |
| Marcadores más probables | Los 10 marcadores más frecuentes en las simulaciones, con su cuota justa = 1/probabilidad |
| Goles totales | Histograma de 0 a 7 o más goles, percentiles 5 y 95, más de 2.5 goles y ambos anotan |

**Simulación contra fórmula:** si se fijaran los parámetros en θ̂, la simulación reproduciría la fórmula dentro del error Montecarlo. Al integrar la incertidumbre, la simulación da algo menos de empates y más de los resultados extremos. Esto es una desigualdad de Jensen: mezclar escenarios con un favorito más claro y escenarios más parejos produce más partidos desiguales que el escenario promedio. Las diferencias típicas son de 0.5 a 1.5 puntos porcentuales.

## 5. Simulación de temporada

1. Se toma la tabla actual (puntos, diferencia de goles y goles a favor de la temporada en curso).
2. **Calendario restante:**
   - En las ligas europeas, todos los cruces local-visita que aún no se han jugado; el formato es de ida y vuelta.
   - En Liga MX (fase regular a una vuelta), cada pareja que aún no se ha enfrentado. La localía sale del calendario publicado si ya se conoce; si no, se sortea 50/50 en cada temporada simulada.
3. Se simulan **10,000 temporadas**: 10 por cada uno de los primeros 1,000 juegos de parámetros. Así, cada temporada mantiene un juego de fuerzas coherente para todos sus partidos, lo que genera la correlación realista de que un equipo sobrevalorado lo está en todos sus partidos.
4. **Orden final:** puntos, diferencia de goles, goles a favor y sorteo.
5. Se reportan, por equipo:
   - Puntos finales esperados y su rango del 90%.
   - Probabilidad de quedar 1º.
   - Probabilidad de zona alta: top 4 en Europa; top 6, es decir liguilla directa, en Liga MX.
   - Probabilidad de descenso: últimos 3 en Premier, LaLiga y Serie A; últimos 2 directos en Ligue 1 y Eredivisie; Liga MX no tiene descenso.
   - Posición media y la distribución completa de posiciones.

No se simulan la liguilla, el repechaje ni los desempates oficiales específicos de cada liga, como el enfrentamiento directo.

## 6. Champions y Europa League (modelo propio)

En las copas se enfrentan clubes de ligas distintas, así que las fuerzas domésticas no son comparables directamente. Como ClubElo no siempre responde, el tablero usa un modelo propio:

- **Datos:** todos los partidos de Champions, Europa y Conference League que tiene ESPN desde julio de 2025, incluidas las rondas previas. Se ajustan juntos porque muchos clubes pasan de una copa a otra.
- **Versión 2 (con valor de mercado), la que se usa cuando valida mejor:**
  ataque de copa = ajuste de su liga + δ_A · ataque doméstico + γ_A · log(valor de la plantilla) + desviación propia;
  defensa de copa = ajuste de su liga + δ_D · defensa doméstica − γ_D · log(valor de la plantilla) + desviación propia.
  - El **valor de mercado** de cada plantilla (Transfermarkt, tabla de clubes de unas 35 ligas europeas) es comparable entre ligas, y por eso es la principal referencia para cruces entre ligas distintas. Los clubes sin valor encontrado reciben el percentil 10 (casi siempre son clubes chicos de rondas previas).
  - Las fuerzas domésticas se **centran en la media de su liga** antes de combinarse. δ (prior 1) y γ (prior 0) se estiman con los partidos europeos.
  - Los **ajustes por liga** tienen ridge 1.0; la **desviación propia** de clubes de otras ligas, ridge 2.0.
  - La **localía** tiene un prior de ×1.18 (ridge 5 en escala logarítmica), cercano a la ventaja histórica en copas europeas.
- **Versión 1 (respaldo):** ataque = ataque doméstico + ajuste por liga (+ desviación propia si el club es de otra liga), sin valor de mercado. Se usa si Transfermarkt no responde o si valida mejor.
- **Estimación:** Poisson ponderado (mismo decaimiento temporal) y penalizado, con gradiente analítico. ρ se fija en el promedio de las ligas domésticas.
- **Incertidumbre:** inversa del Hessiano exacto de la parte Poisson. Las fuerzas domésticas se toman como dadas, así que su incertidumbre no se suma.
- **Simulación de la fase de liga** (36 equipos, 8 partidos cada uno): se juega el calendario restante publicado por ESPN. 1-8 pasan a octavos, 9-24 al playoff y 25-36 quedan eliminados.
- **Validación:** el modelo se ajusta solo con la temporada anterior y se pronostican los partidos jugados de la actual. Advertencia: las fuerzas domésticas usadas son las actuales, lo que favorece un poco al modelo.

## 6 bis. Alineación titular (todas las competiciones)

Las fuerzas del modelo reflejan al equipo "típico" de los últimos meses. Para el partido concreto se ajustan con el XI que se espera que juegue:

1. **XI esperado:** el confirmado por ESPN, si ya se publicó (alrededor de 1 hora antes del partido). Si no, el XI titular del último partido del equipo en cualquier competición que siga el tablero. A ese XI se le quitan los lesionados y suspendidos conocidos (Transfermarkt, parte de la Premier y ESPN), y en su lugar entra el jugador disponible de más valor de la misma línea.
2. **Calidad:** el valor de mercado de cada jugador en Transfermarkt (página de plantilla del club, actualizada cada 7 días). El nombre de ESPN se une con el de Transfermarkt por parecido de nombre; si no se encuentra, se usa el valor mediano de la plantilla.
3. **Razones:** r_ataque = valor de medios y delanteros del XI / valor de medios y delanteros del mejor 1-4-3-3 posible de la plantilla. r_defensa es lo mismo con portero y defensas.
4. **Ajuste:** goles esperados del local × r_ataque(local)^β_a × r_defensa(visita)^(−β_d); lo mismo para el visitante. Un XI con menos valor ofensivo anota menos; una defensa con menos valor recibe más.
5. **Estimación de β:** con los partidos ya jugados de esta temporada cuya alineación conocemos (ligas, últimas 3 semanas, y copas europeas), por máxima verosimilitud Poisson con los goles esperados del modelo como offset y ridge 2. Si la alineación no mejora el log-loss, se usa β = 0.1. Advertencia: es una estimación dentro de muestra y con las plantillas actuales.

## 7. Probabilidad en vivo

En la pestaña En vivo, para un partido en juego en el minuto t con marcador a-b, se simulan 10,000 veces los goles restantes como Poisson(λ · (90 − t)/90) para cada equipo, con los goles esperados λ del modelo antes del partido, y se suman al marcador actual. No considera expulsiones, el tiempo añadido ni que el ritmo de goles cambia según el marcador.

## 8. Cómo leer los números

- **"45% ± 0.9"** significa que, según el modelo, el resultado ocurre 45 de cada 100 veces, y que la simulación tiene un error de muestreo de 0.9 puntos. Es un error pequeño: aumentar las simulaciones no cambiaría la conclusión.
- **"Rango 90%: 24%–70%"** no es un error de la simulación. Mide cuánto sabemos realmente de la fuerza de los equipos: con los datos disponibles, la probabilidad verdadera podría estar en cualquier punto de ese rango. Un rango ancho pide cautela.
- **Cuota justa** = 1 / probabilidad. Solo hay valor esperado positivo si la casa paga más que eso, y aun así, la ventaja debe ser mayor que la incertidumbre del modelo.

## 9. Limitaciones

- **Independencia y estructura:** los goles se modelan como Poisson con una corrección solo para marcadores bajos. No hay dinámica dentro del partido (expulsiones, el equipo que va ganando se cierra, etc.).
- **Alineación aproximada:** hasta que ESPN publica el XI confirmado, se supone que repite el XI del último partido sin los lesionados conocidos. Las rotaciones (por ejemplo, entre semana de copa) y las bajas de último momento no se anticipan. El valor de mercado es una aproximación imperfecta de la calidad: no mide el estado de forma ni el encaje táctico.
- **Sin tiros ni xG en la estimación:** se usan goles, que son ruidosos. Los tiros y córners se muestran, pero no entran al modelo.
- **Tamaño de muestra:** al inicio de temporada y con equipos recién ascendidos hay pocos datos, y el modelo depende más de la penalización y de la temporada anterior.
- **Aproximación normal:** la incertidumbre se aproxima con una normal alrededor del óptimo. Con pocos datos, la distribución real podría ser asimétrica.
- **Fuerzas constantes en el resto de la temporada:** en la simulación de temporada no hay cambios de entrenador, fichajes ni lesiones futuras.
- **Las casas de apuestas tienen más información:** alineaciones, mercado y modelos propios. En la prueba fuera de muestra (pestaña Fiabilidad), las cuotas de cierre suelen ser más precisas que este modelo.

## 10. Por qué no garantiza apuestas ganadoras

Una probabilidad del 60% falla 4 de cada 10 veces. Aunque el modelo estuviera bien calibrado, las rachas perdedoras largas son normales. Además, el margen de la casa (5% a 10% en mercados 1X2) hace que la mayoría de las apuestas tengan valor esperado negativo. El valor esperado positivo solo existe cuando la probabilidad del modelo es claramente mayor que la implícita en la cuota, y el rango de incertidumbre muestra que esa diferencia puede ser ilusoria. Úsalo como herramienta de análisis, con montos pequeños y fijos y un registro de cada apuesta para medir si el método realmente funciona.
