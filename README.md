# StockPredictionProject

StockPredictionProject to kompletna aplikacja do budowy prognoz cen akcji w trybie „end-to-end”: od pobrania danych rynkowych, przez inżynierię cech (z użyciem R), trening modeli (PyTorch LSTM + XGBoost), aż po generowanie predykcji i artefaktów (CSV + wykres). Projekt jest przygotowany pod uruchomienie w Dockerze z GPU (CUDA 12.1) i udostępnia dwa interfejsy: API (FastAPI) oraz UI (Streamlit). Cały przepływ można wykonać jednym requestem (`/pipeline`) albo krokowo (`/fetch`, `/features`, `/train`, `/predict`).

Ważna cecha projektu: logika pipeline jest spójna między uruchomieniem przez API, UI i CLI. API ustawia kontekst (ticker, interwał i tryb force) per-request, a poszczególne moduły czytają te wartości i pracują na wspólnych katalogach roboczych mapowanych jako wolumeny Dockera. Dzięki temu artefakty (modele, scalery, dane, outputy) zostają na hoście i nie giną po restarcie kontenerów.

## Co robi pipeline i dlaczego tak jest zorganizowany

Pipeline składa się z czterech logicznych etapów. Najpierw pobierane są dane notowań (Yahoo Finance), a wynik jest czyszczony do stabilnej struktury kolumn (`Date, Open, High, Low, Close, Volume`) i zapisywany do cache w formacie CSV. Następnie uruchamiany jest etap budowy cech, realizowany przez skrypt R (`src/r/analyze.R`). Ten krok odpowiada za feature engineering (np. wskaźniki techniczne, transformacje czasowe, przygotowanie zbioru cech) i produkuje plik cech w katalogu `FEATURES_DIR`. Potem uruchamiany jest trening dwóch modeli: LSTM (dla sekwencji czasowych) oraz XGBoost (dla predykcji tablicowej). Na końcu wykonywana jest predykcja i generowane są artefakty: `predictions.csv`, `forecast.csv` oraz plik PNG z wykresem.

Wynik końcowy jest hybrydą, która łączy predykcje DL (LSTM) i XGB, a opcjonalnie także wkład z R. W praktyce oznacza to, że aplikacja może produkować zarówno osobne serie (Close/LSTM/XGB), jak i serię hybrydową. UI korzysta z endpointów API, które zwracają gotowe serie do wykresu i metryki „latest”.

## Uruchomienie w Dockerze

Projekt jest przygotowany do pracy w Docker Compose. Najprostszy sposób uruchomienia to build i start całego stosu:

```
docker compose up --build
```
Po starcie dostępne są:

API (FastAPI): http://localhost:8000/docs

UI (Streamlit): http://localhost:8501

Jeżeli uruchamiasz to na Windows, pamiętaj, że build context może być wrażliwy na symlinki/reparse points. Jeżeli pojawią się błędy typu „invalid file request config”, sprawdź, czy w katalogu projektu nie ma obiektów ReparsePoint (symlink/junction) i dodaj problematyczne ścieżki do .dockerignore.

Jak działa konfiguracja środowiska i ścieżek

W projekcie ścieżki katalogów roboczych są zdefiniowane centralnie w src/config/paths.py. Docker Compose mapuje katalogi hosta na katalogi w kontenerze (wolumeny), aby:

nie tracić danych po restarcie kontenera,

łatwo podglądać artefakty na hoście,

rozdzielić dane od obrazu.

W typowej konfiguracji:

./data na hoście → /data w kontenerze (cache, dane przetworzone)

./models → /models (modele, scalery, schema cech)

./output → /output (wyniki, wykresy, logi błędów pipeline)

API ustawia zmienne środowiskowe per request (ticker, interwał, force). UI z kolei komunikuje się z API wewnątrz sieci Dockerowej. W UI są dwie istotne zmienne:

API_BASE – adres API widoczny z kontenera UI (zwykle http://api:8000)

PUBLIC_API_BASE – adres używany do linków pobrań w przeglądarce na hoście (zwykle http://localhost:8000)

Dzięki temu UI może działać w kontenerze, ale linki do pobierania CSV/PNG działają poprawnie z perspektywy użytkownika.

Korzystanie z API (krokowo i end-to-end)

API udostępnia prosty kontrakt: przekazujesz ticker, interval oraz force, a kolejne endpointy uruchamiają odpowiedni etap pipeline. force=true wymusza odświeżenie danych/cech/modeli (w zależności od kroku), co jest przydatne po zmianach w logice lub przy debugowaniu.

Przykład wywołania pełnego pipeline:

curl -X POST http://localhost:8000/pipeline \
  -H "Content-Type: application/json" \
  -d "{\"ticker\":\"AAPL\",\"interval\":\"1d\",\"force\":true}"


Jeżeli chcesz wykonywać kroki ręcznie (np. do debugowania R albo treningu), użyj kolejno:

/fetch – pobranie i zapis cache

/features – uruchomienie R i zapis pliku cech

/train – trening modeli i zapis artefaktów w MODEL_DIR

/predict – wygenerowanie predykcji i zapis plików wynikowych w OUTPUT_DIR

Endpointy wynikowe są używane głównie przez UI:

/results/latest – zwraca ostatni punkt i linki do artefaktów (CSV/PNG)

/results/series – zwraca serię punktów do wykresu

/artifacts – lista plików dla tickera (modele/scalery/outputy)

/download/output/{filename} i /download/model/{filename} – pobieranie plików z kontenera

UI (Streamlit): sterowanie pipeline i wizualizacja

UI jest cienką warstwą na API. Pozwala wybrać ticker i interwał, uruchamiać poszczególne kroki oraz wyświetlać:

wykres serii (Close, LSTM, XGB oraz Hybrid, jeśli endpoint ją zwraca),

metryki „Latest” (ostatnia data, close, LSTM, XGB),

listę artefaktów zapisanych na dysku,

podgląd wygenerowanego wykresu PNG bezpośrednio w przeglądarce.

Jeżeli podgląd obrazu nie działa, najczęstszą przyczyną jest niewłaściwy parametr width w st.image. Poprawny sposób na „rozciągnięcie” obrazu w Streamlit to use_container_width=True.

Artefakty i gdzie ich szukać

Po udanym uruchomieniu predykcji w katalogu output na hoście pojawiają się pliki:

{TICKER}_predictions.csv – serie do UI i analizy (Close/LSTM/XGB/Hybrid oraz aliasy)

{TICKER}_forecast.csv – alternatywny zapis (zależnie od implementacji)

{TICKER}_predictions_plot.png – wykres wygenerowany w pipeline

{TICKER}_pipeline_error.log oraz {TICKER}_predict_error.log – pełne trace-backi, gdy coś pójdzie źle

To jest celowe: debug w projektach z Dockerem jest dużo szybszy, gdy log błędu zostaje na hoście i nie wymaga ręcznego „wchodzenia” do kontenera.

Debug i typowe klasy błędów

Jeżeli endpoint zwraca 500, pierwszym krokiem powinno być sprawdzenie logów w ./output. Projekt zapisuje pełne tracebacki do plików, więc nie musisz polegać wyłącznie na skróconej odpowiedzi HTTP. Bardzo częsty przypadek przy integracji z R to brak pakietów w kontenerze (there is no package called ...). Wtedy trzeba dopisać instalację pakietów R do Dockerfile (lub wykonać ją w warstwie obrazu). Drugi częsty problem to niespójność nazw plików cech (legacy AAPL_features.csv vs docelowe AAPL_1d_features.csv). Projekt wspiera fallbacki, ale najlepiej docelowo ujednolicić format plików we wszystkich modułach.

Jeżeli wykresy wyglądają „nienaturalnie” (np. spadki do zera), najczęstszą przyczyną jest agresywne fillna(0) w predykcji lub merge z R, które produkują wiersze bez DL/XGB i potem są zerowane. W takim przypadku lepszym podejściem jest forward-fill dla serii modelowych i zerowanie tylko serii pomocniczych (np. R_Forecast) – tak, aby wykres nie zawierał sztucznych zer.

Uruchomienie bez Dockera (development)

Możesz uruchomić projekt lokalnie, ale wymagane są zależności systemowe oraz działające R (Rscript) z pakietami. W praktyce Docker jest rekomendowany, bo stabilizuje środowisko.

API:

python3.11 -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000


UI:

streamlit run src/ui/app.py --server.address=0.0.0.0 --server.port=8501

Status projektu i dalsze kierunki

Projekt działa jako pipeline E2E z powtarzalnymi artefaktami na wolumenach i z rozdzieleniem API/UI. Naturalne następne kroki to: dopracowanie walidacji wejść, zawężenie CORS, wersjonowanie modeli per ticker/interwał, oraz lepsza obserwowalność (log levels, structured logs, metryki). W zakresie predykcji warto rozważyć ujednolicenie kontraktu kolumn (DL_Forecast/XGB_Forecast/Hybrid_Forecast vs aliasy do UI) oraz stabilne formaty zapisu danych R (np. zawsze JSON do stdout albo zawsze CSV do FORECASTS_DIR).

