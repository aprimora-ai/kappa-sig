@echo off
setlocal enabledelayedexpansion
REM ===================================================================
REM  Kappa Sentinel -- Full Pipeline + v2 Monitoring + Analyst + Deploy
REM  David Ohio | odavidohio@gmail.com
REM  Run daily or on-demand to update katashi.digital
REM  v5.6 -- v2 monitoring + cockpit + impact sensitivity
REM ===================================================================
echo.
echo ================================================================
echo   KAPPA SENTINEL -- Full Pipeline Run (v5.6)
echo   %date% %time%
echo ================================================================
echo.

cd /d C:\Users\ohiod\Projects\Sentinel

REM Step 1: Kappa-FIN analysis on all universes (engine_v4)
echo [1/14] Running Kappa pipeline (21 universes)...
python src/kappa/pipeline.py
if %errorlevel% neq 0 (
    echo ERROR: Pipeline failed!
    exit /b 1
)

REM Step 2: Kappa v2 Monitoring (engine_v5.6 -- prospective tracking)
echo [2/14] Running Kappa v2 monitoring (Layers 1-3)...
python run_v2_monitor.py
if %errorlevel% neq 0 (
    echo WARNING: v2 monitoring had errors (non-fatal, continuing)
)

REM Step 3: Generate timelines
echo [3/14] Generating timelines...
python src/kappa/timeline_gen.py

REM Step 4: OSINT enrichment (RSS feeds)
echo [4/14] OSINT enrichment (RSS only, GDELT via daemon)...
python src/kappa/osint_enrich.py --no-gdelt

REM Step 5: Alpha Vantage market data
echo [5/14] Alpha Vantage enrichment...
python src/kappa/alpha_vantage.py 2>nul
if %errorlevel% neq 0 (
    echo   Alpha Vantage skipped (API error or rate limit)
)

REM Step 6: Cockpit enrichment (v1)
echo [6/14] Cockpit enrichment (v1)...
python src/kappa/cockpit_enrich.py

REM Step 7: Cockpit enrichment (v2 -- Layers 1-3 + prospective)
echo [7/14] Cockpit enrichment (v2)...
python cockpit_v2_enrich.py
if %errorlevel% neq 0 (
    echo WARNING: v2 cockpit enrichment had errors (non-fatal)
)

REM Step 8: Impact Sensitivity Index (Section 8.12)
echo [8/14] Computing Impact Sensitivity Index...
python compute_impact_sensitivity.py
if %errorlevel% neq 0 (
    echo WARNING: Impact sensitivity had errors (non-fatal)
)

REM Sync summary to dashboard
copy /Y data\reports\sentinel_summary.json dashboard\public\sentinel_summary.json >nul

REM Step 9: Structural interpretation (rule-based insights)
echo [9/14] Structural interpretation...
python src/kappa/structural_interpreter.py

REM Step 10: Inbox snapshot (for the Analyst)
echo [10/14] Generating inbox snapshot...
python generate_inbox_snapshot.py

REM Step 11: Analyst -- daemon or lifecycle
if exist data\analyst\daemon.pid (
    for /f %%p in (data\analyst\daemon.pid) do (
        tasklist /FI "PID eq %%p" /NH 2>nul | find "%%p" >nul
        if !errorlevel! equ 0 (
            echo [11/14] Analista: daemon VIVO PID %%p -- snapshot processado automaticamente
        ) else (
            echo [11/14] Analista: daemon OFF, rodando lifecycle...
            python -m src.kappa.analyst.lifecycle wake 2>nul
            if !errorlevel! neq 0 echo   Analista skipped (Ollama not running or error)
        )
    )
) else (
    echo [11/14] Analista: sem daemon, rodando lifecycle...
    python -m src.kappa.analyst.lifecycle wake 2>nul
    if !errorlevel! neq 0 echo   Analista skipped (Ollama not running or error)
)

REM Step 12: Audit trail
echo [12/14] Generating audit trail...
python generate_audit_trail.py

REM Step 13: Archive pipeline snapshot for history
echo [13/14] Archiving pipeline snapshot...
python archive_pipeline.py

REM Step 14: Build dashboard + sync all JSONs + history
echo [14/14] Building dashboard...
cd dashboard
call node node_modules\vite\bin\vite.js build
copy /Y public\sentinel_summary.json dist\ >nul
copy /Y public\sentinel_cockpit.json dist\ >nul
copy /Y public\sentinel_v2_cockpit.json dist\ >nul 2>nul
copy /Y public\sentinel_timelines.json dist\ >nul
copy /Y public\analyst_briefings.json dist\ >nul 2>nul
copy /Y ..\data\reports\audit_trail.json dist\ >nul 2>nul
xcopy public\history dist\history /E /I /Y >nul 2>nul
cd ..

echo.
echo ================================================================
echo   Pipeline complete! Dashboard built in dashboard\dist\
echo   v2 cockpit: dashboard\public\sentinel_v2_cockpit.json
echo   Impact Sensitivity: data\v2_monitoring\impact_sensitivity.json
echo   Prospective: data\v2_monitoring\prospective_status.json
echo   Deploy: copy dist\ to katashi.digital hosting
echo ================================================================
echo.
