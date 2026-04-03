cd /d C:\Users\ohiod\Projects\kappa-sig
git add -A
git reset -- _check.bat _push.bat
git commit -m "Complete repo: engine, SIG, Sentinel, data, all domains verified"
git push origin main
del _check.bat 2>nul
del _push.bat 2>nul
