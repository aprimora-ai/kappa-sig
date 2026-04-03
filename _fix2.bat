cd /d C:\Users\ohiod\Projects\kappa-sig
git add -A
git reset -- _fix.bat
git commit -m "update: fresh experiment results from verified runs"
git push origin main
del _fix.bat
