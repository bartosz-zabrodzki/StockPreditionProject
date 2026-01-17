Write-Host "Installing R packages into venv_R_libs..."
$RPath = "C:\Program Files\R\R-4.5.2\bin\Rscript.exe"
$LibPath = (Resolve-Path "venv_R_libs").Path

& $RPath -e "dir.create('$LibPath', showWarnings=FALSE)"
& $RPath -e "install.packages(c('forecast','tseries','TTR','dplyr','quantmod'), lib='$LibPath', repos='https://cloud.r-project.org')"

Write-Host "All R packages installed locally."
