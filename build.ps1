param(
    [string]$Python = "python",
    [string]$PdfLaTeX = "pdflatex"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$RepositoryRoot = $PSScriptRoot

Push-Location -LiteralPath $RepositoryRoot
try {
    & $Python "code/simulate.py"
    if ($LASTEXITCODE -ne 0) { throw "Simulation failed." }

    & $Python "code/model_i_stress.py"
    if ($LASTEXITCODE -ne 0) { throw "Model-I stress test failed." }

    & $Python "code/adaptive_network_stress.py"
    if ($LASTEXITCODE -ne 0) { throw "Adaptive-network stress test failed." }

    & $Python "code/design_power.py"
    if ($LASTEXITCODE -ne 0) { throw "Design and power simulation failed." }

    & $Python "code/prepare_jcn_submission.py"
    if ($LASTEXITCODE -ne 0) { throw "JCN submission preparation failed." }

    Push-Location -LiteralPath "submission-jcn"
    try {
        for ($Pass = 1; $Pass -le 3; $Pass++) {
            & $PdfLaTeX "-interaction=nonstopmode" "-halt-on-error" "-file-line-error" "submission.tex"
            if ($LASTEXITCODE -ne 0) { throw "JCN LaTeX compilation failed on pass $Pass." }
        }
    }
    finally {
        Pop-Location
    }

    & $Python "code/build_release_manifest.py"
    if ($LASTEXITCODE -ne 0) { throw "Release-manifest generation failed." }

    & $Python "code/verify_archive.py"
    if ($LASTEXITCODE -ne 0) { throw "Archive and release-manifest verification failed." }
}
finally {
    Pop-Location
}
