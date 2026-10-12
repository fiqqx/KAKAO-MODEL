[CmdletBinding()]
param(
    [string]$DatasetPath,
    [ValidateRange(1, 100000)]
    [int]$Epochs,
    [ValidateRange(2, 100000)]
    [int]$BatchSize,
    [ValidateSet('cu126', 'cu128', 'cpu')]
    [string]$TorchIndex = 'cu126',
    [string]$PredictionImage,
    [string]$Resume
)

# Compatible with Windows PowerShell 5.1; no environment activation is needed.
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$script:FailureExitCode = 1
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PythonPath = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$StartingDirectory = (Get-Location).Path

function Get-AbsoluteInputPath {
    param([string]$Value)
    if ([System.IO.Path]::IsPathRooted($Value)) {
        return [System.IO.Path]::GetFullPath($Value)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $StartingDirectory $Value))
}

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)][string]$Executable,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )
    # Inspect the native exit code explicitly, including on PowerShell 5.1.
    $previousPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        $global:LASTEXITCODE = $null
        & $Executable @Arguments
        $nativeExitCode = $LASTEXITCODE
        if ($null -eq $nativeExitCode) {
            $nativeExitCode = 1
        }
    }
    finally {
        $ErrorActionPreference = $previousPreference
    }
    if ($nativeExitCode -ne 0) {
        $script:FailureExitCode = $nativeExitCode
        throw "Perintah gagal (exit $nativeExitCode): $Executable $($Arguments -join ' ')"
    }
}

function Test-Python312 {
    param([string]$Executable, [string[]]$PrefixArguments = @())
    $previousPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        $global:LASTEXITCODE = $null
        & $Executable @PrefixArguments -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) and sys.maxsize > 2**32 else 1)' 2>$null | Out-Null
        return ($LASTEXITCODE -eq 0)
    }
    finally {
        $ErrorActionPreference = $previousPreference
    }
}

try {
    if ([string]::IsNullOrWhiteSpace($DatasetPath)) {
        $DatasetPath = Join-Path $ProjectRoot 'dataset'
    }
    else {
        $DatasetPath = Get-AbsoluteInputPath $DatasetPath
    }
    if (-not (Test-Path -LiteralPath $DatasetPath -PathType Container)) {
        throw "Folder dataset tidak ditemukan: $DatasetPath. Ekstrak dataset COCO dan gunakan -DatasetPath untuk menentukan foldernya."
    }
    if (-not [string]::IsNullOrWhiteSpace($PredictionImage)) {
        $PredictionImage = Get-AbsoluteInputPath $PredictionImage
        if (-not (Test-Path -LiteralPath $PredictionImage -PathType Leaf)) {
            throw "Gambar prediksi tidak ditemukan: $PredictionImage"
        }
    }
    if (-not [string]::IsNullOrWhiteSpace($Resume)) {
        $Resume = Get-AbsoluteInputPath $Resume
        if (-not (Test-Path -LiteralPath $Resume -PathType Leaf)) {
            throw "Checkpoint untuk resume tidak ditemukan: $Resume"
        }
    }

    Push-Location -LiteralPath $ProjectRoot
    try {
        Write-Host '[1/5] Menyiapkan Python 3.12 64-bit dan virtual environment.'
        if (Test-Path -LiteralPath $PythonPath -PathType Leaf) {
            if (-not (Test-Python312 -Executable $PythonPath)) {
                throw '.venv yang ada bukan Python 3.12 64-bit. Simpan environment lama dengan mengganti nama folder .venv, lalu jalankan lagi. Launcher tidak menghapus environment Anda.'
            }
        }
        else {
            # Preserve an existing, incomplete environment rather than replace it.
            if (Test-Path -LiteralPath (Join-Path $ProjectRoot '.venv')) {
                throw 'Folder .venv sudah ada tetapi python.exe tidak ditemukan. Ganti nama folder tersebut setelah memeriksanya, lalu jalankan lagi.'
            }
            $launcher = Get-Command 'py.exe' -ErrorAction SilentlyContinue
            $systemPython = Get-Command 'python.exe' -ErrorAction SilentlyContinue
            if ($null -ne $launcher -and (Test-Python312 -Executable $launcher.Source -PrefixArguments @('-3.12'))) {
                Invoke-Checked -Executable $launcher.Source -Arguments @('-3.12', '-m', 'venv', (Join-Path $ProjectRoot '.venv'))
            }
            elseif ($null -ne $systemPython -and (Test-Python312 -Executable $systemPython.Source)) {
                Invoke-Checked -Executable $systemPython.Source -Arguments @('-m', 'venv', (Join-Path $ProjectRoot '.venv'))
            }
            else {
                throw 'Python 3.12 64-bit belum ditemukan. Instal Python 3.12 dari python.org (termasuk Python Launcher), buka ulang terminal VS Code, lalu jalankan kembali.'
            }
        }

        if ($TorchIndex -ne 'cpu') {
            Write-Host "Build GPU dipilih: $TorchIndex. Gunakan cu128 untuk RTX seri 50 / Blackwell; pastikan driver NVIDIA mendukung runtime tersebut."
            $nvidiaSmi = Get-Command 'nvidia-smi.exe' -ErrorAction SilentlyContinue
            if ($null -ne $nvidiaSmi) {
                Invoke-Checked -Executable $nvidiaSmi.Source -Arguments @('--query-gpu=name,driver_version', '--format=csv,noheader')
            }
            else {
                Write-Warning 'nvidia-smi tidak ditemukan pada PATH. Pemeriksaan CUDA berikutnya tetap harus lulus; periksa driver NVIDIA jika gagal.'
            }
        }

        Write-Host '[2/5] Memasang dependensi dengan versi yang dipatok.'
        Invoke-Checked -Executable $PythonPath -Arguments @('-m', 'pip', 'install', '--upgrade', 'pip')
        # The explicit local-version suffix also switches an old CPU installation
        # to CUDA (or the reverse); a bare ==2.9.1 would accept either build.
        Invoke-Checked -Executable $PythonPath -Arguments @(
            '-m', 'pip', 'install',
            "torch==2.9.1+$TorchIndex", "torchvision==0.24.1+$TorchIndex",
            '--index-url', "https://download.pytorch.org/whl/$TorchIndex"
        )
        Invoke-Checked -Executable $PythonPath -Arguments @('-m', 'pip', 'install', '-r', (Join-Path $ProjectRoot 'requirements.txt'))

        Write-Host '[3/5] Memeriksa kompatibilitas dependensi.'
        Invoke-Checked -Executable $PythonPath -Arguments @('-m', 'pip', 'check')

        Write-Host '[4/5] Memeriksa model, backward, backend metrik, dan perangkat.'
        $checkArguments = @((Join-Path $ProjectRoot 'scripts\check_environment.py'))
        if ($TorchIndex -ne 'cpu') {
            $checkArguments += '--require-cuda'
        }
        Invoke-Checked -Executable $PythonPath -Arguments $checkArguments

        Write-Host '[5/5] Menjalankan validasi dataset, training, evaluasi, dan prediksi.'
        $pipelineArguments = @((Join-Path $ProjectRoot 'run_pipeline.py'), '--dataset', $DatasetPath)
        if ($PSBoundParameters.ContainsKey('Epochs')) {
            $pipelineArguments += @('--epochs', [string]$Epochs)
        }
        if ($PSBoundParameters.ContainsKey('BatchSize')) {
            $pipelineArguments += @('--batch-size', [string]$BatchSize)
        }
        if ($TorchIndex -ne 'cpu') {
            $pipelineArguments += '--require-cuda'
        }
        if (-not [string]::IsNullOrWhiteSpace($PredictionImage)) {
            $pipelineArguments += @('--image', $PredictionImage)
        }
        if (-not [string]::IsNullOrWhiteSpace($Resume)) {
            $pipelineArguments += @('--resume', $Resume)
        }
        Invoke-Checked -Executable $PythonPath -Arguments $pipelineArguments
        Write-Host 'Selesai. Periksa checkpoint dan hasil pada direktori yang dilaporkan pipeline.'
    }
    finally {
        Pop-Location
    }
}
catch {
    [Console]::Error.WriteLine("GAGAL: $($_.Exception.Message)")
    [Console]::Error.WriteLine('Perbaiki penyebab yang ditampilkan, lalu jalankan kembali. Tidak ada fallback otomatis dari GPU ke CPU.')
    exit $script:FailureExitCode
}

exit 0
