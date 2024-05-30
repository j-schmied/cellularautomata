param (
    [Parameter(Mandatory=$true)]
    [string] $n,

    [Parameter(Mandatory=$false)]
    [switch] $plot,

    [Parameter(Mandatory=$false)]
    [switch] $box
)

if ($n -notmatch '^\d+$') {
    Write-Output "Invalid input: -n parameter must be a positive integer"
    exit
}

$MultiverseId = Get-Date -Format "yyyyMMddHHmmss"
$Processes = @()

$ExperimentalDataDir = "G:\DataMedAssist\Experiments\Cellular_Automata\Puliafito_2012_01_17"
$OutputDir = "G:\DataMedAssist\Eval\cellularautomata_results"

for ($i=0; $i -lt $n; $i++) {
    # Find valid options in README
    # do NOT alter --multiverse or --real!
    if ($box) {
        $Proc = Start-Process -PassThru python -ArgumentList '-m src.cellularautomaton', '--multiverse', $MultiverseId, '--well',  $i, '--box', '--exportcsv', '--output', $OutputDir, '--expdata', $ExperimentalDataDir
    } else {
        $Proc = Start-Process -PassThru python -ArgumentList '-m src.cellularautomaton', '--multiverse', $MultiverseId, '--well',  $i, '--colonial', '--exportcsv', '--output', $OutputDir, '--expdata', $ExperimentalDataDir
    }

    $Processes += $Proc
    Start-Sleep -s 2
}

if ($plot) {
    # Wait for each spawned process to finish before possibly starting to plot
    foreach ($Proc in $Processes) {
        $Proc | Wait-Process
    }

    # Adapt ExperimentalDataDir and OutputDir to your needs!
    # NOTE: --exportcsv is necessary for plotting to work!
    if ($box) {
        python -m src.utils.plot --box --path $OutputDir\box\multiverse_$MultiverseId --expdata $ExperimentalDataDir --output pdf
    } else {
        python -m src.utils.plot --path $OutputDir\colonial\multiverse_$MultiverseId --expdata $ExperimentalDataDir --output pdf
    }
}