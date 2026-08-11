param(
    [string]$Python = "python",
    [string]$BaseConfig = "configs/count_granger_config.yaml",
    [string]$AblationDir = "configs/count_granger_ablations",
    [string]$OutputRoot = "results/count_granger_pair_runs",
    [string]$CacheRoot = "results/cache/count_granger_pair_runs",
    [switch]$StopOnError,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

$RunFull = $true
$RunNoCluster = $false
$RunTargetOnly = $false
$RunSourceOnly = $false
$RunEdgeOnly = $false
$RunResidualOnly = $false
$RunCombinedScore = $false
$RunValidationBest = $false

function Get-SelectedAblations {
    $selected = @()
    if ($RunFull) { $selected += "full" }
    if ($RunNoCluster) { $selected += "no_cluster" }
    if ($RunTargetOnly) { $selected += "target_only" }
    if ($RunSourceOnly) { $selected += "source_only" }
    if ($RunEdgeOnly) { $selected += "edge_only" }
    if ($RunResidualOnly) { $selected += "residual_only" }
    if ($RunCombinedScore) { $selected += "combined_score" }
    if ($RunValidationBest) { $selected += "validation_best" }
    return $selected
}

$datasets = @{
    BGL = "dataset/BGL/BGL.log"
    Thunderbird = "dataset/Thunderbird/Thunderbird.log"
    Spirit = "dataset/Spirit/Spirit1G.log"
}

function ConvertTo-CsvField {
    param([object]$Value)
    if ($null -eq $Value) { return "" }
    $text = [string]$Value
    if ($text.Contains(',') -or $text.Contains('"') -or $text.Contains("`r") -or $text.Contains("`n")) {
        return '"' + $text.Replace('"', '""') + '"'
    }
    return $text
}

function Add-RunAllSummaryRow {
    param([string]$SummaryPath, [hashtable]$Row)
    $columns = @("source", "target", "ablation", "status", "exit_code", "started_at", "finished_at", "precision", "recall", "f1", "roc_auc", "pr_auc", "selected_score_component", "threshold", "num_features", "num_edges", "edge_density", "console_log", "command")
    if (-not (Test-Path $SummaryPath)) { ($columns -join ",") | Out-File -FilePath $SummaryPath -Encoding utf8 }
    $values = foreach ($column in $columns) { ConvertTo-CsvField $Row[$column] }
    Add-Content -Path $SummaryPath -Value ($values -join ",") -Encoding utf8
}

function Get-RegexValue {
    param([string]$Text, [string]$Pattern, [int]$Group = 1)
    $matches = [regex]::Matches($Text, $Pattern)
    if ($matches.Count -eq 0) { return "" }
    return $matches[$matches.Count - 1].Groups[$Group].Value
}

function Parse-ConsoleMetrics {
    param([string]$ConsoleLog)
    $result = @{precision=""; recall=""; f1=""; roc_auc=""; pr_auc=""; selected_score_component=""; threshold=""; num_features=""; num_edges=""; edge_density=""}
    if (-not (Test-Path $ConsoleLog)) { return $result }
    $text = Get-Content -Path $ConsoleLog -Raw -Encoding utf8
    $result.precision = Get-RegexValue $text 'Precision:\s*([0-9.]+),\s*Recall:\s*([0-9.]+),\s*F1-Score:\s*([0-9.]+)' 1
    $result.recall = Get-RegexValue $text 'Precision:\s*([0-9.]+),\s*Recall:\s*([0-9.]+),\s*F1-Score:\s*([0-9.]+)' 2
    $result.f1 = Get-RegexValue $text 'Precision:\s*([0-9.]+),\s*Recall:\s*([0-9.]+),\s*F1-Score:\s*([0-9.]+)' 3
    $result.roc_auc = Get-RegexValue $text "test diagnostics:\s*\{[^\n]*'roc_auc':\s*([0-9.eE+-]+)" 1
    $result.pr_auc = Get-RegexValue $text "test diagnostics:\s*\{[^\n]*'pr_auc':\s*([0-9.eE+-]+)" 1
    $result.selected_score_component = Get-RegexValue $text '\[count-eval\]\s*selected score component=([A-Za-z0-9_]+)' 1
    $result.threshold = Get-RegexValue $text 'Best validation score=[A-Za-z0-9_]+,\s*threshold=([0-9.eE+-]+)' 1
    $result.num_features = Get-RegexValue $text "fit_info=\{[^\n]*'num_features':\s*([0-9]+)" 1
    $result.num_edges = Get-RegexValue $text "fit_info=\{[^\n]*'num_edges':\s*([0-9]+)" 1
    if ($result.num_features -ne "" -and $result.num_edges -ne "") {
        $features = [double]$result.num_features
        $edges = [double]$result.num_edges
        if ($features -gt 0) { $result.edge_density = "{0:F6}" -f ($edges / ($features * $features)) }
    }
    return $result
}

function Write-ConsoleLine {
    param([string]$ConsoleLog, [string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Write-Host $line
    Add-Content -Path $ConsoleLog -Value $line -Encoding utf8
}

function Invoke-CountGrangerCommand {
    param([string]$SourceName, [string]$SourcePath, [string]$TargetName, [string]$TargetPath, [string]$Ablation, [string]$BatchRoot, [string]$BatchCacheRoot, [string]$SummaryPath)
    $pairName = "${SourceName}_to_${TargetName}"
    $commandRoot = Join-Path (Join-Path $BatchRoot $pairName) $Ablation
    $commandCacheRoot = Join-Path (Join-Path $BatchCacheRoot $pairName) $Ablation
    New-Item -ItemType Directory -Force -Path $commandRoot | Out-Null
    New-Item -ItemType Directory -Force -Path $commandCacheRoot | Out-Null
    $consoleLog = Join-Path $commandRoot "console.log"
    $startedAt = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $cmdArgs = @("run_count_granger_ablations.py", "--source_data", $SourcePath, "--target_data", $TargetPath, "--base_config", $BaseConfig, "--ablation_dir", $AblationDir, "--output_root", $commandRoot, "--cache_root", $commandCacheRoot, "--ablations", $Ablation, "--no_summary")
    if ($StopOnError) { $cmdArgs += "--stop_on_error" }
    $commandText = "$Python $($cmdArgs -join ' ')"
    Write-ConsoleLine -ConsoleLog $consoleLog -Message "Running ${pairName}/${Ablation}"
    Write-ConsoleLine -ConsoleLog $consoleLog -Message "Command: $commandText"
    $exitCode = 0
    $status = "ok"
    if (-not $DryRun) {
        & $Python @cmdArgs 2>&1 | Tee-Object -FilePath $consoleLog -Append
        $exitCode = $LASTEXITCODE
        if ($exitCode -ne 0) {
            $status = "failed"
            Write-ConsoleLine -ConsoleLog $consoleLog -Message "FAILED ${pairName}/${Ablation}, exit_code=${exitCode}"
            if ($StopOnError) { throw "${pairName}/${Ablation} failed with exit code ${exitCode}" }
        } else { Write-ConsoleLine -ConsoleLog $consoleLog -Message "Finished ${pairName}/${Ablation}" }
    } else {
        $status = "dry_run"
        Write-ConsoleLine -ConsoleLog $consoleLog -Message "Dry run skipped ${pairName}/${Ablation}"
    }
    $finishedAt = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $metrics = Parse-ConsoleMetrics -ConsoleLog $consoleLog
    Add-RunAllSummaryRow -SummaryPath $SummaryPath -Row @{source=$SourceName; target=$TargetName; ablation=$Ablation; status=$status; exit_code=$exitCode; started_at=$startedAt; finished_at=$finishedAt; precision=$metrics.precision; recall=$metrics.recall; f1=$metrics.f1; roc_auc=$metrics.roc_auc; pr_auc=$metrics.pr_auc; selected_score_component=$metrics.selected_score_component; threshold=$metrics.threshold; num_features=$metrics.num_features; num_edges=$metrics.num_edges; edge_density=$metrics.edge_density; console_log=$consoleLog; command=$commandText}
}

$Ablations = Get-SelectedAblations
if (-not $Ablations -or $Ablations.Count -eq 0) { throw "Enable at least one ablation toggle before running ablations." }
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$batchRoot = Join-Path $OutputRoot $timestamp
$batchCacheRoot = Join-Path $CacheRoot $timestamp
New-Item -ItemType Directory -Force -Path $batchRoot | Out-Null
New-Item -ItemType Directory -Force -Path $batchCacheRoot | Out-Null
$runAllSummary = Join-Path $batchRoot "run_all_summary.csv"
$batchConsole = Join-Path $batchRoot "console.log"
Write-ConsoleLine -ConsoleLog $batchConsole -Message "Count-Granger run-all batch started"
Write-ConsoleLine -ConsoleLog $batchConsole -Message "Ablations: $($Ablations -join ', ')"
Write-ConsoleLine -ConsoleLog $batchConsole -Message "run_all_summary: $runAllSummary"
foreach ($sourceName in $datasets.Keys | Sort-Object) {
    foreach ($targetName in $datasets.Keys | Sort-Object) {
        if ($sourceName -eq $targetName) { continue }
        $sourcePath = $datasets[$sourceName]
        $targetPath = $datasets[$targetName]
        if (-not (Test-Path $sourcePath)) { Write-ConsoleLine -ConsoleLog $batchConsole -Message "Skipping missing source: ${sourcePath}"; continue }
        if (-not (Test-Path $targetPath)) { Write-ConsoleLine -ConsoleLog $batchConsole -Message "Skipping missing target: ${targetPath}"; continue }
        foreach ($ablation in $Ablations) {
            Invoke-CountGrangerCommand -SourceName $sourceName -SourcePath $sourcePath -TargetName $targetName -TargetPath $targetPath -Ablation $ablation -BatchRoot $batchRoot -BatchCacheRoot $batchCacheRoot -SummaryPath $runAllSummary
        }
    }
}
Write-ConsoleLine -ConsoleLog $batchConsole -Message "Count-Granger run-all batch finished"
Write-ConsoleLine -ConsoleLog $batchConsole -Message "run_all_summary: $runAllSummary"
