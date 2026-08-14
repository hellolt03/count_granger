param(
    [string]$Python = "python",
    [string]$BaseConfig = "configs/count_granger_config.yaml",
    [string]$AblationDir = "configs/count_granger_ablations",
    [string]$DirectionConfigDir = "configs/count_granger_directions",
    [string]$OutputRoot = "results/count_granger_pair_runs",
    [string]$CacheRoot = "results/cache/count_granger",
    [switch]$StopOnError,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$script:Utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[Console]::OutputEncoding = $script:Utf8NoBom
$OutputEncoding = $script:Utf8NoBom

$RunFull = $false
$RunNoCluster = $false
$RunTargetOnly = $false
$RunSourceOnly = $false
$RunEdgeOnly = $false
$RunResidualOnly = $false
$RunRobustResidualOnly = $false
$RunEdgeConsistencyOnly = $false
$RunCausalScore = $false
$RunCombinedScore = $false
$RunValidationBest = $false

# $RunDirectionThunderbirdToSpirit = $false
# $RunDirectionThunderbirdToBGL = $false
# $RunDirectionSpiritToBGL = $false
# $RunDirectionThunderbirdToSpiritWeightedSearch = $false
# $RunDirectionThunderbirdToBGLWeightedSearch = $false
# $RunDirectionSpiritToBGLWeightedSearch = $false
$RunDirectionThunderbirdToBGLStrongGe3 = $true
$RunDirectionThunderbirdToBGLStrongGe5 = $true
$RunDirectionThunderbirdToBGLStrongGe10 = $true
$RunDirectionSpiritToBGLStrongGe3 = $true
$RunDirectionSpiritToBGLStrongGe5 = $true
$RunDirectionSpiritToBGLStrongGe10 = $true
# $RunDirectionThunderbirdToBGL120sPost = $false
# $RunDirectionThunderbirdToBGL300sPost = $false
# $RunDirectionSpiritToBGL120sPost = $false
# $RunDirectionSpiritToBGL300sPost = $false

function Get-SelectedAblations {
    $selected = @()
    if ($RunFull) { $selected += "full" }
    if ($RunNoCluster) { $selected += "no_cluster" }
    if ($RunTargetOnly) { $selected += "target_only" }
    if ($RunSourceOnly) { $selected += "source_only" }
    if ($RunEdgeOnly) { $selected += "edge_only" }
    if ($RunResidualOnly) { $selected += "residual_only" }
    if ($RunRobustResidualOnly) { $selected += "robust_residual_only" }
    if ($RunEdgeConsistencyOnly) { $selected += "edge_consistency_only" }
    if ($RunCausalScore) { $selected += "causal_score" }
    if ($RunCombinedScore) { $selected += "combined_score" }
    if ($RunValidationBest) { $selected += "validation_best" }
    return $selected
}

function Get-SelectedDirectionRuns {
    $selected = @()
    if ($RunDirectionThunderbirdToSpirit) {
        $selected += @{ Source = "Thunderbird"; Target = "Spirit"; Config = "Thunderbird_to_Spirit" }
    }
    if ($RunDirectionThunderbirdToBGL) {
        $selected += @{ Source = "Thunderbird"; Target = "BGL"; Config = "Thunderbird_to_BGL" }
    }
    if ($RunDirectionSpiritToBGL) {
        $selected += @{ Source = "Spirit"; Target = "BGL"; Config = "Spirit_to_BGL" }
    }
    if ($RunDirectionThunderbirdToSpiritWeightedSearch) {
        $selected += @{ Source = "Thunderbird"; Target = "Spirit"; Config = "Thunderbird_to_Spirit_weighted_search" }
    }
    if ($RunDirectionThunderbirdToBGLWeightedSearch) {
        $selected += @{ Source = "Thunderbird"; Target = "BGL"; Config = "Thunderbird_to_BGL_weighted_search" }
    }
    if ($RunDirectionSpiritToBGLWeightedSearch) {
        $selected += @{ Source = "Spirit"; Target = "BGL"; Config = "Spirit_to_BGL_weighted_search" }
    }
    if ($RunDirectionThunderbirdToBGLStrongGe3) {
        $selected += @{ Source = "Thunderbird"; Target = "BGL"; Config = "Thunderbird_to_BGL_strong_ge3" }
    }
    if ($RunDirectionThunderbirdToBGLStrongGe5) {
        $selected += @{ Source = "Thunderbird"; Target = "BGL"; Config = "Thunderbird_to_BGL_strong_ge5" }
    }
    if ($RunDirectionThunderbirdToBGLStrongGe10) {
        $selected += @{ Source = "Thunderbird"; Target = "BGL"; Config = "Thunderbird_to_BGL_strong_ge10" }
    }
    if ($RunDirectionSpiritToBGLStrongGe3) {
        $selected += @{ Source = "Spirit"; Target = "BGL"; Config = "Spirit_to_BGL_strong_ge3" }
    }
    if ($RunDirectionSpiritToBGLStrongGe5) {
        $selected += @{ Source = "Spirit"; Target = "BGL"; Config = "Spirit_to_BGL_strong_ge5" }
    }
    if ($RunDirectionSpiritToBGLStrongGe10) {
        $selected += @{ Source = "Spirit"; Target = "BGL"; Config = "Spirit_to_BGL_strong_ge10" }
    }
    if ($RunDirectionThunderbirdToBGL120sPost) {
        $selected += @{ Source = "Thunderbird"; Target = "BGL"; Config = "Thunderbird_to_BGL_120s_post" }
    }
    if ($RunDirectionThunderbirdToBGL300sPost) {
        $selected += @{ Source = "Thunderbird"; Target = "BGL"; Config = "Thunderbird_to_BGL_300s_post" }
    }
    if ($RunDirectionSpiritToBGL120sPost) {
        $selected += @{ Source = "Spirit"; Target = "BGL"; Config = "Spirit_to_BGL_120s_post" }
    }
    if ($RunDirectionSpiritToBGL300sPost) {
        $selected += @{ Source = "Spirit"; Target = "BGL"; Config = "Spirit_to_BGL_300s_post" }
    }
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
    $bytes = [System.IO.File]::ReadAllBytes((Resolve-Path $ConsoleLog))
    $text = [System.Text.Encoding]::UTF8.GetString($bytes)
    $text = $text -replace "`0", ""
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
    param(
        [string]$SourceName,
        [string]$SourcePath,
        [string]$TargetName,
        [string]$TargetPath,
        [string]$Ablation,
        [string]$BatchRoot,
        [string]$BatchCacheRoot,
        [string]$SummaryPath,
        [string]$RunBaseConfig = $BaseConfig,
        [string]$RunAblationDir = $AblationDir
    )
    $pairName = "${SourceName}_to_${TargetName}"
    $commandRoot = Join-Path (Join-Path $BatchRoot $pairName) $Ablation
    $commandCacheRoot = $BatchCacheRoot
    New-Item -ItemType Directory -Force -Path $commandRoot | Out-Null
    New-Item -ItemType Directory -Force -Path $commandCacheRoot | Out-Null
    $consoleLog = Join-Path $commandRoot "console.log"
    $startedAt = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $cmdArgs = @("run_count_granger_ablations.py", "--source_data", $SourcePath, "--target_data", $TargetPath, "--base_config", $RunBaseConfig, "--ablation_dir", $RunAblationDir, "--output_root", $commandRoot, "--cache_root", $commandCacheRoot, "--ablations", $Ablation, "--no_summary")
    if ($StopOnError) { $cmdArgs += "--stop_on_error" }
    $commandText = "$Python $($cmdArgs -join ' ')"
    Write-ConsoleLine -ConsoleLog $consoleLog -Message "Running ${pairName}/${Ablation}"
    Write-ConsoleLine -ConsoleLog $consoleLog -Message "Command: $commandText"
    $exitCode = 0
    $status = "ok"
    if (-not $DryRun) {
        & $Python @cmdArgs 2>&1 | ForEach-Object {
            $line = [string]$_
            Write-Host $line
            Add-Content -Path $consoleLog -Value $line -Encoding utf8
        }
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
$DirectionRuns = Get-SelectedDirectionRuns
if ((-not $Ablations -or $Ablations.Count -eq 0) -and (-not $DirectionRuns -or $DirectionRuns.Count -eq 0)) {
    throw "Enable at least one ablation toggle or direction config toggle before running."
}
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$batchRoot = Join-Path $OutputRoot $timestamp
$batchCacheRoot = $CacheRoot
New-Item -ItemType Directory -Force -Path $batchRoot | Out-Null
New-Item -ItemType Directory -Force -Path $batchCacheRoot | Out-Null
$runAllSummary = Join-Path $batchRoot "run_all_summary.csv"
$batchConsole = Join-Path $batchRoot "console.log"
Write-ConsoleLine -ConsoleLog $batchConsole -Message "Count-Granger run-all batch started"
Write-ConsoleLine -ConsoleLog $batchConsole -Message "Ablations: $($Ablations -join ', ')"
Write-ConsoleLine -ConsoleLog $batchConsole -Message "Direction configs: $(($DirectionRuns | ForEach-Object { $_.Config }) -join ', ')"
Write-ConsoleLine -ConsoleLog $batchConsole -Message "run_all_summary: $runAllSummary"
if ($Ablations -and $Ablations.Count -gt 0) {
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
}
if ($DirectionRuns -and $DirectionRuns.Count -gt 0) {
    foreach ($direction in $DirectionRuns) {
        $sourceName = $direction.Source
        $targetName = $direction.Target
        $directionConfig = $direction.Config
        $sourcePath = $datasets[$sourceName]
        $targetPath = $datasets[$targetName]
        if (-not (Test-Path $sourcePath)) { Write-ConsoleLine -ConsoleLog $batchConsole -Message "Skipping missing direction source: ${sourcePath}"; continue }
        if (-not (Test-Path $targetPath)) { Write-ConsoleLine -ConsoleLog $batchConsole -Message "Skipping missing direction target: ${targetPath}"; continue }
        $directionConfigPath = Join-Path $DirectionConfigDir "${directionConfig}.yaml"
        if (-not (Test-Path $directionConfigPath)) { Write-ConsoleLine -ConsoleLog $batchConsole -Message "Skipping missing direction config: ${directionConfigPath}"; continue }
        Invoke-CountGrangerCommand -SourceName $sourceName -SourcePath $sourcePath -TargetName $targetName -TargetPath $targetPath -Ablation $directionConfig -BatchRoot $batchRoot -BatchCacheRoot $batchCacheRoot -SummaryPath $runAllSummary -RunBaseConfig $BaseConfig -RunAblationDir $DirectionConfigDir
    }
}
Write-ConsoleLine -ConsoleLog $batchConsole -Message "Count-Granger run-all batch finished"
Write-ConsoleLine -ConsoleLog $batchConsole -Message "run_all_summary: $runAllSummary"



