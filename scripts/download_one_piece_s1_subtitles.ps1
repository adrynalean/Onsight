$ErrorActionPreference = 'Stop'

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Destination = Join-Path $ProjectRoot 'data\One_Piece_Anime_S1_English'
$TempRoot = Join-Path $env:TEMP ('opensubs_onepiece_s1_' + [guid]::NewGuid().ToString('N'))

New-Item -ItemType Directory -Force -Path $Destination | Out-Null
New-Item -ItemType Directory -Force -Path $TempRoot | Out-Null

try {
    for ($Episode = 1; $Episode -le 61; $Episode++) {
        $EpisodeLabel = '{0:D2}' -f $Episode
        $Existing = Get-ChildItem -Path $Destination -File |
            Where-Object { $_.Name -match " - $EpisodeLabel\.(ass|srt)$" } |
            Select-Object -First 1

        if ($Existing) {
            Write-Host "Skipping episode $EpisodeLabel; already downloaded."
            continue
        }

        $OpenSubtitlesId = 13623128 + $Episode
        $Url = "https://dl.opensubtitles.org/en/download/sub/$OpenSubtitlesId"
        $ZipPath = Join-Path $TempRoot "episode_$EpisodeLabel.zip"
        $ExtractDir = Join-Path $TempRoot "episode_$EpisodeLabel"

        New-Item -ItemType Directory -Force -Path $ExtractDir | Out-Null
        Write-Host "Downloading episode $EpisodeLabel from $Url"
        Invoke-WebRequest -Uri $Url -OutFile $ZipPath -UseBasicParsing -Headers @{ 'User-Agent' = 'Mozilla/5.0' }
        Expand-Archive -LiteralPath $ZipPath -DestinationPath $ExtractDir -Force

        $Subtitle = Get-ChildItem -Path $ExtractDir -Recurse -File |
            Where-Object { $_.Extension -in '.ass', '.srt' } |
            Select-Object -First 1

        if (-not $Subtitle) {
            throw "No subtitle file found for episode $EpisodeLabel."
        }

        $Target = Join-Path $Destination ("One Piece Season 1 - $EpisodeLabel" + $Subtitle.Extension.ToLowerInvariant())
        Copy-Item -LiteralPath $Subtitle.FullName -Destination $Target -Force
        Start-Sleep -Seconds 3
    }
}
finally {
    if (Test-Path -LiteralPath $TempRoot) {
        Remove-Item -LiteralPath $TempRoot -Recurse -Force
    }
}
