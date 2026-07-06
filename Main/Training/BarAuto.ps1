# --- Configuration ---
$programPath = "C:\Path\To\Your\Program.exe" # Full path to the .exe
$PythonVenvDir = "F:\School\Capstone\Python\PyServer\.venv\Scripts"
$PythonExe = "F:\School\Capstone\Python\PyServer\.venv\Scripts\python.exe" 

$PythonDir = "F:\School\Capstone\Python\PyServer\Main" 
$PythonScript = "Socket ML.py" 
$stopFile = "stop.txt"
$timeFile = "time.txt"
$runFile = "runMode.txt"
$recentRun = "recentRun.txt"
$recentRunError = "recentRunError.txt"

# .\spring-headless.exe --write-dir "F:\BAR Beyond All Reason\Beyond-All-Reason\data" _script.txt

# $PyGUIWindowTitle = "Socket Reader" 

$engineDir = "F:\BAR Beyond All Reason\Beyond-All-Reason\data\engine\recoil_2025.06.24"
$dataDir   = "F:\BAR Beyond All Reason\Beyond-All-Reason\data"
$exePath   = $engineDir + "\spring-headless.exe"
$scriptArg = $engineDir + "\_scriptL.txt"  

# $scriptArgM = "\_scriptM.txt"
# $scriptArgP = "\_scriptP.txt"
# $scriptArgL = "\_scriptL.txt" 
# $scriptArgC = "\_scriptC.txt" 
$scriptArgEval = "\_scriptShowcase.txt"
$scriptArgTraining = "\_scriptTraining.txt"

$totalRunTimeHours = 4                       # How long the script should loop

# --- Script Logic ---
$endTime = (Get-Date).AddHours($totalRunTimeHours)
Write-Host "Script started. Will loop until: $endTime" -ForegroundColor Cyan

do {

    $howLongDidItSurvive = 0

    # Go to venv location
    Push-Location $PythonDir

    Set-Content $stopFile ""
    Set-Content $timeFile "0"

    # Start the program and keep a reference to it
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Starting program..."
    $process = Start-Process -FilePath $PythonExe -ArgumentList "`"$PythonScript`"" -RedirectStandardOutput $recentRun -RedirectStandardError $recentRunError -PassThru -WindowStyle Minimized

    Start-Sleep -Seconds (30) 

    $CurrentRunType = (Get-Content $runFile -Raw).Trim()
    # $CurrentRunType = Get-Content $runFile
    if ($CurrentRunType -eq 'eval') {
        Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Detected eval run type. Using _scriptShowcase.txt for BAR."
        $whichScript = $scriptArgEval
    }
    else {
        Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Detected training run type. Using random script for BAR."
        # $whichScript = Get-Random -InputObject $scriptArgM, $scriptArgP, $scriptArgL, $scriptArgC
        $whichScript = $scriptArgTraining
    }
    # $whichScript = Get-Random -InputObject $scriptArgM, $scriptArgP, $scriptArgL, $scriptArgC
    $scriptArg = $engineDir + $whichScript

    $process2 = Start-Process -FilePath $exePath `
    -ArgumentList "--write-dir", "`"$dataDir`"", "`"$scriptArg`"" `
    -WorkingDirectory $engineDir `
    -PassThru `
    -WindowStyle Minimized
    
    # Start-Process -FilePath "C:\Path\To\BAR\engine\...\spring.exe" -ArgumentList "C:\Path\To\Script\script.txt" -WindowStyle Hidden

    $howLongDidItSurvive = 0

    for ($i = 1; $i -le 10; $i++) {

        # Wait for 10 minutes
        Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Waiting 1 minute..."
        Start-Sleep -Seconds (60)

        if ($process2 -and $process2.HasExited) {
        Write-Host "[$(Get-Date -Format 'HH:mm:ss')] BAR no longer detected at ID/Process."
            break
        }

        $howLongDidItSurvive += 1

        Set-Content $timeFile "$howLongDidItSurvive"
    }

    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] BAR survived for $howLongDidItSurvive minutes."

    # Close the program
    if ($process -and -not $process.HasExited) {
        Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Closing program."
        # kill -SIGINT $process.Id
        #Stop-Process -Id $process.Id -Force
        Set-Content $stopFile "stop"
        Set-Content $timeFile "$howLongDidItSurvive"
    }

    Start-Sleep -Seconds 60

    if ($process2 -and -not $process2.HasExited) {
        Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Closing BAR."
        # kill -SIGINT $process.Id
        Stop-Process -Id $process2.Id -Force
        #Set-Content $stopFile "stop"
    }

    # Loop until initial program is done training
    while ($process -and -not $process.HasExited) {
        Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Checking if done training."
        # kill -SIGINT $process.Id
        #Stop-Process -Id $process.Id -Force
        Set-Content $stopFile "stop"
        Set-Content $timeFile "$howLongDidItSurvive"
        Start-Sleep -Seconds 15
    }

    # Optional: Brief pause before restarting
    Start-Sleep -Seconds 15

} while ((Get-Date) -lt $endTime)

Set-Content $stopFile ""
Set-Content $timeFile ""

Write-Host "Total duration of $totalRunTimeHours hours reached. Script complete." -ForegroundColor Green
