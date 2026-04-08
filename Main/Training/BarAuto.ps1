# --- Configuration ---
$programPath = "C:\Path\To\Your\Program.exe" # Full path to the .exe
$PythonVenvDir = "F:\School\Capstone\Python\PyServer\.venv\Scripts"
$PythonExe = "F:\School\Capstone\Python\PyServer\.venv\Scripts\python.exe" 

$PythonDir = "F:\School\Capstone\Python\PyServer\Main" 
$PythonScript = "Socket ML.py" 
$stopFile = "stop.txt"

# .\spring-headless.exe --write-dir "F:\BAR Beyond All Reason\Beyond-All-Reason\data" _script.txt

$PyGUIWindowTitle = "Socket Reader" 

$engineDir = "F:\BAR Beyond All Reason\Beyond-All-Reason\data\engine\recoil_2025.06.19"
$dataDir   = "F:\BAR Beyond All Reason\Beyond-All-Reason\data"
$exePath   = $engineDir + "\spring-headless.exe"
$scriptArg = $engineDIr + "\_scriptL.txt"  

$scriptArgM = "\_scriptM.txt"
$scriptArgP = "\_scriptP.txt"
$scriptArgL = "\_scriptL.txt" 


$runDurationMinutes = 1                     # How long the program stays open
$totalRunTimeHours = 1                       # How long the script should loop

# --- Script Logic ---
$endTime = (Get-Date).AddHours($totalRunTimeHours)
Write-Host "Script started. Will loop until: $endTime" -ForegroundColor Cyan

do {

    

    # Go to venv location
    Push-Location $PythonDir

    Set-Content $stopFile ""

    # Start the program and keep a reference to it
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Starting program..."
    $process = Start-Process -FilePath $PythonExe -ArgumentList "`"$PythonScript`"" -PassThru

    Start-Sleep -Seconds (30) 

    $whichScript = Get-Random -InputObject $scriptArgM, $scriptArgP, $scriptArgL
    $scriptArg = $engineDIr + $whichScript

    $process2 = Start-Process -FilePath $exePath `
    -ArgumentList "--write-dir", "`"$dataDir`"", "`"$scriptArg`"" `
    -WorkingDirectory $engineDir `
    -PassThru
    
    # Start-Process -FilePath "C:\Path\To\BAR\engine\...\spring.exe" -ArgumentList "C:\Path\To\Script\script.txt" -WindowStyle Hidden

    for ($i = 1; $i -le 10; $i++) {

        # Wait for 10 minutes
        Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Waiting 1 minute..."
        Start-Sleep -Seconds (60)

        if ($process2 -and $process2.HasExited) {
        Write-Host "[$(Get-Date -Format 'HH:mm:ss')] BAR no longer detected at ID/Process."
            break
        }
    }

    # Close the program
    if ($process -and -not $process.HasExited) {
        Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Closing program."
        # kill -SIGINT $process.Id
        #Stop-Process -Id $process.Id -Force
        Set-Content $stopFile "stop"
    }

    if ($process2 -and -not $process2.HasExited) {
        Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Closing BAR."
        # kill -SIGINT $process.Id
        Stop-Process -Id $process2.Id -Force
        #Set-Content $stopFile "stop"
    }

    # Optional: Brief pause before restarting
    Start-Sleep -Seconds 15 

} while ((Get-Date) -lt $endTime)

Write-Host "Total duration of $totalRunTimeHours hours reached. Script complete." -ForegroundColor Green
