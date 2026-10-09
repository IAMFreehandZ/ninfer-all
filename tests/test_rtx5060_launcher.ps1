$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$launcher = Join-Path $root 'scripts/rtx5060-run.ps1'
if (-not (Test-Path -LiteralPath $launcher)) { throw 'FAIL: RTX 5060 launcher is missing.' }
. $launcher

function Assert-Equal($Actual, $Expected, [string]$Message) {
    if (($Actual | ConvertTo-Json -Compress -Depth 10) -cne ($Expected | ConvertTo-Json -Compress -Depth 10)) {
        throw "FAIL: $Message. Actual: $($Actual | ConvertTo-Json -Compress -Depth 10)"
    }
}
function Assert-Throws([scriptblock]$Action, [string]$Pattern) {
    try { & $Action } catch {
        if ($_.Exception.Message -notmatch $Pattern) { throw "FAIL: unexpected error: $_" }
        return
    }
    throw "FAIL: expected error matching $Pattern"
}

Assert-Equal (ConvertTo-NInferWslPath '\\wsl.localhost\Ubuntu\home\linux user\models\Bonsai 2.ninfer') '/home/linux user/models/Bonsai 2.ninfer' 'native WSL model path preserves spaces'
Assert-Equal (ConvertTo-NInferWslPath '\\wsl$\Ubuntu\home\linux user\models\Bonsai 2.ninfer') '/home/linux user/models/Bonsai 2.ninfer' 'Windows WSL share name maps to the same native model'
Assert-Equal (ConvertTo-NInferWslPath '\\wsl.localhost\Ubuntu-24.04\home\models\Bonsai.ninfer' -Distribution 'Ubuntu-24.04') '/home/models/Bonsai.ninfer' 'explicit distribution owns the native model path'
Assert-Throws { ConvertTo-NInferWslPath '\\wsl.localhost\Debian\home\models\Bonsai.ninfer' -Distribution Ubuntu } 'distribution'
Assert-Throws { ConvertTo-NInferWslPath '\\fileserver\models\Bonsai.ninfer' } 'path'

$fixture = Join-Path $env:TEMP ('ninfer launcher ' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path (Join-Path $fixture 'build/apps'), (Join-Path $fixture 'models/bonsai2') -Force | Out-Null
try {
    Set-Content -LiteralPath (Join-Path $fixture 'build/apps/ninfer-serve') -Value ''
    Set-Content -LiteralPath (Join-Path $fixture 'models/bonsai2/Ternary-Bonsai-2-27B-ninfer-v3.ninfer') -Value ''
    $plan = Get-NInferLaunchPlan -Root $fixture
    Assert-Equal $plan.ServerArguments @($plan.ModelPath, '--host', '127.0.0.1', '--port', '8080', '--model-id', 'bonsai2-27b', '--max-context', '65536', '--max-concurrency', '2', '--kv-dtype', 'rk4v4', '--gdn-state-fp16', '--kv-capacity', '131072', '--concurrent-prefill', '--cuda-graph-allowance-mib', '256', '--spec', 'dflash2', '--draft-tokens', '7', '--lm-head-draft', '--temperature', '0', '--no-thinking') 'default serving arguments'
    Assert-Equal ($plan.ServerArguments -contains '--greedy') $false 'server temperature default preserves per-request sampling overrides'
    $thinkingPlan = Get-NInferLaunchPlan -Root $fixture -Temperature 0.7 -Thinking
    Assert-Equal $thinkingPlan.ServerArguments[-2..-1] @('--temperature', '0.7') 'explicit temperature and thinking arguments'
    Assert-Equal ($thinkingPlan.ServerArguments -contains '--no-thinking') $false 'thinking opt-in omits no-thinking'
    $disabledThinking = Get-NInferLaunchPlan -Root $fixture -Thinking:$false
    Assert-Equal $disabledThinking.ServerArguments $plan.ServerArguments 'explicit false retains nonthinking default'
    $originalCulture = [Threading.Thread]::CurrentThread.CurrentCulture
    try {
        [Threading.Thread]::CurrentThread.CurrentCulture = [Globalization.CultureInfo]::GetCultureInfo('fr-FR')
        $culturePlan = Get-NInferLaunchPlan -Root $fixture -Temperature 0.7 -Thinking
        Assert-Equal $culturePlan.ServerArguments[-1] '0.7' 'temperature uses invariant decimal formatting'
    } finally { [Threading.Thread]::CurrentThread.CurrentCulture = $originalCulture }
    Assert-Throws { Get-NInferLaunchPlan -Root $fixture -Temperature -0.1 } 'Temperature|range|validation'
    Assert-Throws { Get-NInferLaunchPlan -Root $fixture -Temperature 2.1 } 'Temperature|range|validation'
    Assert-Throws { Get-NInferLaunchPlan -Root $fixture -Temperature ([double]::NaN) } 'Temperature|range|validation'
    Assert-Equal ($plan.ServerArguments -contains '--fast-prefill-kernel') $false 'fast prefill is disabled by default'
    $fastPlan = Get-NInferLaunchPlan -Root $fixture -FastPrefill
    Assert-Equal $fastPlan.ServerArguments ($plan.ServerArguments + @('--fast-prefill-kernel')) 'fast prefill is an explicit serving argument'
    $fastNone = Get-NInferLaunchPlan -Root $fixture -Spec none -FastPrefill
    Assert-Equal ($fastNone.ServerArguments -contains '--fast-prefill-kernel') $true 'fast prefill works without speculation'
    Assert-Equal ($fastNone.ServerArguments -contains '--spec') $false 'fast prefill preserves speculation selection'
    $disabledFast = Get-NInferLaunchPlan -Root $fixture -FastPrefill:$false
    Assert-Equal $disabledFast.ServerArguments $plan.ServerArguments 'explicit false keeps fast prefill disabled'
    if ($plan.ModelPath -notmatch '^/mnt/[a-z]/.*ninfer launcher ') { throw 'FAIL: model path with spaces was not preserved.' }
    $none = Get-NInferLaunchPlan -Root $fixture -Spec none -Context 8192 -Port 8090
    Assert-Equal ($none.ServerArguments -contains '--spec') $false 'none disables speculation'
    Assert-Equal ($none.ServerArguments -contains '--draft-tokens') $false 'none omits draft width'
    Assert-Equal $none.Url 'http://localhost:8090' 'custom port URL'
    Assert-Equal $none.ServerArguments[8] '8192' 'custom context'
    $dflash = Get-NInferLaunchPlan -Root $fixture -Spec dflash2 -DraftTokens 7
    Assert-Equal $dflash.ServerArguments[-8..-1] @('--spec', 'dflash2', '--draft-tokens', '7', '--lm-head-draft', '--temperature', '0', '--no-thinking') 'DFlash2 arguments'
    $dflashFive = Get-NInferLaunchPlan -Root $fixture -Spec dflash2 -DraftTokens 5
    Assert-Equal $dflashFive.ServerArguments[-8..-1] @('--spec', 'dflash2', '--draft-tokens', '5', '--lm-head-draft', '--temperature', '0', '--no-thinking') 'explicit DFlash2 five-draft option'
    $mtpThree = Get-NInferLaunchPlan -Root $fixture -Spec mtp -DraftTokens 3
    Assert-Equal $mtpThree.ServerArguments[-8..-1] @('--spec', 'mtp', '--draft-tokens', '3', '--lm-head-draft', '--temperature', '0', '--no-thinking') 'explicit MTP three-draft option'
    Assert-Throws { Get-NInferLaunchPlan -Root $fixture -Spec invalid } 'Spec|ValidateSet|validation'
    Assert-Throws { Get-NInferLaunchPlan -Root $fixture -Context 0 } 'Context|range|validation'
    Assert-Throws { Get-NInferLaunchPlan -Root $fixture -Port 65536 } 'Port|range|validation'
    Assert-Throws { Get-NInferLaunchPlan -Root $fixture -DraftTokens 16 } 'DraftTokens|range|validation'
    $single = Get-NInferLaunchPlan -Root $fixture -Concurrency 1
    Assert-Equal $single.ServerArguments[10] '1' 'explicit single-request concurrency'
    Assert-Equal ($single.ServerArguments -contains '--kv-capacity') $false 'single request keeps the KV pool at its context'
    Assert-Equal ($single.ServerArguments -contains '--concurrent-prefill') $false 'single request omits concurrent prefill'
    Assert-Equal $single.ServerArguments[14..15] @('--cuda-graph-allowance-mib', '128') 'single-request graph allowance'
    $longPair = Get-NInferLaunchPlan -Root $fixture -Context 262144
    Assert-Equal $longPair.ServerArguments[14..15] @('--kv-capacity', '262144') 'concurrent KV pool stops at the single-request maximum'
    $four = Get-NInferLaunchPlan -Root $fixture -Context 32768 -Concurrency 4
    Assert-Equal $four.ServerArguments[14..18] @('--kv-capacity', '131072', '--concurrent-prefill', '--cuda-graph-allowance-mib', '512') 'four requests share four contexts'
    Assert-Throws { Get-NInferLaunchPlan -Root $fixture -Concurrency 0 } 'Concurrency|range|validation'
    Assert-Throws { Get-NInferLaunchPlan -Root $fixture -Concurrency 9 } 'Concurrency|range|validation'
    Remove-Item -LiteralPath (Join-Path $fixture 'build/apps/ninfer-serve')
    Assert-Throws { Get-NInferLaunchPlan -Root $fixture } 'server.*missing|missing.*server'
    Set-Content -LiteralPath (Join-Path $fixture 'build/apps/ninfer-serve') -Value ''
    Remove-Item -LiteralPath (Join-Path $fixture 'models/bonsai2/Ternary-Bonsai-2-27B-ninfer-v3.ninfer')
    Assert-Throws { Get-NInferLaunchPlan -Root $fixture } 'model.*missing|missing.*model'

    # Exercise Windows argument parsing through an actual native child process.
    $capture = Join-Path $fixture 'capture arguments.ps1'
    Set-Content -LiteralPath $capture -Value '$args | ConvertTo-Json -Compress'
    $values = @('/mnt/c/a space/file', 'single''quote', 'literal$dollar', 'a"quote', 'trailing\', '')
    $interpreter = [System.Diagnostics.Process]::GetCurrentProcess().MainModule.FileName
    $info = New-Object System.Diagnostics.ProcessStartInfo
    $info.FileName = $interpreter
    $info.Arguments = ConvertTo-NInferWindowsArguments (@('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $capture) + $values)
    $info.UseShellExecute = $false
    $info.RedirectStandardOutput = $true
    $process = [System.Diagnostics.Process]::Start($info)
    $captured = $process.StandardOutput.ReadToEnd()
    $process.WaitForExit()
    Assert-Equal $process.ExitCode 0 'argument capture exit code'
    Assert-Equal ($captured | ConvertFrom-Json) $values 'Windows native argument round trip'

    # Invoke the actual entry point with Root omitted, as the CMD launcher does.
    $info.Arguments = ConvertTo-NInferWindowsArguments @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $launcher, '-Server', (Join-Path $fixture 'missing-server'), '-NoBrowser')
    $info.RedirectStandardError = $true
    $process = [System.Diagnostics.Process]::Start($info)
    $entryOutput = $process.StandardOutput.ReadToEnd() + $process.StandardError.ReadToEnd()
    $process.WaitForExit()
    Assert-Equal $process.ExitCode 1 'external PowerShell entry point failure status'
    if ($entryOutput -notmatch 'server is missing' -or $entryOutput -match 'PSScriptRoot|empty string') {
        throw "FAIL: omitted Root did not report the missing server: $entryOutput"
    }
    $info.FileName = $env:ComSpec
    $info.Arguments = '/d /c ""' + (Join-Path $root 'Start-Bonsai.cmd') + '" -Server "' + (Join-Path $fixture 'missing-server') + '" -NoBrowser"'
    $info.RedirectStandardInput = $true
    $process = [System.Diagnostics.Process]::Start($info)
    $process.StandardInput.WriteLine('')
    $process.StandardInput.Close()
    $entryOutput = $process.StandardOutput.ReadToEnd() + $process.StandardError.ReadToEnd()
    $process.WaitForExit()
    Assert-Equal $process.ExitCode 1 'CMD entry point preserves failure status'
    if ($entryOutput -notmatch 'server is missing') { throw "FAIL: CMD entry point error was lost: $entryOutput" }
    $info.RedirectStandardInput = $false
    Write-Host 'PASS: RTX 5060 launcher arguments, paths, validation, and native quoting.'

    if (Get-Command wsl.exe -ErrorAction SilentlyContinue) {
        New-Item -ItemType Directory -Path (Join-Path $fixture 'scripts') -Force | Out-Null
        Copy-Item -LiteralPath (Join-Path $root 'scripts/rtx5060-run.sh') -Destination (Join-Path $fixture 'scripts/rtx5060-run.sh')
        $pythonCapture = Join-Path $fixture 'capture arguments.py'
        [IO.File]::WriteAllText($pythonCapture, "import json, os, sys`nprint(json.dumps({'args': sys.argv[1:], 'pid': os.getpid(), 'group': os.getpgrp()}))`n")
        $pidPath = Join-Path $fixture 'wrapper.pid'
        $nativeArguments = @('--distribution', 'Ubuntu', '--exec', '/usr/bin/setsid', '--wait', '/bin/bash',
            (ConvertTo-NInferWslPath (Join-Path $fixture 'scripts/rtx5060-run.sh')), (ConvertTo-NInferWslPath $pidPath),
            '/usr/bin/python3', (ConvertTo-NInferWslPath $pythonCapture)) + $values
        $info.FileName = (Get-Command wsl.exe).Source
        $info.Arguments = ConvertTo-NInferWindowsArguments $nativeArguments
        $info.RedirectStandardInput = $true
        $process = [System.Diagnostics.Process]::Start($info)
        $process.StandardInput.NewLine = "`n"
        $process.StandardInput.WriteLine('start')
        $process.StandardInput.Flush()
        $captured = $process.StandardOutput.ReadToEnd()
        $process.WaitForExit()
        Assert-Equal $process.ExitCode 0 'WSL argument capture exit code'
        $decoded = $captured | ConvertFrom-Json
        Assert-Equal $decoded.args $values 'WSL native argument round trip'
        Assert-Equal $decoded.pid ([int](Get-Content -LiteralPath $pidPath)) 'WSL recorded process identity'
        Assert-Equal $decoded.group $decoded.pid 'isolated server process group'
        $process.StandardInput.Close()
        $process.Dispose()
        $ErrorActionPreference = 'Continue'
        & (Get-Command wsl.exe).Source --distribution Ubuntu --exec /bin/kill -0 -- "-$($decoded.group)" 2>$null
        $probeExitCode = $LASTEXITCODE
        $ErrorActionPreference = 'Stop'
        if ($probeExitCode -eq 0) { throw 'FAIL: normal wrapper exit left its server group running.' }

        # Assignment/startup failure before the handshake cannot launch a Linux server.
        $abortedPidPath = Join-Path $fixture 'aborted.pid'
        $abortArguments = @('--distribution', 'Ubuntu', '--exec', '/usr/bin/setsid', '--wait', '/bin/bash',
            (ConvertTo-NInferWslPath (Join-Path $fixture 'scripts/rtx5060-run.sh')), (ConvertTo-NInferWslPath $abortedPidPath),
            '/usr/bin/python3', (ConvertTo-NInferWslPath $pythonCapture))
        $info.Arguments = ConvertTo-NInferWindowsArguments $abortArguments
        $process = [System.Diagnostics.Process]::Start($info)
        $process.StandardInput.Close()
        $captured = $process.StandardOutput.ReadToEnd()
        $process.WaitForExit()
        Assert-Equal $process.ExitCode 1 'closed lifetime pipe before permission prevents launch'
        Assert-Equal $captured '' 'aborted launch did not execute the server'
        if (Test-Path -LiteralPath $abortedPidPath) { throw 'FAIL: aborted launch published a server process.' }
        $process.Dispose()
        Write-Host 'PASS: closing the lifetime pipe before the start handshake prevents server execution.'

        # A harmless sleeping server never becomes healthy. Timeout must terminate it.
        [IO.File]::WriteAllText((Join-Path $fixture 'build/apps/ninfer-serve'), "#!/bin/bash`nexec /bin/sleep 60`n")
        Set-Content -LiteralPath (Join-Path $fixture 'models/bonsai2/Ternary-Bonsai-2-27B-ninfer-v3.ninfer') -Value ''
        $listener = New-Object Net.Sockets.TcpListener([Net.IPAddress]::Loopback, 0)
        $listener.Start()
        $testPort = $listener.LocalEndpoint.Port
        $listener.Stop()
        $timeoutPlan = Get-NInferLaunchPlan -Root $fixture -Port $testPort -Spec none
        Assert-Throws { Invoke-NInferLauncher -Plan $timeoutPlan -Distribution Ubuntu -NoBrowser -StartupTimeoutSeconds 1 } 'did not become ready'
        $launchPidFile = Get-ChildItem -LiteralPath (Join-Path $fixture 'logs/launch') -Recurse -Filter server.pid
        $launchedPid = (Get-Content -LiteralPath $launchPidFile.FullName -Raw).Trim()
        $ErrorActionPreference = 'Continue'
        & (Get-Command wsl.exe).Source --distribution Ubuntu --exec /bin/kill -0 $launchedPid 2>$null
        $probeExitCode = $LASTEXITCODE
        $ErrorActionPreference = 'Stop'
        if ($probeExitCode -eq 0) { throw 'FAIL: startup timeout left the sleeping server running.' }
        Write-Host 'PASS: WSL arguments, process group identity, and startup timeout cleanup (no GPU).'

        [IO.File]::WriteAllText((Join-Path $fixture 'build/apps/ninfer-serve'), "#!/bin/bash`nprintf 'fixture startup failure\n' >&2`nexit 7`n")
        Assert-Throws { Invoke-NInferLauncher -Plan $timeoutPlan -Distribution Ubuntu -NoBrowser -StartupTimeoutSeconds 5 } 'code 7.*fixture startup failure'
        Write-Host 'PASS: startup failure preserves the server exit code and stderr.'

        # A real hidden -File owner must release a ready server when forcibly terminated.
        $httpFixture = Join-Path $fixture 'owned http server.py'
        [IO.File]::WriteAllText($httpFixture, @'
import http.server, json, pathlib, signal, sys, threading, time
class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'healthy')
        if self.path == '/exit':
            threading.Thread(target=server.shutdown, daemon=True).start()
    def log_message(self, *args): pass
port = int(sys.argv[sys.argv.index('--port') + 1])
pathlib.Path(__file__).with_suffix('.arguments.json').write_text(json.dumps(sys.argv[1:]))
# Deliberately require the lifetime guard's KILL fallback after TERM.
signal.signal(signal.SIGTERM, signal.SIG_IGN)
if pathlib.Path(__file__).with_suffix('.delay').exists(): time.sleep(30)
print('fixture server log line', file=sys.stderr, flush=True)
server = http.server.HTTPServer(('127.0.0.1', port), Handler)
try: server.serve_forever()
finally: server.server_close()
'@)
        $pythonPath = (ConvertTo-NInferWslPath $httpFixture).Replace("'", "'\''")
        [IO.File]::WriteAllText((Join-Path $fixture 'build/apps/ninfer-serve'), "#!/bin/bash`nexec /usr/bin/python3 '$pythonPath' " + '"$@"' + "`n")
        foreach ($exitMode in @('forced-ready', 'normal-ready', 'forced-startup')) {
        if ($exitMode -eq 'forced-startup') { Set-Content -LiteralPath ([IO.Path]::ChangeExtension($httpFixture, '.delay')) -Value '' }
        $existingPidFiles = @(Get-ChildItem -LiteralPath (Join-Path $fixture 'logs/launch') -Recurse -Filter server.pid | ForEach-Object FullName)
        $listener.Start()
        $testPort = $listener.LocalEndpoint.Port
        $listener.Stop()
        $ownerOut = Join-Path $fixture 'owner.stdout.log'
        $ownerErr = Join-Path $fixture 'owner.stderr.log'
        $ownerArguments = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $launcher, '-Root', $fixture, '-Port', "$testPort", '-NoBrowser', '-StartupTimeoutSeconds', '30')
        if ($exitMode -eq 'normal-ready') { $ownerArguments += @('-Spec', 'mtp', '-DraftTokens', '3', '-FastPrefill', '-Temperature', '0.7', '-Thinking') }
        if ($exitMode -eq 'forced-startup') { $ownerArguments += @('-Spec', 'none') }
        $owner = $null
        $ownedWsl = @()
        $ownedLinuxPid = ''
        try {
            $owner = Start-Process -FilePath $interpreter -ArgumentList (ConvertTo-NInferWindowsArguments $ownerArguments) -WindowStyle Hidden -PassThru -RedirectStandardOutput $ownerOut -RedirectStandardError $ownerErr
            $owner.Handle | Out-Null
            $healthUrl = "http://127.0.0.1:$testPort/health"
            $ready = $false
            $deadline = (Get-Date).AddSeconds(40)
            while ((Get-Date) -lt $deadline) {
                if ($owner.HasExited) { throw "FAIL: external launcher exited: $(Get-Content -LiteralPath $ownerErr -Raw)" }
                $currentPidFile = Get-ChildItem -LiteralPath (Join-Path $fixture 'logs/launch') -Recurse -Filter server.pid |
                    Where-Object { $_.FullName -notin $existingPidFiles } | Select-Object -First 1
                if ($exitMode -eq 'forced-startup' -and $currentPidFile) { $ready = $true; break }
                try {
                    $response = Invoke-WebRequest -UseBasicParsing -Uri $healthUrl -TimeoutSec 1 -ErrorAction Stop
                    if ($response.StatusCode -eq 200 -and $currentPidFile -and
                        (Get-Content -LiteralPath $ownerOut -Raw) -match 'Bonsai 2 is ready') {
                        $ready = $true; break
                    }
                } catch { }
                Start-Sleep -Milliseconds 200
            }
            if (-not $ready) { throw 'FAIL: fake HTTP server did not become ready.' }
            $ownedWsl = @(Get-CimInstance Win32_Process -Filter "ParentProcessId=$($owner.Id) AND Name='wsl.exe'" | ForEach-Object { $_.ProcessId })
            if ($ownedWsl.Count -ne 1) { throw "FAIL: expected one owned WSL process, got $($ownedWsl.Count)." }
            $ownedLinuxPid = (Get-Content -LiteralPath $currentPidFile.FullName -Raw).Trim()
            if ($exitMode -ne 'forced-startup') {
                # The launcher's console shows the server log as it is written.
                $logDeadline = (Get-Date).AddSeconds(10)
                while ((Get-Content -LiteralPath $ownerOut -Raw) -notmatch 'fixture server log line' -and (Get-Date) -lt $logDeadline) {
                    Start-Sleep -Milliseconds 200
                }
                Assert-Equal ((Get-Content -LiteralPath $ownerOut -Raw) -match 'fixture server log line') $true 'launcher console mirrors the server log'
                $actualServerArguments = Get-Content -LiteralPath ([IO.Path]::ChangeExtension($httpFixture, '.arguments.json')) -Raw | ConvertFrom-Json
                $contextIndex = [array]::IndexOf($actualServerArguments, '--max-context')
                Assert-Equal $actualServerArguments[$contextIndex + 1] '65536' 'external script forwards the documented default context to the WSL server'
                Assert-Equal ($actualServerArguments -contains '--fast-prefill-kernel') ($exitMode -eq 'normal-ready') 'external script forwards optional fast prefill to the actual WSL server'
                $temperatureIndex = [array]::IndexOf($actualServerArguments, '--temperature')
                $expectedTemperature = if ($exitMode -eq 'normal-ready') { '0.7' } else { '0' }
                Assert-Equal $actualServerArguments[$temperatureIndex + 1] $expectedTemperature 'external script forwards startup temperature to the WSL server'
                Assert-Equal ($actualServerArguments -contains '--no-thinking') ($exitMode -ne 'normal-ready') 'external script forwards thinking selection to the WSL server'
                Assert-Equal ($actualServerArguments -contains '--greedy') $false 'external script allows per-request sampling overrides'
                $specIndex = [array]::IndexOf($actualServerArguments, '--spec')
                $draftIndex = [array]::IndexOf($actualServerArguments, '--draft-tokens')
                $expectedSpec = if ($exitMode -eq 'normal-ready') { 'mtp' } else { 'dflash2' }
                $expectedDrafts = if ($exitMode -eq 'normal-ready') { '3' } else { '7' }
                Assert-Equal $actualServerArguments[$specIndex + 1] $expectedSpec 'external script forwards default or explicit speculative backend'
                Assert-Equal $actualServerArguments[$draftIndex + 1] $expectedDrafts 'external script forwards default or explicit draft count'
            }
            if ($exitMode -eq 'normal-ready') {
                Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:$testPort/exit" -TimeoutSec 2 | Out-Null
                if (-not $owner.WaitForExit(10000)) { throw 'FAIL: launcher did not exit after normal server shutdown.' }
                Assert-Equal $owner.ExitCode 0 'normal server exit is preserved by the external launcher'
            } else {
                Stop-Process -Id $owner.Id -Force
                $owner.WaitForExit()
            }
            $deadline = (Get-Date).AddSeconds(10)
            do {
                $stillHealthy = $false
                try { $stillHealthy = (Invoke-WebRequest -UseBasicParsing -Uri $healthUrl -TimeoutSec 1 -ErrorAction Stop).StatusCode -eq 200 } catch { }
                if (-not $stillHealthy) { break }
                Start-Sleep -Milliseconds 200
            } while ((Get-Date) -lt $deadline)
            if ($stillHealthy) {
                $remainingWsl = @($ownedWsl | Where-Object { Get-Process -Id $_ -ErrorAction SilentlyContinue })
                throw "FAIL: forced launcher termination left the HTTP server running; remaining WSL clients: $($remainingWsl -join ',')."
            }
            foreach ($childId in $ownedWsl) {
                if (Get-Process -Id $childId -ErrorAction SilentlyContinue) { throw 'FAIL: forced launcher termination left its WSL child running.' }
            }
            # HTTP is absent during startup already, so independently wait for the whole group.
            $deadline = (Get-Date).AddSeconds(10)
            do {
                $ErrorActionPreference = 'Continue'
                & (Get-Command wsl.exe).Source --distribution Ubuntu --exec /bin/kill -0 -- "-$ownedLinuxPid" 2>$null
                $probeExitCode = $LASTEXITCODE
                $ErrorActionPreference = 'Stop'
                if ($probeExitCode -ne 0) { break }
                Start-Sleep -Milliseconds 200
            } while ((Get-Date) -lt $deadline)
            if ($probeExitCode -eq 0) { throw 'FAIL: forced launcher termination left its Linux process running.' }
            Write-Host "PASS: $exitMode releases the external PowerShell launcher's WSL child and Linux server group (no GPU)."
        } finally {
            if ($owner) {
                $ownedWsl += @(Get-CimInstance Win32_Process -Filter "ParentProcessId=$($owner.Id) AND Name='wsl.exe'" | ForEach-Object { $_.ProcessId })
            }
            if (-not $ownedLinuxPid) {
                $cleanupPidFile = Get-ChildItem -LiteralPath (Join-Path $fixture 'logs/launch') -Recurse -Filter server.pid |
                    Where-Object { $_.FullName -notin $existingPidFiles } | Select-Object -First 1
                if ($cleanupPidFile) { $ownedLinuxPid = (Get-Content -LiteralPath $cleanupPidFile.FullName -Raw).Trim() }
            }
            if ($owner -and -not $owner.HasExited) { Stop-Process -Id $owner.Id -Force -ErrorAction SilentlyContinue }
            foreach ($childId in $ownedWsl) { Stop-Process -Id $childId -Force -ErrorAction SilentlyContinue }
            if ($ownedLinuxPid -match '^[1-9][0-9]*$') {
                try { & (Get-Command wsl.exe).Source --distribution Ubuntu --exec /bin/kill -KILL -- "-$ownedLinuxPid" 2>&1 | Out-Null } catch { }
            }
            if ($owner) { $owner.Dispose() }
        }
        }
    }
} finally {
    # The generated fixture is exclusively owned by this test and remains under TEMP.
    $resolvedFixture = [IO.Path]::GetFullPath($fixture)
    $resolvedTemp = [IO.Path]::GetFullPath($env:TEMP).TrimEnd('\') + '\'
    if ($resolvedFixture.StartsWith($resolvedTemp, [StringComparison]::OrdinalIgnoreCase)) {
        Remove-Item -LiteralPath $resolvedFixture -Recurse -Force
    }
}
