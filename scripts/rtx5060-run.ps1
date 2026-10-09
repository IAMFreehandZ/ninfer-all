<#
.SYNOPSIS
Runs Ternary Bonsai 2 27B on the RTX 5060 Ti through Ubuntu WSL2.
.DESCRIPTION
Keep this terminal open while using the app; it shows the server log, which is also saved under logs\launch. Press Ctrl+C to stop the server.
Requires the existing WSL build and model; launch does no setup or downloads.
.PARAMETER Concurrency
Requests served at the same time, 1 to 8. Default 2. They share one KV pool of Concurrency times Context tokens, at most 262144.
.PARAMETER FastPrefill
Opts into the alternate prefill kernel, which can improve prefill throughput and reduce decode throughput.
.PARAMETER Temperature
Startup sampling temperature from 0 to 2. Default 0 selects greedy sampling; requests can override it.
.PARAMETER Thinking
Enable model thinking by default. Thinking can delay the visible answer; requests can override it.
.EXAMPLE
.\scripts\rtx5060-run.ps1 -Context 32768 -Spec none -NoBrowser
#>
[CmdletBinding()]
param(
    [string]$Root = '',
    [ValidateRange(1, 262144)][int]$Context = 65536,
    [ValidateRange(1, 8)][int]$Concurrency = 2,
    [ValidateSet('none', 'mtp', 'dflash2')][string]$Spec = 'dflash2',
    [ValidateRange(1, 15)][int]$DraftTokens = 7,
    [ValidateRange(1, 65535)][int]$Port = 8080,
    [string]$Distribution = 'Ubuntu',
    [string]$Model = '',
    [string]$Server = '',
    [ValidateRange(0.0, 2.0)][ValidateScript({ -not [double]::IsNaN($_) })][double]$Temperature = 0,
    [switch]$Thinking,
    [switch]$FastPrefill,
    [switch]$NoBrowser,
    [ValidateRange(1, 3600)][int]$StartupTimeoutSeconds = 600
)

# Windows PowerShell binds script parameters before populating PSScriptRoot.
if (-not $Root) { $Root = Split-Path -Parent $PSScriptRoot }

function ConvertTo-NInferWslPath([string]$Path, [string]$Distribution = 'Ubuntu') {
    $absolute = [IO.Path]::GetFullPath($Path)
    if ($absolute -match '^\\\\(?:wsl\.localhost|wsl\$)\\([^\\]+)(\\.*)$') {
        if ($Matches[1] -ne $Distribution) {
            throw "WSL path belongs to distribution '$($Matches[1])'; selected distribution is '$Distribution'."
        }
        return $Matches[2].Replace('\', '/')
    }
    if ($absolute -notmatch '^([A-Za-z]):\\(.*)$') {
        throw "Use a local Windows drive or WSL share path for WSL launch: $Path"
    }
    return '/mnt/' + $Matches[1].ToLowerInvariant() + '/' + $Matches[2].Replace('\', '/')
}

function ConvertTo-NInferWindowsArguments([string[]]$Values) {
    # Quote the CreateProcess argument vector, including embedded quotes and trailing slashes.
    return (($Values | ForEach-Object {
        if ($_ -and $_ -notmatch '[\s"]') { $_ } else {
            $escaped = [regex]::Replace($_, '(\\*)"', '$1$1\"')
            $escaped = [regex]::Replace($escaped, '(\\+)$', '$1$1')
            '"' + $escaped + '"'
        }
    }) -join ' ')
}

function New-NInferProcessJob {
    # The job owns the Windows WSL client. A separate stdin lifetime pipe makes the Linux
    # supervisor stop its server when this Windows owner or its client disappears.
    if (-not ('NInferLauncher.ProcessJob' -as [type])) {
        Add-Type -TypeDefinition @'
using System;
using System.ComponentModel;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Win32.SafeHandles;

namespace NInferLauncher {
    public sealed class ProcessJob : IDisposable {
        private SafeFileHandle handle;
        private FileStream stdout, stderr;
        private Task stdoutCopy, stderrCopy;
        private readonly object consoleLock = new object();

        [StructLayout(LayoutKind.Sequential)]
        private struct BasicLimits {
            public long PerProcessUserTimeLimit, PerJobUserTimeLimit;
            public uint LimitFlags;
            public UIntPtr MinimumWorkingSetSize, MaximumWorkingSetSize;
            public uint ActiveProcessLimit;
            public UIntPtr Affinity;
            public uint PriorityClass, SchedulingClass;
        }
        [StructLayout(LayoutKind.Sequential)]
        private struct IoCounters {
            public ulong ReadOperationCount, WriteOperationCount, OtherOperationCount;
            public ulong ReadTransferCount, WriteTransferCount, OtherTransferCount;
        }
        [StructLayout(LayoutKind.Sequential)]
        private struct ExtendedLimits {
            public BasicLimits Basic;
            public IoCounters Io;
            public UIntPtr ProcessMemoryLimit, JobMemoryLimit;
            public UIntPtr PeakProcessMemoryUsed, PeakJobMemoryUsed;
        }
        [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
        private static extern SafeFileHandle CreateJobObject(IntPtr attributes, string name);
        [DllImport("kernel32.dll", SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        private static extern bool SetInformationJobObject(SafeFileHandle job, int infoClass,
                                                           ref ExtendedLimits info, uint size);
        [DllImport("kernel32.dll", SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        private static extern bool AssignProcessToJobObject(SafeFileHandle job, IntPtr process);

        public ProcessJob() {
            handle = CreateJobObject(IntPtr.Zero, null);
            if (handle.IsInvalid) {
                int error = Marshal.GetLastWin32Error();
                handle.Dispose();
                throw new Win32Exception(error, "Cannot create the NInfer process ownership job.");
            }
            var limits = new ExtendedLimits();
            limits.Basic.LimitFlags = 0x2000; // JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            if (!SetInformationJobObject(handle, 9, ref limits,
                                         (uint)Marshal.SizeOf(typeof(ExtendedLimits)))) {
                int error = Marshal.GetLastWin32Error();
                handle.Dispose();
                throw new Win32Exception(error, "Cannot configure the NInfer process ownership job.");
            }
        }
        public void Assign(Process process) {
            if (!AssignProcessToJobObject(handle, process.Handle)) {
                throw new Win32Exception(Marshal.GetLastWin32Error(),
                                         "Cannot assign WSL to the NInfer process ownership job.");
            }
        }
        public Process Start(string executable, string arguments, string output, string error) {
            stdout = new FileStream(output, FileMode.Create, FileAccess.Write, FileShare.Read, 1, true);
            stderr = new FileStream(error, FileMode.Create, FileAccess.Write, FileShare.Read, 1, true);
            var process = new Process();
            process.StartInfo = new ProcessStartInfo(executable, arguments) {
                UseShellExecute = false, CreateNoWindow = true,
                WindowStyle = ProcessWindowStyle.Hidden,
                RedirectStandardInput = true, RedirectStandardOutput = true,
                RedirectStandardError = true
            };
            bool started = false;
            try {
                process.Start();
                started = true;
                Assign(process);
                stdoutCopy = Mirror(process.StandardOutput.BaseStream, stdout);
                stderrCopy = Mirror(process.StandardError.BaseStream, stderr);
                process.StandardInput.NewLine = "\n";
                process.StandardInput.WriteLine("start");
                process.StandardInput.Flush();
                return process;
            } catch {
                // No handshake was sent if assignment failed, so Linux cannot launch a server.
                if (started && !process.HasExited) { process.Kill(); process.WaitForExit(); }
                process.Dispose();
                throw;
            }
        }
        // Writes a server stream to its log file and to this console as it arrives, so the
        // terminal shows the server log live.
        private async Task Mirror(Stream source, FileStream log) {
            var buffer = new byte[8192];
            var decoder = new UTF8Encoding(false).GetDecoder();
            var text = new char[Encoding.UTF8.GetMaxCharCount(buffer.Length)];
            int read;
            while ((read = await source.ReadAsync(buffer, 0, buffer.Length).ConfigureAwait(false)) > 0) {
                await log.WriteAsync(buffer, 0, read).ConfigureAwait(false);
                int count = decoder.GetChars(buffer, 0, read, text, 0);
                lock (consoleLock) {
                    Console.Out.Write(text, 0, count);
                    Console.Out.Flush();
                }
            }
        }
        public void FlushLogs() {
            if (stdoutCopy != null) stdoutCopy.Wait();
            if (stderrCopy != null) stderrCopy.Wait();
            if (stdout != null) stdout.Flush();
            if (stderr != null) stderr.Flush();
        }
        public void Dispose() {
            handle.Dispose();
            try { FlushLogs(); } finally {
                if (stdout != null) stdout.Dispose();
                if (stderr != null) stderr.Dispose();
            }
        }
    }
}
'@
    }
    return New-Object NInferLauncher.ProcessJob
}

function Get-NInferLaunchPlan {
    [CmdletBinding()]
    param(
        [string]$Root = '',
        [ValidateRange(1, 262144)][int]$Context = 65536,
        [ValidateRange(1, 8)][int]$Concurrency = 2,
        [ValidateSet('none', 'mtp', 'dflash2')][string]$Spec = 'dflash2',
        [ValidateRange(1, 15)][int]$DraftTokens = 7,
        [ValidateRange(1, 65535)][int]$Port = 8080,
        [string]$Distribution = 'Ubuntu',
        [string]$Model = '',
        [string]$Server = '',
        [ValidateRange(0.0, 2.0)][ValidateScript({ -not [double]::IsNaN($_) })][double]$Temperature = 0,
        [switch]$Thinking,
        [switch]$FastPrefill
    )
    if (-not $Root) { $Root = Split-Path -Parent $PSScriptRoot }
    $rootPath = [IO.Path]::GetFullPath($Root)
    if (-not $Server) { $Server = Join-Path $rootPath 'build/apps/ninfer-serve' }
    if (-not $Model) { $Model = Join-Path $rootPath 'models/bonsai2/Ternary-Bonsai-2-27B-ninfer-v3.ninfer' }
    if (-not [IO.Path]::IsPathRooted($Server)) { $Server = Join-Path $rootPath $Server }
    if (-not [IO.Path]::IsPathRooted($Model)) { $Model = Join-Path $rootPath $Model }
    if (-not (Test-Path -LiteralPath $Server -PathType Leaf)) {
        throw "NInfer server is missing: $Server. Build the RTX 5060 Ti WSL target first; see README.md."
    }
    if (-not (Test-Path -LiteralPath $Model -PathType Leaf)) {
        throw "Bonsai model is missing: $Model. Restore the model file or pass -Model with its exact path."
    }
    $modelPath = ConvertTo-NInferWslPath $Model -Distribution $Distribution
    $arguments = @($modelPath, '--host', '127.0.0.1', '--port', "$Port", '--model-id', 'bonsai2-27b',
        '--max-context', "$Context", '--max-concurrency', "$Concurrency", '--kv-dtype', 'rk4v4', '--gdn-state-fp16')
    # Concurrent requests share one KV pool. At most 262144 tokens keeps it within the measured
    # single-request footprint; requests that do not fit together wait in the queue.
    if ($Concurrency -gt 1) {
        $arguments += @('--kv-capacity', "$([Math]::Min([long]$Concurrency * $Context, 262144))", '--concurrent-prefill')
    }
    # Startup measured 32-41 MiB of CUDA graphs for one request and 77-85 MiB for two against a
    # computed 960 MiB per request, so a fixed 128 MiB per request returns that reservation to
    # the free memory left beside the KV pool.
    $arguments += @('--cuda-graph-allowance-mib', "$(128 * $Concurrency)")
    if ($Spec -ne 'none') {
        $arguments += @('--spec', $Spec, '--draft-tokens', "$DraftTokens", '--lm-head-draft')
    }
    $arguments += @('--temperature', $Temperature.ToString('G', [Globalization.CultureInfo]::InvariantCulture))
    if (-not $Thinking) { $arguments += '--no-thinking' }
    if ($FastPrefill) { $arguments += '--fast-prefill-kernel' }
    [pscustomobject]@{
        Root = $rootPath
        ServerPath = ConvertTo-NInferWslPath $Server -Distribution $Distribution
        ModelPath = $modelPath
        ServerArguments = $arguments
        Url = "http://localhost:$Port"
        HealthUrl = "http://127.0.0.1:$Port/health"
    }
}

function Invoke-NInferLauncher {
    param($Plan, [string]$Distribution, [switch]$NoBrowser, [int]$StartupTimeoutSeconds)
    $wsl = Get-Command wsl.exe -ErrorAction SilentlyContinue
    if (-not $wsl) { throw 'WSL is missing. Install WSL2 and Ubuntu, then follow the RTX 5060 Ti build instructions in README.md.' }
    $wrapper = Join-Path $Plan.Root 'scripts/rtx5060-run.sh'
    if (-not (Test-Path -LiteralPath $wrapper -PathType Leaf)) { throw "Launcher helper is missing: $wrapper" }
    $check = & $wsl.Source --distribution $Distribution --exec /usr/bin/test -x $Plan.ServerPath 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "Ubuntu WSL could not execute the server. Check 'wsl -l -v', use -Distribution for your Ubuntu name, and build the Linux server. $($check -join ' ')"
    }
    & $wsl.Source --distribution $Distribution --exec /usr/bin/test -x /usr/bin/setsid
    if ($LASTEXITCODE -ne 0) { throw 'Ubuntu is missing /usr/bin/setsid. Restore its util-linux package before launching.' }
    $connection = New-Object Net.Sockets.TcpClient
    try {
        $pending = $connection.BeginConnect('127.0.0.1', [int]([uri]$Plan.Url).Port, $null, $null)
        if ($pending.AsyncWaitHandle.WaitOne(500)) {
            try { $connection.EndConnect($pending) } catch { }
        }
        if ($connection.Connected) { throw "Port $(([uri]$Plan.Url).Port) is already in use. Stop the existing server or choose -Port." }
    } finally { $connection.Dispose() }

    $logDirectory = Join-Path $Plan.Root ('logs/launch/' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '-' + [guid]::NewGuid().ToString('N').Substring(0, 8))
    New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
    $pidFile = Join-Path $logDirectory 'server.pid'
    $stdout = Join-Path $logDirectory 'server.stdout.log'
    $stderr = Join-Path $logDirectory 'server.stderr.log'
    $arguments = @('--distribution', $Distribution, '--exec', '/usr/bin/setsid', '--wait', '/bin/bash',
        (ConvertTo-NInferWslPath $wrapper -Distribution $Distribution), (ConvertTo-NInferWslPath $pidFile -Distribution $Distribution), $Plan.ServerPath) + $Plan.ServerArguments
    $process = $null
    $job = $null
    try {
        $job = New-NInferProcessJob
        Write-Host "Starting Bonsai 2 at $($Plan.Url). The server log follows and is saved in $logDirectory"
        Write-Host 'Keep this terminal open. Press Ctrl+C to stop the server and release GPU memory.'
        $process = $job.Start($wsl.Source, (ConvertTo-NInferWindowsArguments $arguments), $stdout, $stderr)
        $deadline = (Get-Date).AddSeconds($StartupTimeoutSeconds)
        $ready = $false
        while ((Get-Date) -lt $deadline) {
            if ($process.HasExited) {
                $job.FlushLogs()
                $details = if (Test-Path -LiteralPath $stderr) { (Get-Content -LiteralPath $stderr -Tail 25) -join [Environment]::NewLine } else { '' }
                throw "NInfer exited during startup (code $($process.ExitCode)). $details"
            }
            try {
                $response = Invoke-WebRequest -UseBasicParsing -Uri $Plan.HealthUrl -TimeoutSec 2 -ErrorAction Stop
                if ($response.StatusCode -eq 200) { $ready = $true; break }
            } catch { }
            Start-Sleep -Milliseconds 500
        }
        if (-not $ready) { throw "NInfer did not become ready within $StartupTimeoutSeconds seconds. Inspect $stderr and $stdout." }
        Write-Host "Bonsai 2 is ready: $($Plan.Url)"
        if (-not $NoBrowser) { Start-Process $Plan.Url | Out-Null }
        while (-not $process.WaitForExit(500)) { }
        $job.FlushLogs()
        if ($process.ExitCode -ne 0) { throw "NInfer exited with code $($process.ExitCode). Inspect $stderr." }
    } finally {
        if ($null -ne $process -and -not $process.HasExited) {
            $process.StandardInput.Close()
            if (Test-Path -LiteralPath $pidFile) {
                $linuxPid = (Get-Content -LiteralPath $pidFile -Raw).Trim()
                if ($linuxPid -match '^[1-9][0-9]*$') {
                    # The server can exit between HasExited and kill; that race is harmless.
                    try { & $wsl.Source --distribution $Distribution --exec /bin/kill -TERM -- "-$linuxPid" 2>&1 | Out-Null } catch { }
                    if (-not $process.WaitForExit(10000)) {
                        try { & $wsl.Source --distribution $Distribution --exec /bin/kill -KILL -- "-$linuxPid" 2>&1 | Out-Null } catch { }
                    }
                }
            }
            if (-not $process.WaitForExit(3000)) { Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue }
        }
        if ($null -ne $job) { $job.Dispose() }
        if ($null -ne $process) { $process.Dispose() }
    }
}

if ($MyInvocation.InvocationName -ne '.') {
    $ErrorActionPreference = 'Stop'
    try {
        $plan = Get-NInferLaunchPlan -Root $Root -Context $Context -Concurrency $Concurrency -Spec $Spec -DraftTokens $DraftTokens -Port $Port -Distribution $Distribution -Model $Model -Server $Server -Temperature $Temperature -Thinking:$Thinking -FastPrefill:$FastPrefill
        Invoke-NInferLauncher -Plan $plan -Distribution $Distribution -NoBrowser:$NoBrowser -StartupTimeoutSeconds $StartupTimeoutSeconds
    } catch {
        Write-Error $_ -ErrorAction Continue
        exit 1
    }
}
