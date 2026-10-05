#!/usr/bin/env bash
# Internal WSL entry: stdin is the lifetime pipe held open by the Windows owner.
set -euo pipefail
if (( $# < 2 )); then
    printf 'usage: rtx5060-run.sh PID_FILE SERVER [arguments...]\n' >&2
    exit 2
fi
pid_file=$1
shift
if [[ ! -x $1 ]]; then
    printf 'NInfer server is missing or not executable: %s\n' "$1" >&2
    exit 1
fi
# Windows permits launch only after assigning its WSL client to the ownership job.
# A closed pipe before this handshake must never leave a newly started server.
IFS= read -r permission || exit 1
[[ $permission == start ]] || exit 2
exec 3<&0
/usr/bin/setsid -- "$@" 3<&- </dev/null &
server_pid=$!
printf '%s\n' "$server_pid" > "$pid_file"
(
    grace_pid=''
    stop_grace() {
        if [[ -n $grace_pid ]]; then
            /bin/kill -TERM "$grace_pid" 2>/dev/null || true
            wait "$grace_pid" 2>/dev/null || true
        fi
    }
    trap 'stop_grace; exit 0' TERM INT
    # This read blocks for the whole inference lifetime; it has no polling wakeups.
    while IFS= read -r ignored <&3; do :; done
    /bin/kill -TERM -- "-$server_pid" 2>/dev/null || true
    /bin/sleep 2 &
    grace_pid=$!
    wait "$grace_pid"
    /bin/kill -KILL -- "-$server_pid" 2>/dev/null || true
) &
guard_pid=$!
cleanup() {
    trap - EXIT
    /bin/kill -TERM "$guard_pid" 2>/dev/null || true
    wait "$guard_pid" 2>/dev/null || true
    # Remove any descendants still in the server's isolated group after it exits.
    /bin/kill -KILL -- "-$server_pid" 2>/dev/null || true
}
trap cleanup EXIT
if wait "$server_pid"; then exit 0; else exit "$?"; fi
