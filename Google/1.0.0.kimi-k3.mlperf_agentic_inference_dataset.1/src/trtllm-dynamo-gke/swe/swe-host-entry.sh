#!/bin/bash
# kimi-swe-host pod entry point: the Kimi-K3 SWE-bench service on GKE (1-node pilot).
#
# Runs NVIDIA's drop scripts byte-identical (swe-service.py, swe-enroot-transport.py,
# swe_cleanup.py) on the pinned endpoints c97ddfcce service. The only change from Slurm is
# the launcher: Enroot 4.2.1 is installed in this pod, and srun/squeue/scancel are local
# shims (./srun, ./squeue, ./scancel). The service runs as a dedicated unprivileged user that
# this script creates (SWE_SERVICE_USER, see "service user" below), like a Slurm user, not as
# root; this script stays root only to set up, forward SIGTERM and check the teardown.
#
# SWE_POD_MODE:
#   smoke    No GitLab token needed. Checks tools, egress and disk. Runs the drop's service
#            lifecycle (uv sync, start, SWE_READY, SIGTERM teardown) with a dummy registry,
#            and drives the drop-patched sandbox transport through the shims against a
#            public arm64 image (swe_probe.py smoke). Prints SMOKE_OK or SMOKE_FAIL.
#   service  The service for a pilot run. Stays up until the pod is deleted. On SIGTERM it
#            stops the service the way Slurm does, checks the teardown, and scans the run
#            directory for the service token and the registry token (counts only).
#
# Production (swe_pilot.py serve, one sflow run of the Kimi recipes; service mode only):
#   SWE_HANDOFF=1       The drop's connection file goes to $STATE/handoff/<run id>/ instead of
#                       the run directory: the recipe's benchmark task on this node reads it
#                       there (GKE port of the drop's SWE_CONNECTION artifact).
#   SWE_SCAN_REPORTS=D  D is the node's report directory, mounted read-only. The teardown leak
#                       scan also covers D/<run id>, the benchmark's report for this run.
#
# Metadata block (CS-NET-001, both modes): while the service runs, an iptables rule for the
# service user's uid keeps it, and so its sandboxes, away from the GCE and GKE metadata
# servers; root keeps its access. The uid is the dedicated service user's, which nothing else
# on the node uses (checked before the rule goes in). A self-test proves the block before the
# service starts (fail closed); the teardown re-checks and removes it.
#   SWE_METADATA_BLOCK=0  No block. Approved fallback only; logs a WARNING.
#
# Secrets: Enroot reads the registry token from a 0600 copy of the Secret on a
# memory-backed volume. Nothing here prints a token; checks report counts and file names.

set -euo pipefail
umask 022
# Root runs python3 here. Start from / so the image WORKDIR, which the service user may be
# able to write, is never a module search path; root python also uses -I (see probe()).
cd /
export HOME=/root      # never the service user's home (see AS_USER)

log() { printf '[swe-host %s] %s\n' "$(date -u +%H:%M:%SZ)" "$*"; }
die() { log "FATAL: $*"; exit 1; }

# ------------------------------------------------------------------- fixed layout
CM_HOST=/opt/kimi-swe/cm/host     # ConfigMap kimi-swe-host: our files, Enroot .deb, service tarball
CM_DROP=/opt/kimi-swe/cm/drop     # ConfigMap kimi-swe-drop-scripts: NVIDIA's three scripts
WORK=/opt/kimi-swe                # writable, this container only
PRIV=/run/kimi-swe                # memory-backed emptyDir
SECRET_DIR=/etc/kimi-swe/secret   # Secret kimi-swe-enroot (service mode only)
STATE=<node-local-ssd>/kimi-swe   # hostPath, node-local SSD
REVISION=c97ddfcce80392e9f30b86baabaa998355d4327e
REGISTRY_PATH=registry.gitlab.com/nvidia/mlperf-inference-partner/nv-mlpinf-partner/swebench-arm64

# Everything downloaded or taken from the drop is pinned. Any mismatch stops the pod.
declare -A PINNED=(
  [swe-service.py]=d6ca1795e4db77f608770498ccb8ccd4f34e7c18e954d1e6939b0a497be3967e
  [swe-enroot-transport.py]=4a427b88229d1583efa6fbb3072b4bb496cd27708f210b941496d522ec4ed242
  [swe_cleanup.py]=5491c3e02fad9c55b9abfde98ef80fa5d7aaf723a65cb2f7d773e82b7476d87f
  [enroot_4.2.1-1_arm64.deb]=267018b335815a6b32a562915de1f3fad579f7c9d6748e98c62e7369b40be421
  [enroot-caps_4.2.1-1_arm64.deb]=12b811d0c95e879e42e4ac762fa85869a5f3cc9dbb06ab483c01e12e911284a1
  [swe_source_c97.tgz]=674c82a5f098db8cfe045c8564fe5db953681841f845c4fb7d1a5e3ad22b4de9
)

MODE=${SWE_POD_MODE:-}
RUN_ID=${SWE_RUN_ID:-}
# The dedicated service user (see "service user" below). swe_pilot.py sets all three from its
# one constant (SERVICE_USER, SERVICE_UID, SERVICE_GID); there is no default here.
SVC_USER=${SWE_SERVICE_USER:-}
SVC_UID=${SWE_SERVICE_UID:-}
SVC_GID=${SWE_SERVICE_GID:-}
RUN_AS=${SWE_RUN_AS:-$SVC_USER}   # root: fallback only, without the metadata block
IMAGE_CACHE=${SWE_IMAGE_CACHE:-run}
PREWARM=${SWE_PREWARM:-}
SMOKE_IMAGE=${SWE_SMOKE_IMAGE:-mirror.gcr.io#library/debian:trixie-slim}
HANDOFF=${SWE_HANDOFF:-0}
SCAN_REPORTS=${SWE_SCAN_REPORTS:-}
METADATA_BLOCK=${SWE_METADATA_BLOCK:-1}

case "$MODE" in smoke|service) ;; *) die "SWE_POD_MODE must be smoke or service" ;; esac
[[ $RUN_ID =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$ ]] || die "SWE_RUN_ID must be 1-64 chars of [A-Za-z0-9._-]"
[[ $SVC_USER =~ ^[a-z_][a-z0-9_-]{0,31}$ && $SVC_UID =~ ^[1-9][0-9]{0,9}$ && $SVC_GID =~ ^[1-9][0-9]{0,9}$ ]] \
  || die "SWE_SERVICE_USER, SWE_SERVICE_UID and SWE_SERVICE_GID must be a user name and two numbers (swe_pilot.py sets them)"
[[ $RUN_AS == "$SVC_USER" || $RUN_AS == root ]] || die "SWE_RUN_AS must be $SVC_USER (the default) or root"
case "$IMAGE_CACHE" in run|persistent) ;; *) die "SWE_IMAGE_CACHE must be run or persistent" ;; esac
[[ -z $PREWARM || $PREWARM =~ ^(first:[0-9]{1,3}|[A-Za-z0-9_.,-]+)$ ]] \
  || die "SWE_PREWARM must be first:N or a comma list of instance ids"
[[ -z $PREWARM || ( $MODE == service && $IMAGE_CACHE == persistent ) ]] \
  || die "SWE_PREWARM needs service mode and SWE_IMAGE_CACHE=persistent"
[[ $SMOKE_IMAGE =~ ^[A-Za-z0-9./#:_@-]+$ ]] || die "SWE_SMOKE_IMAGE has unexpected characters"
case "$HANDOFF" in 0|1) ;; *) die "SWE_HANDOFF must be 0 or 1" ;; esac
[[ $HANDOFF == 0 || $MODE == service ]] || die "SWE_HANDOFF needs service mode"
[[ -z $SCAN_REPORTS || ( $SCAN_REPORTS =~ ^/[A-Za-z0-9._/-]+$ && $SCAN_REPORTS != *..* ) ]] \
  || die "SWE_SCAN_REPORTS must be an absolute path"
case "$METADATA_BLOCK" in 0|1) ;; *) die "SWE_METADATA_BLOCK must be 0 or 1" ;; esac
[[ $(id -u) == 0 ]] || die "the entry script must start as root; it runs the service as $RUN_AS"
[[ $(uname -m) == aarch64 ]] || die "arm64 node required"

# SIGTERM (pod deletion) only sets a flag; long steps run in the background and are
# stopped in order, so the drop's own teardown always runs.
TERM_SEEN=0
trap 'TERM_SEEN=1' TERM INT

sha_ok() { [[ $(sha256sum "$1" | cut -d' ' -f1) == "$2" ]]; }

# Root helpers (swe_probe.py conn/snapshot/leakscan/postcheck-report), stdlib only.
# -I: no script dir, cwd, user site or PYTHON* variables on the module path.
probe() { python3 -I "$WORK/lib/swe_probe.py" "$@"; }

# Run a command in the background and wait for it. On SIGTERM, forward it and wait.
run_interruptible() {
  "$@" &
  local pid=$! rc
  while :; do
    wait "$pid" && rc=0 || rc=$?
    kill -0 "$pid" 2>/dev/null || return "$rc"
    if [[ $TERM_SEEN == 1 ]]; then
      kill -TERM "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
      return 143
    fi
  done
}

# ------------------------------------------------------------------- setup
stage() {
  install -d -m 0755 "$WORK/stage" "$WORK/drop" "$WORK/bin" "$WORK/lib" "$WORK/src"
  # Copy first, then verify the copies: ConfigMap volumes can change under a running pod.
  cp -L "$CM_HOST"/* "$WORK/stage/"
  cp -L "$CM_DROP/swe-service.py" "$CM_DROP/swe-enroot-transport.py" "$CM_DROP/swe_cleanup.py" "$WORK/drop/"
  chmod 0644 "$WORK/drop/"*
  local f
  for f in swe-service.py swe-enroot-transport.py swe_cleanup.py; do
    sha_ok "$WORK/drop/$f" "${PINNED[$f]}" || die "sha256 mismatch: drop script $f"
  done
  for f in enroot_4.2.1-1_arm64.deb enroot-caps_4.2.1-1_arm64.deb swe_source_c97.tgz; do
    sha_ok "$WORK/stage/$f" "${PINNED[$f]}" || die "sha256 mismatch: $f"
  done
  (cd "$WORK/stage" && sha256sum --quiet --strict -c MANIFEST.sha256) \
    || die "host ConfigMap files do not match MANIFEST.sha256 (re-run: swe_pilot.py configmaps)"
  install -m 0755 "$WORK/stage/srun" "$WORK/stage/squeue" "$WORK/stage/scancel" "$WORK/bin/"
  install -m 0644 "$WORK/stage/swe_probe.py" "$WORK/lib/"
  tar -xzf "$WORK/stage/swe_source_c97.tgz" -C "$WORK/src" --no-same-owner --no-same-permissions
  [[ $(< "$WORK/src/swebench_service/REVISION") == "$REVISION" ]] || die "service tarball REVISION mismatch"
  log "staged: drop scripts, Enroot .debs and service source $REVISION match their sha256 pins"
}

install_enroot() {
  export DEBIAN_FRONTEND=noninteractive
  # hostPath $STATE/tmp is mounted as /tmp and kubelet creates it 0755 root. Give it the mode
  # of a normal /tmp first: apt verifies signatures as its unprivileged _apt user in /tmp.
  chmod 1777 /tmp
  log "installing Enroot 4.2.1 and enroot+caps from the pinned .debs, plus their Debian dependencies and iptables"
  apt-get update -qq >/dev/null
  # enroot+caps (as on Slurm/Pyxis nodes): its postinst gives enroot-aufs2ovlfs
  # cap_sys_admin,cap_mknod+pe and enroot-mksquashovlfs cap_sys_admin+pe, so the non-root
  # service user can convert image layers. Without it, `enroot import` fails at
  # "Converting whiteouts" with "failed to set capabilities".
  # iptables: the metadata block (see metadata_block_apply). Debian's package has both
  # backends (iptables-legacy, iptables-nft); like the other unpinned dependencies it is
  # verified by Debian's archive signatures.
  apt-get install -y -qq --no-install-recommends \
    "$WORK/stage/enroot_4.2.1-1_arm64.deb" "$WORK/stage/enroot-caps_4.2.1-1_arm64.deb" \
    ca-certificates util-linux iptables >/dev/null
  [[ $(enroot version) == 4.2.1 ]] || die "enroot version is not 4.2.1"
  local c
  for c in unshare flock setpriv mksquashfs unsquashfs jq parallel curl getcap \
           iptables-legacy iptables-nft ip6tables-legacy ip6tables-nft \
           iptables-legacy-save iptables-nft-save ip6tables-legacy-save ip6tables-nft-save; do
    command -v "$c" >/dev/null || die "missing $c after install"
  done
  caps_ok() {   # caps_ok FILE CAP...: FILE carries every CAP as permitted+effective file caps
    local out cap
    out=$(getcap "$1") || return 1
    for cap in "${@:2}"; do [[ $out == *"$cap"* ]] || return 1; done
    [[ $out == *"=ep"* || $out == *"+ep"* ]]
  }
  caps_ok "$(command -v enroot-aufs2ovlfs)" cap_sys_admin cap_mknod \
    || die "enroot-aufs2ovlfs lacks cap_sys_admin,cap_mknod (enroot+caps)"
  caps_ok "$(command -v enroot-mksquashovlfs)" cap_sys_admin \
    || die "enroot-mksquashovlfs lacks cap_sys_admin (enroot+caps)"
  cat > /etc/enroot/enroot.conf <<'EOF'
# kimi-swe-host (GKE). Enroot 4.2.1 defaults except for one setting.
# This pod is privileged, so its /dev holds every host device (GPUs, disks). Give sandboxes
# the minimal /dev that Docker gives containers instead: null, zero, full, random, urandom,
# tty, pts, shm, mqueue.
ENROOT_RESTRICT_DEV yes
EOF
  # With ENROOT_RESTRICT_DEV, Enroot's hooks.d/10-devices.sh bind-mounts the host's /dev/log
  # (the syslog socket journald provides on a Slurm node) into every container, without
  # nofail, so `enroot start` fails if it is missing. This pod runs no syslog daemon. Its /dev
  # is a container-local tmpfs (no hostPath), so add an empty, root-owned, read-only
  # placeholder: syslog() in a sandbox then fails as it does with no daemon running, and the
  # sandboxes' mapped users cannot write to it.
  if [[ ! -e /dev/log ]]; then
    rm -f /dev/log      # drop a dangling symlink, if any
    install -m 0444 /dev/null /dev/log
    log "no /dev/log in this pod: created an empty read-only placeholder for Enroot's device hook"
  fi
}

# >>> service user (CS-NET-001) >>>
# ------------------------------------------------------------------- service user
# The service, its sandboxes and the pilot client run as a dedicated user that this script
# creates in the pod: SWE_SERVICE_USER, SWE_SERVICE_UID and SWE_SERVICE_GID (swe_pilot.py's
# SERVICE_USER, SERVICE_UID and SERVICE_GID: kimiswe, 61000:61000). The metadata block matches
# its uid on the whole node, so nothing else may run as it. The image's own user, appuser, is
# uid 1000, and GKE's hostNetwork DaemonSets run containers as 1000 and 1001: a block for 1000
# would cut them off. So the user is new, and this fails closed if the image already uses its
# name, uid or gid in any other combination. It owns nothing in the image (appuser owns the
# venv, the WORKDIR and its home), and its home is a fresh 0700 directory in this container.
SVC_ETC=/etc                    # the account files (tests point it at a copy)
SVC_HOME=$WORK/home             # fresh in every container; never under the image's /home

svc_id_ok() {   # svc_id_ok N: 0 if N may be the service uid or gid
  # 60578-61183 is unused in systemd's uid map: above useradd's automatic range (UID_MAX 60000)
  # and below systemd's dynamic users (61184-65519), nonroot (65532) and nobody (65534). So no
  # image's useradd and no systemd on the node hands it out, and 1000 and 1001 are never in it.
  [[ $1 =~ ^[1-9][0-9]{0,9}$ ]] && (( $1 >= 60578 && $1 <= 61183 ))
}

svc_lines() {   # svc_lines passwd|group FIELD VALUE: the lines of that file whose FIELD is VALUE
  awk -F: -v f="$2" -v v="$3" '$f == v' "$SVC_ETC/$1"
}

service_user() {   # create SVC_USER, fail closed; sets RUN_UID, RUN_GID and RUN_HOME
  local name=$SVC_USER uid=$SVC_UID gid=$SVC_GID pw gr out how
  { svc_id_ok "$uid" && svc_id_ok "$gid"; } || die "service user: uid:gid $uid:$gid is not in" \
    "60578-61183 (unused in systemd's uid map; never 1000, 1001, 65532 or 65534)"
  # The name, the uid and the gid are each unused, or already exactly ours: one passwd line
  # name:uid:gid (no other user with that uid or that primary gid) and one group line name:gid
  # (no other group with that gid or with the name as a member).
  pw=$(awk -F: -v n="$name" -v u="$uid" -v g="$gid" '$1 == n || $3 == u || $4 == g { print $1 ":" $3 ":" $4 }' \
         "$SVC_ETC/passwd") || die "service user: cannot read $SVC_ETC/passwd"
  [[ -z $pw || $pw == "$name:$uid:$gid" ]] || die "service user: $SVC_ETC/passwd already uses the" \
    "name $name, uid $uid or gid $gid otherwise (name:uid:gid): $(tr '\n' ' ' <<<"$pw")"
  gr=$(awk -F: -v n="$name" -v g="$gid" '$1 == n || $3 == g || index("," $4 ",", "," n ",") { print $1 ":" $3 }' \
         "$SVC_ETC/group") || die "service user: cannot read $SVC_ETC/group"
  [[ -z $gr || $gr == "$name:$gid" ]] || die "service user: $SVC_ETC/group already uses the name" \
    "$name or gid $gid otherwise (name:gid): $(tr '\n' ' ' <<<"$gr")"
  if [[ -z $gr ]]; then
    out=$(groupadd -g "$gid" "$name" 2>&1) || die "service user: groupadd failed: $out"
  fi
  if [[ -z $pw ]]; then
    # -M: no home from the image's skeleton; -N: the group above. useradd warns that the uid is
    # above UID_MAX; that is the point (see svc_id_ok), so its output shows only on failure.
    out=$(useradd -u "$uid" -g "$gid" -M -N -d "$SVC_HOME" -s /usr/sbin/nologin \
            -c "kimi-swe SWE service" "$name" 2>&1) || die "service user: useradd failed: $out"
  fi
  [[ $(svc_lines passwd 3 "$uid" | cut -d: -f1,3,4) == "$name:$uid:$gid" \
     && $(svc_lines group 3 "$gid" | cut -d: -f1,3) == "$name:$gid" ]] \
    || die "service user: $SVC_ETC/passwd and $SVC_ETC/group do not show exactly $name ($uid:$gid)"
  [[ ! -e $SVC_HOME && ! -L $SVC_HOME ]] || die "service user: home $SVC_HOME exists already; it must be fresh"
  install -d -m 0700 -o "$uid" -g "$gid" "$SVC_HOME"
  RUN_UID=$uid RUN_GID=$gid RUN_HOME=$SVC_HOME
  how="created in this container"
  if [[ -n $pw ]]; then how="already in the image with exactly these ids"; fi
  log "service user $name ($uid:$gid): $how; home $SVC_HOME (new, 0700); no supplementary groups"
}
# <<< service user <<<

prepare() {
  local foreign
  if [[ $RUN_AS == root ]]; then
    RUN_UID=0 RUN_GID=0 RUN_HOME=/root
  else
    service_user
  fi
  [[ $(id -u "$RUN_AS") == "$RUN_UID" && $(id -g "$RUN_AS") == "$RUN_GID" ]] \
    || die "id $RUN_AS does not report uid:gid $RUN_UID:$RUN_GID"
  # The service user's identity variables go only to its processes. Root keeps its own HOME,
  # so no root tool (curl, python) reads config from a directory the service user can write.
  # setpriv: no supplementary groups, no inheritable capabilities.
  AS_USER=(env "HOME=$RUN_HOME" "USER=$RUN_AS" "LOGNAME=$RUN_AS"
           setpriv "--reuid=$RUN_UID" "--regid=$RUN_GID" --clear-groups --inh-caps=-all --)
  if [[ $RUN_UID == 0 ]]; then
    log "WARNING: SWE_RUN_AS=root. Sandboxes then run as real root (remapped). Fallback only."
  fi
  install -d -m 0755 "$STATE" "$STATE/runs"
  RUN="$STATE/runs/$RUN_ID"
  [[ ! -e $RUN ]] || die "run directory exists already: $RUN"
  install -d -m 0700 -o "$RUN_UID" -g "$RUN_GID" "$RUN"
  SVC_OUT="$RUN/swe_service"
  # The drop expects the connection file to exist; it chmods it 0600 and writes {url, token}.
  if [[ $HANDOFF == 1 ]]; then
    # Production: the recipe's benchmark task on this node (root, hostPath artifact
    # SWE_HANDOFF) reads it from here. Root-owned directories that others cannot list; the
    # file itself belongs to the service user, as the drop needs (it chmods and rewrites it).
    install -d -m 0755 "$STATE/handoff"
    [[ ! -e $STATE/handoff/$RUN_ID ]] || die "handoff directory exists already: $STATE/handoff/$RUN_ID"
    install -d -m 0711 "$STATE/handoff/$RUN_ID"
    CONN="$STATE/handoff/$RUN_ID/swe-connection.json"
  else
    CONN="$RUN/swe-connection.json"
  fi
  install -m 0600 -o "$RUN_UID" -g "$RUN_GID" /dev/null "$CONN"
  printf '{}' > "$CONN"
  chmod 0711 "$PRIV"
  install -d -m 0700 "$PRIV/root"
  install -d -m 0700 -o "$RUN_UID" -g "$RUN_GID" "$PRIV/enroot"
  if [[ $IMAGE_CACHE == persistent ]]; then
    # Optional: keep imported images across runs. Same cache keys; only the location moves.
    # Reused only if the service user already owns every entry. Never chowned: images that
    # another uid wrote (for example appuser, uid 1000, before the dedicated user) are not
    # trusted, so such a cache must be deleted instead.
    if [[ -e $STATE/enroot-cache || -L $STATE/enroot-cache ]]; then
      [[ -d $STATE/enroot-cache && ! -L $STATE/enroot-cache ]] || die "$STATE/enroot-cache is not a directory"
      foreign=$(find "$STATE/enroot-cache" -xdev \( ! -uid "$RUN_UID" -o ! -gid "$RUN_GID" \) -printf . | wc -c) \
        || die "cannot check who owns the entries of $STATE/enroot-cache"
      [[ $foreign == 0 ]] || die "the persistent image cache $STATE/enroot-cache has $foreign entries" \
        "that $RUN_AS ($RUN_UID:$RUN_GID) does not own, for example from runs as uid 1000. Delete it" \
        "on the node, or run without --persist-cache"
    fi
    install -d -m 0700 -o "$RUN_UID" -g "$RUN_GID" "$STATE/enroot-cache" "$SVC_OUT"
    ln -s "$STATE/enroot-cache" "$SVC_OUT/enroot-cache"
    chown -h "$RUN_UID:$RUN_GID" "$SVC_OUT/enroot-cache"
  fi
  # Allocation identity. The drop needs SLURM_JOB_ID and SLURMD_NODENAME; the pinned
  # runtime does not pass them to srun, so the shim reads the expected values from here.
  NODE=$(hostname -s)
  [[ $NODE =~ ^[a-z0-9]([a-z0-9-]*[a-z0-9])?$ ]] || die "unexpected node name"
  export SLURM_JOB_ID=1 SLURMD_NODENAME=$NODE
  printf '{"jobid": "%s", "node": "%s"}\n' "$SLURM_JOB_ID" "$NODE" > "$WORK/shim.json"
  chmod 0644 "$WORK/shim.json"
  log "run $RUN_ID on $NODE: service user $RUN_AS ($RUN_UID:$RUN_GID), image cache: $IMAGE_CACHE"
}

load_credentials() {
  local src="$SECRET_DIR/.credentials" login
  [[ -s $src ]] || die "Secret kimi-swe-enroot is missing or empty (run: swe_pilot.py secret)"
  login=$(awk '$1 == "machine" && $2 == "registry.gitlab.com" && $3 == "login" { print $4; exit }' "$src")
  [[ $login =~ ^[A-Za-z0-9._-]+$ ]] \
    || die ".credentials has no valid 'machine registry.gitlab.com login USER password TOKEN' line"
  grep -q '^machine[[:space:]]\+gitlab\.com[[:space:]]' "$src" \
    || log "WARNING: .credentials has no gitlab.com line; the registry's token realm is gitlab.com"
  install -m 0600 -o "$RUN_UID" -g "$RUN_GID" "$src" "$PRIV/enroot/.credentials"
  REGISTRY="${login}@${REGISTRY_PATH}"
  log "registry: $REGISTRY (token not shown)"
}

service_env() {
  # Mirror the Slurm host step: no client venv, no inherited Python or uv settings.
  unset PYTHONPATH PYTHONHOME UV_PROJECT_ENVIRONMENT UV_NO_SYNC UV_PYTHON_INSTALL_DIR \
        UV_COMPILE_BYTECODE UV_LINK_MODE VBENCH_PROJECT_PATH VIRTUAL_ENV
  export PATH="$WORK/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
  # HOME, USER and LOGNAME for the service user are set by AS_USER (see prepare).
  export BENCHMARK_MODE=acc
  export SWE_SOURCE="$WORK/src/swebench_service"
  SWE_UV=$(command -v uv) || die "uv not found in the image"
  export SWE_UV
  export SWE_IMAGE_REGISTRY="$REGISTRY"
  export SWE_ENROOT_CONFIG_PATH="$PRIV/enroot"
  export SWE_CONNECTION="$CONN"
  export SFLOW_WORKFLOW_OUTPUT_DIR="$RUN" SFLOW_TASK_OUTPUT_DIR="$SVC_OUT"
}

# >>> metadata block (CS-NET-001) >>>
# ------------------------------------------------------------------- metadata block
# The service and the Enroot sandboxes it starts run as $RUN_UID (a sandbox's root is that uid
# on the node) in the node's network namespace (hostNetwork). Without a block they could ask
# the GCE metadata server for the node service account's token. While the service runs, our
# own filter-table chain KIMI-SWE-MDBLOCK refuses that uid's traffic to the metadata server
# <ip> (every port but DNS, 53), to GKE's metadata server <ip> and, where
# ip6tables works, to fd20:ce::254. Its one jump, at the top of OUTPUT, matches only that uid:
# root (kubelet, GKE agents, this script) and every other uid keep their access. The uid is the
# dedicated service user's (see "service user"), and before the rules go in no socket in the
# node's network namespace may belong to it yet, so nothing else on the node is cut off.
# Nothing else on the node changes: no table, chain or rule that is not ours is ever flushed or
# deleted. The rules go to the backend the node uses (the one with more rules, as in
# kube-proxy's iptables-wrapper); -w waits for the host's xtables lock, which swe_pilot.py mounts.
# A self-test proves the block before the service starts (fail closed). The teardown re-checks
# and removes it; an EXIT trap removes it if the script ends early. If the pod is SIGKILLed,
# the next host pod on the node removes it (until then it only restricts that uid).
MDB_CHAIN=KIMI-SWE-MDBLOCK
MDB_COMMENT="kimi-swe-host: GCE metadata block for the SWE service user"
MDB_IP4=<ip>                    # GCE metadata server; also the VPC resolver (53)
MDB_GKE4=<ip>                   # GKE metadata server (Workload Identity), port 988
MDB_IP6=fd20:ce::254                       # GCE metadata server over IPv6 (dual-stack nodes)
MDB_PATH=/computeMetadata/v1/instance/id   # the only metadata path the self-test requests
MDB_PROC_NET=/proc/net                     # the node's socket tables (this pod uses its network)
MDB_BACKEND=                               # legacy or nft
MDB_V6_TOOL=0                              # 1: ip6tables works on this node
MDB_V6=0                                   # 1: the IPv6 rules are installed
MDB_ACTIVE=0                               # 1: rules of ours may exist; the EXIT trap removes them
MDB_REFUSED_SELFTEST=0
MDB_HAVE_CHAIN=0
MDB_JUMPS=
NODE_LOCK_FD=

# One host pod per node: smoke and service pods share $STATE and the node's iptables. The
# kernel drops the lock when this script ends, even on SIGKILL, so a block found while we hold
# it was left by a pod that is gone. Service-user processes never get the descriptor: every
# command that runs as the service user closes it with {NODE_LOCK_FD}<&-.
node_lock() {
  local lock="$STATE/host-entry.lock"
  install -d -m 0755 "$STATE"
  (umask 077 && : >> "$lock")
  exec {NODE_LOCK_FD}<"$lock"
  flock -n "$NODE_LOCK_FD" \
    || die "another kimi-swe host pod (smoke or service) runs on this node; one at a time"
  log "node lock held ($lock): no other kimi-swe host pod runs on this node"
}

mdb_ipt() {   # mdb_ipt 4|6 ARGS...: iptables or ip6tables of the node's backend (filter table)
  local fam=$1
  shift
  if [[ $fam == 6 ]]; then
    "ip6tables-$MDB_BACKEND" -w 10 "$@"
  else
    "iptables-$MDB_BACKEND" -w 10 "$@"
  fi
}

mdb_count_rules() {   # mdb_count_rules legacy|nft: rules in that backend, IPv4 + IPv6, all tables
  { timeout 30 "iptables-$1-save" || true; timeout 30 "ip6tables-$1-save" || true; } 2>/dev/null \
    | awk '/^-A / { n++ } END { print n + 0 }'
}

mdb_pick_backend() {
  local nft legacy
  nft=$(mdb_count_rules nft)
  legacy=$(mdb_count_rules legacy)
  if (( nft > legacy )); then
    MDB_BACKEND=nft
  elif (( legacy > nft )); then
    MDB_BACKEND=legacy
  else
    die "metadata block: cannot tell which iptables backend the node uses (rules: nft $nft, legacy $legacy)"
  fi
  log "metadata block: iptables backend $MDB_BACKEND (rules on the node: nft $nft, legacy $legacy;" \
      "the larger wins, as in kube-proxy's iptables-wrapper)"
}

mdb_state() {   # mdb_state 4|6: MDB_HAVE_CHAIN=1 if our chain exists; MDB_JUMPS: OUTPUT rules to it
  local rules
  rules=$(mdb_ipt "$1" -S) || return 1
  MDB_HAVE_CHAIN=0
  if grep -qxF -- "-N $MDB_CHAIN" <<<"$rules"; then MDB_HAVE_CHAIN=1; fi
  MDB_JUMPS=$(awk -v c="$MDB_CHAIN" '$1 == "-A" && $2 == "OUTPUT" && $(NF - 1) == "-j" && $NF == c' \
              <<<"$rules")
}

mdb_remove_family() {   # mdb_remove_family 4|6: our OUTPUT jumps, then our chain; 0 if none is left
  local fam=$1 uid
  mdb_state "$fam" || return 1
  # Each jump goes by its exact spec, never by rule number: kube-proxy may insert rules into
  # OUTPUT at any time, and a number could then name its rule instead of ours.
  for uid in $(awk '{ for (i = 1; i < NF; i++) if ($i == "--uid-owner") print $(i + 1) }' <<<"$MDB_JUMPS"); do
    [[ $uid =~ ^[0-9]+$ ]] || return 1
    mdb_ipt "$fam" -D OUTPUT -m owner --uid-owner "$uid" -m comment --comment "$MDB_COMMENT" \
      -j "$MDB_CHAIN" || return 1
  done
  mdb_state "$fam" || return 1
  [[ -z $MDB_JUMPS ]] || return 1
  if [[ $MDB_HAVE_CHAIN == 1 ]]; then
    mdb_ipt "$fam" -F "$MDB_CHAIN" || return 1
    mdb_ipt "$fam" -X "$MDB_CHAIN" || return 1
    mdb_state "$fam" || return 1
  fi
  [[ $MDB_HAVE_CHAIN == 0 && -z $MDB_JUMPS ]]
}

mdb_remove_all() {   # both families; 0 if nothing of ours is left
  local rc=0
  mdb_remove_family 4 || rc=1
  if [[ $MDB_V6_TOOL == 1 ]]; then mdb_remove_family 6 || rc=1; fi
  return "$rc"
}

mdb_apply_family() {   # mdb_apply_family 4|6: our chain first, then the one jump for the service uid
  local fam=$1 ip=$MDB_IP4 reject=icmp-port-unreachable
  if [[ $fam == 6 ]]; then ip=$MDB_IP6 reject=icmp6-port-unreachable; fi
  mdb_ipt "$fam" -N "$MDB_CHAIN" || return 1
  mdb_ipt "$fam" -A "$MDB_CHAIN" -d "$ip" -p udp --dport 53 -j RETURN || return 1
  mdb_ipt "$fam" -A "$MDB_CHAIN" -d "$ip" -p tcp --dport 53 -j RETURN || return 1
  mdb_ipt "$fam" -A "$MDB_CHAIN" -d "$ip" -j REJECT --reject-with "$reject" || return 1
  if [[ $fam == 4 ]]; then
    mdb_ipt 4 -A "$MDB_CHAIN" -d "$MDB_GKE4" -j REJECT --reject-with "$reject" || return 1
  fi
  # The jump comes last, so a half-built chain is never in use; position 1 puts it before
  # any ACCEPT on the node.
  mdb_ipt "$fam" -I OUTPUT 1 -m owner --uid-owner "$RUN_UID" -m comment --comment "$MDB_COMMENT" \
    -j "$MDB_CHAIN"
}

mdb_show() {   # mdb_show 4|6: log our rules as iptables -S prints them
  local fam=$1 rules line
  mdb_state "$fam" || { log "WARNING: metadata block: cannot list the IPv$fam rules"; return 0; }
  rules=$(mdb_ipt "$fam" -S "$MDB_CHAIN" 2>/dev/null) || rules=
  while IFS= read -r line; do
    if [[ $line == -A* ]]; then log "metadata block: IPv$fam rule: $line"; fi
  done <<<"$MDB_JUMPS"$'\n'"$rules"
}

mdb_get() {   # mdb_get root|user URL: prints "HTTP_CODE CURL_EXIT", never the response body
  local url=$2 code rc=0
  local curl=(curl -q -sS -g -o /dev/null --noproxy '*' --connect-timeout 3 -m 10
              -H 'Metadata-Flavor: Google' -w '%{http_code}')
  if [[ $1 == root ]]; then
    code=$("${curl[@]}" --retry 2 "$url" 2>/dev/null) || rc=$?
  else
    code=$("${AS_USER[@]}" "${curl[@]}" "$url" 2>/dev/null {NODE_LOCK_FD}<&-) || rc=$?
  fi
  printf '%s %s\n' "${code:-000}" "$rc"
}

mdb_refused() {   # packets our REJECT rules have refused so far, IPv4 + IPv6
  local n=0 fam out
  for fam in 4 6; do
    if [[ $fam == 6 && $MDB_V6 != 1 ]]; then continue; fi
    out=$(mdb_ipt "$fam" -L "$MDB_CHAIN" -n -v -x 2>/dev/null) || continue
    n=$(( n + $(awk '$3 == "REJECT" { s += $1 } END { print s + 0 }' <<<"$out") ))
  done
  echo "$n"
}

mdb_socket_precheck() {   # fail closed: no socket in the node's network namespace is RUN_UID's yet
  # The kernel's socket tables, as `ss` reads them: column 8 of each row is the owner's uid,
  # column 10 the inode (0: a closing socket that no process holds any more). tcp, udp and raw
  # always exist; tcp6, udp6 and raw6 all exist unless IPv6 is off; icmp, icmp6, udplite and
  # udplite6 (same layout) count where they exist. A live socket of RUN_UID means that something
  # else on this node runs as the service uid, and the block would cut it off too: stop before
  # any rule goes in. Sockets of uid 1000 (GKE's hostNetwork agents) are counted for the log.
  local f files=() missing6=() out mine orphans n1000 bad v6="IPv4 and IPv6"
  for f in tcp udp raw; do
    [[ -r $MDB_PROC_NET/$f ]] || die "metadata block: cannot read $MDB_PROC_NET/$f"
    files+=("$MDB_PROC_NET/$f")
  done
  for f in tcp6 udp6 raw6; do
    if [[ -r $MDB_PROC_NET/$f ]]; then files+=("$MDB_PROC_NET/$f"); else missing6+=("$f"); fi
  done
  if (( ${#missing6[@]} == 3 )); then
    v6="IPv4; no IPv6 socket tables"
  elif (( ${#missing6[@]} > 0 )); then
    die "metadata block: some IPv6 socket tables are missing in $MDB_PROC_NET (${missing6[*]})"
  fi
  for f in icmp icmp6 udplite udplite6; do
    if [[ -r $MDB_PROC_NET/$f ]]; then files+=("$MDB_PROC_NET/$f"); fi
  done
  out=$(awk -v uid="$RUN_UID" '
      FNR == 1 { if ($0 !~ / uid /) bad++; next }
      NF == 0 { next }
      NF < 10 || $8 !~ /^[0-9]+$/ || $10 !~ /^[0-9]+$/ { bad++; next }
      $8 == uid && $10 != 0 { mine++ }
      $8 == uid && $10 == 0 { orphans++ }
      $8 == 1000 && $10 != 0 { n1000++ }
      END { printf "%d %d %d %d\n", mine, orphans, n1000, bad }' "${files[@]}") \
    || die "metadata block: cannot read the socket tables in $MDB_PROC_NET"
  read -r mine orphans n1000 bad <<<"$out"
  (( bad == 0 )) || die "metadata block: $bad line(s) of the socket tables in $MDB_PROC_NET are not" \
    "in the expected format"
  (( mine == 0 )) || die "metadata block: $mine socket(s) in the node's network namespace belong to" \
    "uid $RUN_UID before the service starts: something else on this node runs as the service uid," \
    "and the block would cut it off too. No rule installed"
  log "metadata block: $mine sockets of uid $RUN_UID in the node's network namespace before the" \
      "service starts (${#files[@]} socket tables, $v6): ok"
  if (( orphans > 0 )); then
    log "metadata block: also $orphans closing socket(s) of uid $RUN_UID that no process holds any" \
        "more (left by an ended process, for example an earlier host pod's; not counted)"
  fi
  log "metadata block: $n1000 socket(s) of uid 1000 in the node's network namespace (GKE runs" \
      "hostNetwork agents as uid 1000; informational: the block does not match them)"
}

metadata_block_apply() {   # after prepare, before the service starts
  local fam r url before refused ns fams=IPv4 probes=0
  if [[ $METADATA_BLOCK == 0 ]]; then
    log "WARNING: SWE_METADATA_BLOCK=0: no metadata block. $RUN_AS and its sandboxes can reach the" \
        "GCE metadata server and the node's service account (CS-NET-001). Approved fallback only."
    return 0
  fi
  [[ $RUN_UID != 0 ]] || die "the metadata block needs an unprivileged service user: a rule for" \
    "uid 0 would cut off kubelet and the GKE agents. SWE_RUN_AS=root needs SWE_METADATA_BLOCK=0" \
    "and its own sign-off"
  mdb_socket_precheck
  mdb_pick_backend
  if mdb_ipt 6 -S OUTPUT >/dev/null 2>&1; then MDB_V6_TOOL=1; fi
  MDB_ACTIVE=1                   # from here on, the EXIT trap removes whatever is ours
  for fam in 4 6; do
    if [[ $fam == 6 && $MDB_V6_TOOL != 1 ]]; then continue; fi
    mdb_state "$fam" || die "metadata block: cannot list the IPv$fam filter table (iptables-$MDB_BACKEND)"
    if [[ $MDB_HAVE_CHAIN == 1 || -n $MDB_JUMPS ]]; then
      log "metadata block: removing a stale IPv$fam $MDB_CHAIN left by an earlier host pod on this node"
      mdb_remove_family "$fam" || die "metadata block: cannot remove the stale IPv$fam $MDB_CHAIN or" \
        "its OUTPUT jump; remove them by hand"
    fi
  done
  before=$(mdb_get user "http://$MDB_IP4$MDB_PATH")
  log "metadata block: before the block, $RUN_AS gets HTTP ${before% *} (curl exit ${before#* })" \
      "from $MDB_IP4 (informational)"
  mdb_apply_family 4 || die "metadata block: cannot install the IPv4 rules (iptables-$MDB_BACKEND)"
  if [[ $MDB_V6_TOOL == 1 ]]; then
    if mdb_apply_family 6; then
      MDB_V6=1
      fams="IPv4 and IPv6"
    else
      log "WARNING: metadata block: cannot install the IPv6 rules; removing the partial IPv6 rules"
      mdb_remove_family 6 || die "metadata block: cannot remove the partial IPv6 rules"
    fi
  fi
  if [[ $MDB_V6 == 0 ]]; then
    r=$(mdb_get root "http://[$MDB_IP6]$MDB_PATH")
    [[ ${r% *} == 000 ]] || die "metadata block: no IPv6 block, but root reaches the metadata server" \
      "at [$MDB_IP6] (HTTP ${r% *})"
    # Only "unreachable" (7) or a timeout (28) shows there is no IPv6 path; any other curl error
    # proves nothing about the network, so it fails closed too.
    [[ ${r#* } == 7 || ${r#* } == 28 ]] || die "metadata block: no IPv6 block, and root's request to" \
      "[$MDB_IP6] failed with curl exit ${r#* }, not as unreachable (7) or timed out (28)"
    log "metadata block: no IPv6 block (ip6tables-$MDB_BACKEND unusable here); the metadata server" \
        "is not reachable over IPv6 from this node (curl exit ${r#* }): ok"
  fi
  mdb_show 4
  if [[ $MDB_V6 == 1 ]]; then mdb_show 6; fi
  # Self-test, fail closed. Only HTTP codes and curl exit codes are logged, never a body.
  r=$(mdb_get root "http://$MDB_IP4$MDB_PATH")
  [[ $r == "200 0" ]] || die "metadata block self-test: root must get HTTP 200 from $MDB_IP4," \
    "got HTTP ${r% *} (curl exit ${r#* })"
  log "metadata block self-test: root -> http://$MDB_IP4$MDB_PATH: HTTP 200: ok"
  local urls=("http://$MDB_IP4$MDB_PATH" "http://$MDB_GKE4:988/")
  if [[ $MDB_V6 == 1 ]]; then
    r=$(mdb_get root "http://[$MDB_IP6]$MDB_PATH")
    log "metadata block self-test: root -> http://[$MDB_IP6]$MDB_PATH: HTTP ${r% *}" \
        "(curl exit ${r#* }; informational)"
    urls+=("http://[$MDB_IP6]$MDB_PATH")
  fi
  for url in "${urls[@]}"; do
    # curl exit 7: refused at once, by the REJECT rule. A timeout (28) would mean that
    # something else dropped the request, not our rule.
    r=$(mdb_get user "$url")
    [[ $r == "000 7" ]] || die "metadata block self-test: $RUN_AS -> $url must be refused" \
      "(HTTP 000, curl exit 7), got HTTP ${r% *}, curl exit ${r#* }"
    log "metadata block self-test: $RUN_AS -> $url: refused (HTTP 000, curl exit 7): ok"
    probes=$((probes + 1))
  done
  refused=$(mdb_refused)
  (( refused >= probes )) || die "metadata block self-test: our REJECT rules counted $refused" \
    "refused packet(s) for $probes refused request(s)"
  MDB_REFUSED_SELFTEST=$refused
  log "metadata block self-test: our REJECT rules counted $refused refused packet(s): ok"
  ns=$(awk '$1 == "nameserver" { print $2 }' /etc/resolv.conf 2>/dev/null | paste -sd, -) || ns=
  if timeout 30 "${AS_USER[@]}" getent hosts registry.gitlab.com >/dev/null 2>&1 {NODE_LOCK_FD}<&-; then
    log "metadata block self-test: DNS as $RUN_AS (nameserver ${ns:-unknown}): registry.gitlab.com resolves: ok"
  else
    die "metadata block self-test: DNS as $RUN_AS fails (nameserver ${ns:-unknown}):" \
      "registry.gitlab.com does not resolve"
  fi
  log "metadata block: active for uid $RUN_UID ($fams, iptables-$MDB_BACKEND); self-test passed"
}

metadata_block_remove() {   # teardown, after postcheck: re-check, then remove; 1 if either fails
  local fail=0 r during fams=IPv4
  if [[ $METADATA_BLOCK == 0 ]]; then
    log "WARNING: metadata block: none in this run (SWE_METADATA_BLOCK=0)"
    return 0
  fi
  if [[ $MDB_ACTIVE != 1 ]]; then
    log "WARNING: metadata block: not active at teardown"
    return 1
  fi
  during=$(( $(mdb_refused) - MDB_REFUSED_SELFTEST ))
  if (( during < 0 )); then during="an unknown number of"; fi
  log "metadata block: $during packet(s) from uid $RUN_UID to the metadata servers refused while" \
      "the service ran"
  r=$(mdb_get user "http://$MDB_IP4$MDB_PATH")
  if [[ $r == "000 7" ]]; then
    log "POSTCHECK metadata block still in place at teardown ($RUN_AS -> $MDB_IP4 refused): ok"
  else
    log "POSTCHECK WARNING: metadata block NOT in place at teardown: $RUN_AS -> $MDB_IP4 got" \
        "HTTP ${r% *}, curl exit ${r#* }"
    fail=1
  fi
  if [[ $MDB_V6_TOOL == 1 ]]; then fams="IPv4 and IPv6"; fi
  if mdb_remove_all; then
    MDB_ACTIVE=0
    log "metadata block removed: no $MDB_CHAIN chain and no OUTPUT jump to it ($fams," \
        "iptables-$MDB_BACKEND)"
  else
    log "WARNING: metadata block: cannot remove $MDB_CHAIN or its OUTPUT jump ($fams," \
        "iptables-$MDB_BACKEND); the next host pod on this node removes them"
    fail=1
  fi
  return "$fail"
}

exit_cleanup() {   # EXIT trap: remove the block if the script ends before the teardown did
  set +e
  if [[ $MDB_ACTIVE != 1 ]]; then return 0; fi
  # Stop the service user's processes first (this pod's only), so none outlives the block.
  if [[ -n ${RUN_UID:-} && $RUN_UID != 0 ]]; then pkill -KILL -U "$RUN_UID" 2>/dev/null; fi
  if mdb_remove_all; then
    MDB_ACTIVE=0
    log "metadata block removed on exit: no $MDB_CHAIN chain and no OUTPUT jump to it"
  else
    log "WARNING: metadata block: cannot remove $MDB_CHAIN on exit; the next host pod on this" \
        "node removes it"
  fi
}
# <<< metadata block <<<

# ------------------------------------------------------------------- service lifecycle
SVC_PID=
SVC_RC=
SCRATCH_BEFORE=

start_service() {
  SCRATCH_BEFORE=$(ls -d /tmp/kimi-swe-* 2>/dev/null | sort || true)
  (cd "$RUN" && exec "${AS_USER[@]}" python3 "$WORK/drop/swe-service.py" {NODE_LOCK_FD}<&-) &
  SVC_PID=$!
  log "swe-service.py started (pid $SVC_PID)"
}

wait_ready() {
  local deadline=$((SECONDS + $1))
  while (( SECONDS < deadline )); do
    [[ $TERM_SEEN == 0 ]] || return 1
    kill -0 "$SVC_PID" 2>/dev/null || { log "swe-service.py exited before SWE_READY"; return 1; }
    if probe conn --connection "$CONN" 2>/dev/null; then
      return 0
    fi
    sleep 5
  done
  log "no SWE_READY within $1 s"
  return 1
}

stop_service() {
  [[ -n $SVC_PID ]] || return 0
  # Keep the token for the leak scan; the service resets the connection file on exit.
  probe snapshot --connection "$CONN" --out "$PRIV/root/service-token" \
    || log "WARNING: no service token snapshot"
  log "stopping swe-service.py with SIGTERM, as Slurm does at teardown"
  kill -TERM "$SVC_PID" 2>/dev/null || true
  local rc
  while :; do
    wait "$SVC_PID" && rc=0 || rc=$?
    kill -0 "$SVC_PID" 2>/dev/null || break
  done
  SVC_RC=$rc
  SVC_PID=
  log "swe-service.py exited with $SVC_RC (143 = stopped by SIGTERM)"
}

postcheck() {
  local fail=0 after retained left=none i netrc=() roots=() owners
  # 1. Leftover processes of the service user. Slurm kills every process left in a step when
  # it ends; do the same first, so nothing still running can race the root checks below.
  # Warning only. The pod has no hostPID, so this sees only this pod's processes.
  if [[ $RUN_UID == 0 ]]; then
    log "POSTCHECK leftover processes: not checked (SWE_RUN_AS=root)"
  else
    left=$(pgrep -U "$RUN_UID" -c || true)
    left=${left:-0}
    if [[ $left == 0 ]]; then
      log "POSTCHECK no $RUN_AS processes left after the drop's teardown: ok"
    else
      log "POSTCHECK WARNING: $left $RUN_AS processes left after the drop's teardown; killing them"
      pkill -KILL -U "$RUN_UID" || true
      for i in {1..10}; do
        pgrep -U "$RUN_UID" > /dev/null || break
        sleep 1
      done
      if pgrep -U "$RUN_UID" > /dev/null; then
        log "POSTCHECK WARNING: $RUN_AS processes still present 10 s after SIGKILL"
      fi
    fi
    # Every entry in the run directory belongs to the service user (the pilot client runs as it
    # too); root writes its reports below, after this count. Warning only. Lists owners only.
    owners=$(find "$RUN" -xdev -printf '%U:%G\n' | awk -v want="$RUN_UID:$RUN_GID" \
               '{ n++ } $0 != want { bad++ } END { printf "%d %d\n", n, bad }') || owners=
    if [[ $owners =~ ^([0-9]+)\ 0$ ]]; then
      log "POSTCHECK all ${BASH_REMATCH[1]} entries in the run directory belong to $RUN_AS ($RUN_UID:$RUN_GID): ok"
    elif [[ $owners =~ ^([0-9]+)\ ([0-9]+)$ ]]; then
      log "POSTCHECK WARNING: ${BASH_REMATCH[2]} of ${BASH_REMATCH[1]} entries in the run directory do" \
          "not belong to $RUN_AS ($RUN_UID:$RUN_GID)"
    else
      log "POSTCHECK WARNING: cannot list who owns the entries of the run directory"
    fi
  fi
  # 2. The drop resets the connection file to {} on exit.
  if probe conn --connection "$CONN" --expect-reset; then
    log "POSTCHECK connection file reset to {}: ok"
  else
    log "POSTCHECK connection file was NOT reset"; fail=1
  fi
  # 3. The drop's teardown removes its scratch (and the pyxis_* containers in it).
  after=$(ls -d /tmp/kimi-swe-* 2>/dev/null | sort || true)
  retained=$(comm -13 <(printf '%s\n' "$SCRATCH_BEFORE") <(printf '%s\n' "$after") | sed '/^$/d' || true)
  if [[ -z $retained ]]; then
    log "POSTCHECK service scratch and its pyxis_* containers removed: ok"
  else
    log "POSTCHECK scratch retained: $(tr '\n' ' ' <<<"$retained")"; fail=1
  fi
  # 4. No secret in the run directory or /tmp. The registry token comes from the root-owned
  # Secret mount, not from the service user's copy.
  if [[ $MODE == service ]]; then netrc=(--netrc "$SECRET_DIR/.credentials"); fi
  roots=(--root "$RUN" --root /tmp)
  if [[ -n $SCAN_REPORTS ]]; then
    # Production: the benchmark's report for this run (read-only here). The recipe replaces
    # the service token in its *.yaml before sflow collects it, as the drop's teardown does.
    if [[ -d $SCAN_REPORTS/$RUN_ID ]]; then
      roots+=(--root "$SCAN_REPORTS/$RUN_ID")
      log "POSTCHECK the leak scan includes this run's report directory"
    else
      log "POSTCHECK no report directory for $RUN_ID on this node (the benchmark did not start)"
    fi
  fi
  if probe leakscan "${roots[@]}" \
      --secret-file "$PRIV/root/service-token" "${netrc[@]}" \
      --report "$RUN/postcheck-leaks.json"; then
    log "POSTCHECK leak scan: no secret found"
  else
    log "POSTCHECK leak scan: secret found or scan failed (see above)"; fail=1
  fi
  rm -f "$PRIV/root/service-token"
  if ! probe postcheck-report --out "$RUN/postcheck.json" --service-rc "${SVC_RC:-none}" \
      --fail "$fail" --retained "$retained" --leftover "$left"; then
    log "POSTCHECK could not write postcheck.json"; fail=1
  fi
  return "$fail"
}

# ------------------------------------------------------------------- modes
smoke_mode() {
  local fail=0 url code t0 mdb=()
  for url in https://pypi.org/simple/pip/ https://files.pythonhosted.org/ \
             https://huggingface.co/api/datasets/princeton-nlp/SWE-bench_Verified \
             https://registry.gitlab.com/v2/ https://gitlab.com/ https://<registry>/; do
    code=$(curl -q -sS -o /dev/null -m 20 -w '%{http_code}' "$url" 2>/dev/null) || code=000
    log "SMOKE egress $url -> HTTP $code"
    [[ $code != 000 ]] || fail=1
  done
  log "SMOKE disk free on $STATE: $(df -h --output=avail "$STATE" | tail -1 | tr -d ' ')"
  log "SMOKE service user: $(id "$RUN_AS")"
  # With the block on, the probe also checks it from inside a sandbox (TCP connects only).
  if [[ $METADATA_BLOCK == 1 ]]; then mdb=(--metadata-block); fi

  start_service
  t0=$SECONDS
  if wait_ready 1800; then
    log "SMOKE SWE_READY after $((SECONDS - t0)) s (uv sync + service start)"
    url=$(probe conn --connection "$CONN" --print-url) || url=
    if [[ $url =~ ^http://127\.0\.0\.1:[0-9]{1,5}$ ]]; then
      code=$(curl -q -sS -o /dev/null -m 5 -w '%{http_code}' "$url/health" 2>/dev/null) || code=000
    else
      log "SMOKE connection URL is not http://127.0.0.1:<port>"; code=000
    fi
    log "SMOKE service /health -> HTTP $code"
    [[ $code == 200 ]] || fail=1
    if run_interruptible "${AS_USER[@]}" "$SVC_OUT/service-venv/bin/python" "$WORK/lib/swe_probe.py" \
        smoke --image "$SMOKE_IMAGE" --drop "$WORK/drop" --out "$RUN/probe" "${mdb[@]}" {NODE_LOCK_FD}<&-; then
      log "SMOKE sandbox probe: ok"
    else
      log "SMOKE sandbox probe: FAILED"; fail=1
    fi
  else
    fail=1
  fi
  stop_service
  [[ $SVC_RC == 143 ]] || { log "SMOKE expected exit 143 after SIGTERM, got $SVC_RC"; fail=1; }
  postcheck || fail=1
  metadata_block_remove || fail=1
  if [[ $fail == 0 ]]; then log "SMOKE_OK"; else log "SMOKE_FAIL"; fi
  return "$fail"
}

service_mode() {
  local rc fail=0
  start_service
  if wait_ready 1800; then
    log "SWE_READY seen; the client may start"
    if [[ -n $PREWARM ]]; then
      log "prewarming SWE images ($PREWARM) into $STATE/enroot-cache through the drop transport"
      if run_interruptible "${AS_USER[@]}" "$SVC_OUT/service-venv/bin/python" "$WORK/lib/swe_probe.py" \
          prewarm --ids "$PREWARM" --registry "$SWE_IMAGE_REGISTRY" --cache "$SVC_OUT/enroot-cache" \
          --drop "$WORK/drop" --out "$RUN/prewarm" {NODE_LOCK_FD}<&-; then
        log "PREWARM_DONE"
      else
        log "PREWARM_FAILED"
      fi
    fi
  fi
  while [[ -n $SVC_PID ]]; do
    if [[ $TERM_SEEN == 1 ]]; then stop_service; break; fi
    wait "$SVC_PID" && rc=0 || rc=$?
    if ! kill -0 "$SVC_PID" 2>/dev/null; then
      SVC_RC=$rc; SVC_PID=
      log "swe-service.py exited on its own with $SVC_RC"
    fi
  done
  postcheck || fail=1
  metadata_block_remove || fail=1
  [[ $TERM_SEEN == 1 && $SVC_RC == 143 ]] || fail=1
  if [[ $fail == 0 ]]; then log "service run finished: ok"; else log "service run finished: FAILED"; fi
  return "$fail"
}

# The metadata block must not outlive this script, however it ends (die, set -e). The TERM/INT
# trap above stays as it is: SIGTERM still only sets TERM_SEEN.
trap exit_cleanup EXIT
stage
install_enroot
node_lock
prepare
metadata_block_apply
if [[ $MODE == service ]]; then
  load_credentials
else
  REGISTRY="<email>/smoke/none"   # never pulled in smoke mode
fi
service_env
[[ $TERM_SEEN == 0 ]] || die "SIGTERM during setup; the service was not started"
if [[ $MODE == smoke ]]; then smoke_mode; else service_mode; fi
