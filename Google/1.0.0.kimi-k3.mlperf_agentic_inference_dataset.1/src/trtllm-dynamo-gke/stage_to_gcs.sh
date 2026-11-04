#!/usr/bin/env bash
# v1.0.3: mirror the Kimi K3 NVFP4 checkpoint HuggingFace -> GCS, as a GCE
# STARTUP SCRIPT on a throwaway staging VM. stage-kimi-k3-assets.yaml (v1.0.3)
# then fans it out from GCS to the GKE nodes.
#
# Why a VM and not a Job in the cluster: the GKE node pools are scoped
# devstorage.read_only and Workload Identity is off, so no pod on that cluster
# can write to GCS. The nodes can read, which is all the fan-out needs.
#
# Why a startup script and not ssh: nothing to log in to. Progress goes to the
# serial console (gcloud compute instances get-serial-port-output) and the
# final log + verification record go to ${BUCKET}/staging-logs/.
#
# What v1.0.3 changes vs v1.0.1's stage_to_gcs.sh: the server checkpoint is
# nvidia/Kimi-K3-NVFP4@b2428a0b (v1.0.3 NVIDIA/src/kimi-k3/README.md "Model and
# data"), not moonshotai/Kimi-K3@f831ab66. The DSpark draft
# (RadixArk/Kimi-K3-DSpark@3c5bac30) and the dataset are unchanged and already
# in the bucket, so this script stages only the new checkpoint.
#
# Every file is verified against the pinned revision before upload: sha256 for
# the LFS files, git blob sha1 for the small files (manifest from the HF tree
# API, passed in as instance metadata `manifest`: path<TAB>size<TAB>kind<TAB>hash).
# A mismatch stops the script before anything is uploaded.
#
# Usage (from a workstation; see v1.0.3/kimi-k3/README.md). The bucket lives in
# the cluster's own project, so the VM's default service account can write it
# and the node pools (devstorage.read_only) can read it. The VM has no external
# IP: it sits on the cluster's VPC and reaches HuggingFace through its Cloud NAT.
#   gcloud compute instances create kimi-k3-stager --project=<cluster-project> \
#     --zone=us-central1-a --machine-type=n2-standard-32 \
#     --network=<cluster-net> --subnet=<cluster-subnet> --no-address --scopes=cloud-platform \
#     --image-family=ubuntu-2404-lts-amd64 --image-project=ubuntu-os-cloud --boot-disk-size=50GB \
#     --create-disk=name=kimi-k3-stage-data,device-name=stage,size=2500GB,type=pd-ssd,auto-delete=yes \
#     --metadata=bucket=<bucket><bucket> \
#     --metadata-from-file=startup-script=stage_to_gcs.sh,manifest=<manifest.tsv>
#   ... wait for "STAGE_DONE" or "STAGE_FAILED" on the serial console, then
#   gcloud compute instances delete kimi-k3-stager --zone=us-central1-a
# Image: Ubuntu, because the cluster project's org policy
# (compute.trustedImageProjects) allows ubuntu-os-cloud but not debian-cloud.
# Ubuntu's GCE image ships gcloud/gsutil as a snap under /snap/bin.
set -uo pipefail
export HOME="${HOME:-/root}"
export PATH="${PATH:-/usr/sbin:/usr/bin:/sbin:/bin}:/snap/bin"

MODEL_REPO=nvidia/Kimi-K3-NVFP4
MODEL_REV=b2428a0b83a8b712ff2e1a8448a103d4175341f1
MODEL_DIR_NAME=Kimi-K3-NVFP4
EXPECTED_FILES=116
EXPECTED_BYTES=1610039528856
WORK=<node-mount>/stage
DEV=/dev/disk/by-id/google-stage
LOG=/var/log/kimi-k3-stage.log
DONE=/var/lib/kimi-k3-stage.done

exec > >(tee -a "${LOG}" > /dev/ttyS0) 2>&1
ts() { date -u +%FT%TZ; }
md() { curl -sf -H 'Metadata-Flavor: Google' "http://<internal-host>/computeMetadata/v1/instance/attributes/$1"; }
BUCKET="$(md bucket || true)"
upload_log() { [ -n "${BUCKET}" ] && gsutil -q cp "${LOG}" "${BUCKET}/staging-logs/kimi-k3-nvfp4-${MODEL_REV:0:8}.log" || true; }
fail() { echo "$(ts) STAGE_FAILED: $*"; upload_log; exit 1; }

[ -f "${DONE}" ] && { echo "$(ts) STAGE_DONE (already done on an earlier boot)"; exit 0; }
[[ "${BUCKET}" == <bucket> ]] || fail "instance metadata 'bucket' must be <bucket>"
md manifest > /root/manifest.tsv || fail "instance metadata 'manifest' missing"
n=$(wc -l < /root/manifest.tsv)
sum=$(awk -F'\t' '{s += $2} END {printf "%d", s}' /root/manifest.tsv)
[ "${n}" = "${EXPECTED_FILES}" ] && [ "${sum}" = "${EXPECTED_BYTES}" ] \
  || fail "manifest has ${n} files / ${sum} B, expected ${EXPECTED_FILES} / ${EXPECTED_BYTES}"
echo "$(ts) STAGE_START ${MODEL_REPO}@${MODEL_REV:0:8} -> ${BUCKET}/models/${MODEL_DIR_NAME}"

# --- data disk ---------------------------------------------------------------
[ -b "${DEV}" ] || fail "data disk ${DEV} not attached"
mkdir -p "${WORK}"
if ! blkid "${DEV}" > /dev/null 2>&1; then
  mkfs.ext4 -q -F -m 0 -E lazy_itable_init=0,lazy_journal_init=0,discard "${DEV}" || fail "mkfs"
fi
mountpoint -q "${WORK}" || mount -o discard,defaults "${DEV}" "${WORK}" || fail "mount"
df -h "${WORK}"

# --- tools -------------------------------------------------------------------
export DEBIAN_FRONTEND=noninteractive
# Ubuntu's unattended-upgrades can hold the dpkg lock right after first boot.
apt_ok=0
for i in 1 2 3 4 5; do
  if apt-get -qq -o DPkg::Lock::Timeout=300 update \
     && apt-get -qq -o DPkg::Lock::Timeout=300 install -y python3-venv > /dev/null; then
    apt_ok=1; break
  fi
  sleep 30
done
[ "${apt_ok}" = 1 ] || fail "apt"
python3 -m venv /opt/hfenv && /opt/hfenv/bin/pip install -q --upgrade pip \
  && /opt/hfenv/bin/pip install -q "huggingface_hub[cli]" || fail "pip install huggingface_hub"
export PATH="/opt/hfenv/bin:${PATH}"
# huggingface_hub >= 1.x downloads via Xet; this replaces HF_HUB_ENABLE_HF_TRANSFER.
export HF_XET_HIGH_PERFORMANCE=1
hf version 2>/dev/null || true
# gcloud/gsutil come from the google-cloud-cli snap; wait for snap seeding and
# fail now rather than after the 1.6 TB download.
snap wait system seed.loaded 2>/dev/null || true
command -v gcloud > /dev/null && command -v gsutil > /dev/null || fail "gcloud/gsutil not on PATH"
gcloud storage ls "${BUCKET}" > /dev/null || fail "cannot list ${BUCKET}"

# --- download (resumable) ----------------------------------------------------
# HF's Xet CDN answers HTTP 429 when many clients share one egress IP (the
# cluster's Cloud NAT). hf download keeps completed files, so each attempt
# resumes; 8 attempts with a capped backoff ride out a busy window.
DST="${WORK}/${MODEL_DIR_NAME}"
ATTEMPTS=8
ok=0
for attempt in $(seq 1 "${ATTEMPTS}"); do
  echo "$(ts) download attempt ${attempt}/${ATTEMPTS}"
  if hf download "${MODEL_REPO}" --revision "${MODEL_REV}" --local-dir "${DST}" --quiet; then ok=1; break; fi
  [ "${attempt}" -lt "${ATTEMPTS}" ] && sleep $(( attempt * 60 < 300 ? attempt * 60 : 300 ))
done
[ "${ok}" = 1 ] || fail "hf download failed after ${ATTEMPTS} attempts"
echo "$(ts) DOWNLOADED $(du -sh --exclude=.cache "${DST}" | cut -f1)"

# --- verify every file against the pinned revision -----------------------------
cat > /root/verify.py <<'PY'
import hashlib, os, sys
from concurrent.futures import ThreadPoolExecutor
root, manifest = sys.argv[1], sys.argv[2]
rows = [l.rstrip("\n").split("\t") for l in open(manifest) if l.strip()]
def check(row):
    path, size, kind, want = row[0], int(row[1]), row[2], row[3]
    p = os.path.join(root, path)
    if not os.path.isfile(p):
        return f"MISSING {path}"
    if os.path.getsize(p) != size:
        return f"SIZE {path} {os.path.getsize(p)} != {size}"
    h = hashlib.sha256() if kind == "sha256" else hashlib.sha1(b"blob %d\0" % size)
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(16 << 20), b""):
            h.update(chunk)
    return None if h.hexdigest() == want else f"HASH {path}"
with ThreadPoolExecutor(16) as ex:
    bad = [r for r in ex.map(check, rows) if r]
extra = []
for d, dirs, files in os.walk(root):
    dirs[:] = [x for x in dirs if not (d == root and x == ".cache")]
    for f in files:
        rel = os.path.relpath(os.path.join(d, f), root)
        extra.append(rel)
extra = sorted(set(extra) - {r[0] for r in rows})
for b in bad:
    print("VERIFY_FAIL", b)
for e in extra:
    print("VERIFY_EXTRA", e)
print(f"VERIFIED {len(rows) - len(bad)}/{len(rows)} files, extra={len(extra)}")
sys.exit(1 if bad or extra else 0)
PY
echo "$(ts) verifying sha256/sha1 of ${EXPECTED_FILES} files"
python3 /root/verify.py "${DST}" /root/manifest.tsv > /root/verify.txt; vrc=$?
cat /root/verify.txt
[ "${vrc}" = 0 ] || fail "verification failed (nothing uploaded)"

# --- upload --------------------------------------------------------------------
echo "$(ts) uploading to ${BUCKET}/models/${MODEL_DIR_NAME}"
gcloud storage rsync --recursive --exclude='^\.cache/' "${DST}" "${BUCKET}/models/${MODEL_DIR_NAME}" \
  || fail "gcloud storage rsync failed (re-run to resume)"
# `gcloud storage du -s` right-aligns the size; a 13-digit size leaves no space
# before the URL ("1610039528856<bucket>"), so take the leading integer.
got=$(gcloud storage du -s "${BUCKET}/models/${MODEL_DIR_NAME}" | grep -oE '^[[:space:]]*[0-9]+' | head -1 | tr -d '[:space:]')
[[ "${got}" =~ ^[0-9]+$ ]] || fail "could not parse the byte count from gcloud storage du"
cnt=$(gcloud storage ls -r "${BUCKET}/models/${MODEL_DIR_NAME}/**" | grep -vc '/$')
[ "${got}" = "${EXPECTED_BYTES}" ] && [ "${cnt}" = "${EXPECTED_FILES}" ] \
  || fail "bucket has ${cnt} objects / ${got} B, expected ${EXPECTED_FILES} / ${EXPECTED_BYTES}"
{
  echo "repo=${MODEL_REPO}"; echo "revision=${MODEL_REV}"; echo "files=${cnt}"; echo "bytes=${got}"
  echo "verified_utc=$(ts)"; cat /root/verify.txt
} > /root/verified.txt
gsutil -q cp /root/verified.txt "${BUCKET}/staging-logs/kimi-k3-nvfp4-${MODEL_REV:0:8}.verified.txt" || fail "record upload"
touch "${DONE}"
echo "$(ts) STAGE_DONE ${cnt} objects, ${got} B in ${BUCKET}/models/${MODEL_DIR_NAME}"
upload_log
# Stop billing for the VM; the operator deletes it (and its auto-delete disk).
shutdown -h +1 "kimi-k3 staging done" || true
