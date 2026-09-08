#!/usr/bin/env python3
"""gpu.py - a GPU VM for the depth-estimation thesis that follows the GPUs.

There is always at most one VM, called depth-l4. Stopping it keeps its disk, so
the next "up" restarts it in place in about a minute. Only when its zone has no
GPU capacity is it moved: the disk is captured as an image and the VM is
re-created from that image in another zone, so the new VM starts from exactly
where you left off (packages, caches, symlinks and files included). Your home
directory (/home/HP) is additionally mirrored in the bucket as a backup.

    python gpu.py up             restart the VM, or move it if its zone has no GPU
    python gpu.py up --fresh     always move (e.g. to hunt for an L4 when you got a T4)
    python gpu.py up --l4-only   do not fall back to a T4
    python gpu.py up --spot      try cheap Spot L4s first (may be interrupted)
    python gpu.py up --us        also try US zones (your data leaves the EU)
    python gpu.py down           save the home dir to the bucket and stop the VM
                                 (disk kept, about 7 kr/day; auto-deleted after 7 idle days)
    python gpu.py down --delete  capture the disk as an image, then delete the VM
    python gpu.py status         show the VM, its zone, images and the last sync
    python gpu.py ssh            open a shell on the VM
    python gpu.py save-env       capture the disk as an image now (optional; moves
                                 and --delete do this automatically)

Requires the gcloud CLI, logged in (gcloud auth login).
"""
import argparse
import datetime as dt
import os
import shutil
import subprocess
import sys
import tempfile
import time

PROJECT = "locallife-thesis-depth"
NAME = "depth-l4"
BUCKET = "gs://locallife-thesis-depth-data"
IMAGE_FAMILY = "depth-env"
IMAGES_TO_KEEP = 2          # newest N images of the family are kept, older ones pruned
DISK_GB = "200"             # cannot be smaller than the image (200 GB)

# Candidate (zone, machine type, accelerator) tiers, tried in order.
L4_SHAPES = ["g2-standard-4", "g2-standard-8"]
L4_ZONES_EU = [
    "europe-west1-b", "europe-west1-c",                   # Belgium
    "europe-west4-a", "europe-west4-b", "europe-west4-c", # Netherlands
    "europe-west2-a", "europe-west2-b",                   # London
    "europe-west3-a", "europe-west3-b",                   # Frankfurt
    "europe-west6-b", "europe-west6-c",                   # Zurich
]
T4_ZONES_EU = [
    "europe-west1-b", "europe-west1-c", "europe-west1-d",
    "europe-west4-a", "europe-west4-b", "europe-west4-c",
    "europe-west2-a", "europe-west2-b", "europe-west3-b",
    "europe-central2-b", "europe-central2-c",             # Warsaw
]
L4_ZONES_US = ["us-central1-a", "us-central1-b", "us-central1-c",
               "us-east4-a", "us-east4-b", "us-east4-c",
               "us-east1-b", "us-east1-c", "us-east1-d",
               "us-west1-a", "us-west1-b"]
T4_ZONES_US = ["us-central1-a", "us-central1-b", "us-central1-c", "us-central1-f",
               "us-east1-c", "us-east1-d", "us-west1-a", "us-west1-b"]
T4_SHAPE = "n1-standard-8"

GCLOUD = shutil.which("gcloud")
if not GCLOUD:
    sys.exit("gcloud not found on PATH - install the Google Cloud CLI first")
SSH_KEY_FILE = os.environ.get("GPU_SSH_KEY_FILE")   # optional: use a specific SSH key

# --------------------------------------------------------------------------- #
# Scripts that run on the VM (passed as metadata; run as root on every boot).
# --------------------------------------------------------------------------- #
STARTUP_SCRIPT = r'''#!/bin/bash
# depth-l4 startup: make sure /home/HP is current, install sync + idle-shutdown, mount bucket.
BUCKET=gs://locallife-thesis-depth-data
U=HP; H=/home/$U
LOG=/var/log/depth-startup.log
exec > >(tee -a $LOG /dev/ttyS0) 2>&1     # log file + serial console (visible via get-serial-port-output)
echo "=== startup $(date -Is) on $(hostname)"
# NB: gcloud only honours a top-level alternation reliably when the whole thing is one group
# (verified empirically with --dry-run; bare 'a|b' only applied the last alternative).
EXCL='(^\.cache/.*|^\.nv/.*|^snap/.*|^__pycache__/.*|.*/__pycache__/.*|^\.ipynb_checkpoints/.*|.*/\.ipynb_checkpoints/.*|.*\.pyc$)'
MD=http://metadata.google.internal/computeMetadata/v1
ID=$(curl -s -H 'Metadata-Flavor: Google' $MD/instance/id)
HOME_SOURCE=$(curl -sf -H 'Metadata-Flavor: Google' $MD/instance/attributes/home-source || echo bucket)
MARK=/var/lib/depth/instance-id     # id of the instance whose home is current on this disk
mkdir -p /var/lib/depth

# 0. tools. The image's gcloud is a snap, and snaps cannot be launched during shutdown
#    ("snap run" fails), so the sync helper calls the snap's raw binary directly - that
#    works at shutdown (verified). gcsfuse mounts the bucket; installed once if missing.
GC=$(command -v /usr/bin/gcloud || command -v /snap/google-cloud-cli/current/bin/gcloud || command -v gcloud)
echo "using $GC"
if ! command -v gcsfuse >/dev/null 2>&1; then
  echo "installing gcsfuse"
  [ -f /usr/share/keyrings/cloud.google.gpg ] || curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
  echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt gcsfuse-$(lsb_release -cs) main" > /etc/apt/sources.list.d/gcsfuse.list
  apt-get update -qq >/dev/null 2>&1 && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq fuse gcsfuse >/dev/null 2>&1 || echo "WARNING: gcsfuse install failed (use gcloud storage cp instead of /mnt/bucket)"
fi

# 1. wait for the user account (the guest agent creates it from the ssh-keys metadata)
for i in $(seq 1 30); do id $U >/dev/null 2>&1 && break; sleep 2; done
id $U >/dev/null 2>&1 || echo "WARNING: user $U not present yet"

# 2. push helper: bucket mirrors the VM. Guarded so a broken boot can never wipe the bucket.
cat > /opt/sync-home.sh <<EOF
#!/bin/bash
[ -f $MARK ] || { echo "sync-home: home not initialised on this disk, refusing to push"; exit 1; }
[ "\$(ls -A $H 2>/dev/null | wc -l)" -gt 2 ] || { echo "sync-home: home looks empty, refusing"; exit 1; }
$GC storage rsync --recursive --preserve-posix --delete-unmatched-destination-objects --exclude '$EXCL' $H $BUCKET/home/$U \
  && date -Is | $GC storage cp - $BUCKET/home/.last-sync && echo "sync-home: ok \$(date -Is)"
EOF
chmod +x /opt/sync-home.sh

# 3. home dir.
#    same instance restarted           -> local disk is current, nothing to do
#    image captured from the last disk -> local disk is current; push it as the new backup
#    plain image                       -> pull from the bucket (seed the bucket if empty)
if [ "$(cat $MARK 2>/dev/null)" = "$ID" ]; then
  echo "same instance restarted: local home is current"; HOME_STATE=local
elif [ "$HOME_SOURCE" = "local" ]; then
  echo "new VM from the previous VM's disk image: local home is current, refreshing the backup"
  echo $ID > $MARK; /opt/sync-home.sh && HOME_STATE=carried || HOME_STATE=carried-unsynced
elif $GC storage ls $BUCKET/home/$U/ >/dev/null 2>&1; then
  echo "fresh disk: pulling home from bucket"
  $GC storage rsync --recursive --preserve-posix --delete-unmatched-destination-objects --exclude "$EXCL" $BUCKET/home/$U $H \
    && echo $ID > $MARK && echo "pull ok" && HOME_STATE=pulled
else
  echo "bucket has no home yet: seeding it from this image"
  echo $ID > $MARK; /opt/sync-home.sh && HOME_STATE=seeded
fi
chown -R $U:$U $H

# 4. idle shutdown: after 30 min with no GPU load, no SSH connection and low CPU load,
#    sync the home dir and stop this VM (disk kept, so "up" restarts it in place).
#    touch /tmp/keep-running to disable until reboot.
cat > /opt/idle-shutdown.sh <<'EOF'
#!/bin/bash
STATE=/var/run/idle-count; THRESH=6   # 6 checks x 5 min
[ -f /tmp/keep-running ] && { echo 0 > $STATE; exit 0; }
UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | sort -rn | head -1)
[ -z "$UTIL" ] && exit 0
SSH=$(ss -Htn state established '( sport = :22 )' 2>/dev/null | wc -l)
LOAD=$(awk '{print int($1*100)}' /proc/loadavg)
if [ "$UTIL" -lt 5 ] && [ "$SSH" -eq 0 ] && [ "$LOAD" -lt 50 ]; then
  C=$(( $(cat $STATE 2>/dev/null || echo 0) + 1 )); echo $C > $STATE
  if [ "$C" -ge "$THRESH" ]; then
    echo "idle for 30 min: syncing and stopping $(date -Is)" >> /var/log/depth-startup.log
    /opt/sync-home.sh >> /var/log/depth-startup.log 2>&1
    /sbin/poweroff
  fi
else
  echo 0 > $STATE
fi
exit 0
EOF
chmod +x /opt/idle-shutdown.sh
echo '*/5 * * * * root /opt/idle-shutdown.sh' > /etc/cron.d/idle-shutdown

# 5. the bucket as a folder, for datasets and results (best effort)
if command -v gcsfuse >/dev/null 2>&1; then
  mkdir -p /mnt/bucket
  grep -q ' /mnt/bucket ' /proc/mounts || gcsfuse --implicit-dirs -o allow_other --uid "$(id -u $U)" --gid "$(id -g $U)" \
    locallife-thesis-depth-data /mnt/bucket && echo "bucket mounted at /mnt/bucket"
fi

# 6. report readiness on the serial console (the launcher waits for this line)
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
PY=/opt/conda/bin/python; [ -x $PY ] || PY=python3
TORCH=$($PY -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available())' 2>&1 | tail -1)
echo "DEPTH_READY ts=$(date -u +%s) gpu=[$GPU] $TORCH home=${HOME_STATE:-FAILED}" | tee /dev/ttyS0
'''

SHUTDOWN_SCRIPT = r'''#!/bin/bash
# Runs on every stop/delete/preemption: last-chance sync of the home dir.
exec >> /var/log/depth-startup.log 2>&1
echo "=== shutdown-script $(date -Is)"
[ -x /opt/sync-home.sh ] && /opt/sync-home.sh
echo "=== shutdown-script done $(date -Is)"
'''


# --------------------------------------------------------------------------- #
def gcloud(*args, check=True, capture=True, timeout=None):
    cmd = [GCLOUD, *args]
    r = subprocess.run(cmd, text=True, timeout=timeout,
                       stdout=subprocess.PIPE if capture else None,
                       stderr=subprocess.PIPE if capture else None)
    if check and r.returncode != 0:
        raise RuntimeError((r.stderr or r.stdout or "").strip())
    return r


def ssh_args():
    return ["--ssh-key-file", SSH_KEY_FILE] if SSH_KEY_FILE else []


def find_instance():
    """Return (zone, status, last_stop) for the VM, or None."""
    r = gcloud("compute", "instances", "list", "--project", PROJECT,
               "--filter", f"name={NAME}",
               "--format", "value(zone.basename(),status,lastStopTimestamp)")
    line = r.stdout.strip()
    if not line:
        return None
    parts = line.split("\t")
    return parts[0], parts[1], (parts[2] if len(parts) > 2 else "")


def last_sync():
    r = gcloud("storage", "cat", f"{BUCKET}/home/.last-sync", check=False)
    return r.stdout.strip() if r.returncode == 0 else ""


def parse_ts(s):
    """RFC3339 -> aware datetime, or None."""
    try:
        return dt.datetime.fromisoformat(s.strip().replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def list_images():
    """Family images, newest first: [(name, creationTimestamp)]."""
    r = gcloud("compute", "images", "list", "--project", PROJECT, "--filter", f"family={IMAGE_FAMILY}",
               "--sort-by", "~creationTimestamp", "--format", "value(name,creationTimestamp)")
    return [tuple(l.split("\t")) for l in r.stdout.strip().splitlines() if l.strip()]


def make_image(zone, why):
    """Capture the VM's disk as a new image in the family; prune old ones. Returns the name."""
    name = f"{IMAGE_FAMILY}-{dt.datetime.now().strftime('%Y%m%d-%H%M')}"
    if any(n == name for n, _ in list_images()):
        name += "b"
    print(f"capturing the VM's disk as image {name} ({why}; 3-8 min)...", flush=True)
    gcloud("compute", "images", "create", name, "--project", PROJECT,
           "--source-disk", NAME, "--source-disk-zone", zone, "--family", IMAGE_FAMILY,
           "--storage-location", "eu", "--force", timeout=1800)
    for old, _ in list_images()[IMAGES_TO_KEEP:]:
        gcloud("compute", "images", "delete", old, "--project", PROJECT, "--quiet", check=False)
        print(f"  pruned old image {old}")
    return name


def candidates(args):
    tiers = []
    if args.spot:
        tiers += [(z, s, "nvidia-l4", "SPOT") for z in L4_ZONES_EU for s in L4_SHAPES]
    tiers += [(z, s, "nvidia-l4", "STANDARD") for z in L4_ZONES_EU for s in L4_SHAPES]
    if not args.l4_only:
        tiers += [(z, T4_SHAPE, "nvidia-tesla-t4", "STANDARD") for z in T4_ZONES_EU]
    if args.us:
        tiers += [(z, s, "nvidia-l4", "STANDARD") for z in L4_ZONES_US for s in L4_SHAPES]
        if not args.l4_only:
            tiers += [(z, T4_SHAPE, "nvidia-tesla-t4", "STANDARD") for z in T4_ZONES_US]
    return tiers


def write_temp(content):
    f = tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False, newline="\n")
    f.write(content)
    f.close()
    return f.name


def try_create(zone, machine, accel, model, startup, shutdown, image, home_source):
    cmd = ["compute", "instances", "create", NAME, "--project", PROJECT, "--zone", zone,
           "--machine-type", machine, "--accelerator", f"type={accel},count=1",
           "--image-project", PROJECT,
           "--boot-disk-size", f"{DISK_GB}GB", "--boot-disk-type", "pd-balanced",
           "--maintenance-policy", "TERMINATE", "--scopes", "cloud-platform",
           "--metadata", f"install-nvidia-driver=True,home-source={home_source}",
           "--metadata-from-file", f"startup-script={startup},shutdown-script={shutdown}",
           "--provisioning-model", model, "--quiet"]
    cmd += ["--image", image] if image else ["--image-family", IMAGE_FAMILY]
    if model == "SPOT":
        cmd += ["--instance-termination-action", "STOP", "--no-restart-on-failure"]
    else:
        cmd += ["--restart-on-failure"]
    r = gcloud(*cmd, check=False, timeout=300)
    if r.returncode == 0:
        return True, ""
    err = r.stderr or r.stdout
    if "ZONE_RESOURCE_POOL_EXHAUSTED" in err or "STOCKOUT" in err or "resource_availability" in err:
        return False, "stockout"
    if "already exists" in err:
        return False, "already exists"
    return False, err.strip().splitlines()[-1][:120] if err.strip() else "unknown error"


def wait_ready(zone, t0, minutes=12):
    """Wait for a DEPTH_READY line stamped after t0 (older boots leave lines in the buffer)."""
    print(f"waiting for the VM to become ready (up to {minutes} min) ", end="", flush=True)
    deadline = time.time() + minutes * 60
    while time.time() < deadline:
        r = gcloud("compute", "instances", "get-serial-port-output", NAME,
                   "--project", PROJECT, "--zone", zone, check=False)
        for line in (r.stdout or "").splitlines():
            if "DEPTH_READY" in line:
                line = line[line.index("DEPTH_READY"):]
                ts = line.split("ts=")[1].split()[0] if "ts=" in line else None
                if ts is None or int(ts) >= t0 - 120:
                    print()
                    return line
        print(".", end="", flush=True)
        time.sleep(10)
    print()
    return None


def set_default_zone(zone):
    # Only touch the gcloud default zone when this project is the default project,
    # so the launcher never changes someone's settings for other projects.
    r = gcloud("config", "get", "core/project", check=False)
    if r.stdout.strip() == PROJECT:
        gcloud("config", "set", "compute/zone", zone, check=False)
        return f"(zone {zone} is now your gcloud default, so --zone can be left out)"
    return f"(your gcloud default project is not {PROJECT}; pass --zone={zone} explicitly)"


def finish(zone, line):
    note = set_default_zone(zone)
    print(line or "no ready line seen within the wait; check `python gpu.py status`")
    print(f"\nready: gcloud compute ssh {NAME} --zone={zone}   {note}")


def hunt(args, image, home_source):
    startup, shutdown = write_temp(STARTUP_SCRIPT), write_temp(SHUTDOWN_SCRIPT)
    try:
        tiers = candidates(args)
        print(f"looking for a GPU ({len(tiers)} zone/shape combinations)...")
        for zone, machine, accel, model in tiers:
            label = f"{accel.replace('nvidia-', '').replace('tesla-', '').upper()} {machine} {model.lower()} in {zone}"
            print(f"  {label:50s} ", end="", flush=True)
            t0 = int(time.time())
            ok, why = try_create(zone, machine, accel, model, startup, shutdown, image, home_source)
            if ok:
                print("CREATED")
                finish(zone, wait_ready(zone, t0))
                return
            print(why)
            if why == "already exists":
                sys.exit("an instance with this name exists - run `python gpu.py status`")
        sys.exit("\nNo GPU capacity found anywhere in the list right now. Try again in a few minutes"
                 + ("" if args.us else ", or add --us to include US zones") + ".")
    finally:
        for p in (startup, shutdown):
            try:
                os.remove(p)
            except OSError:
                pass


def cmd_up(args):
    inst = find_instance()
    if not inst:
        print(f"no {NAME} VM exists; creating one from the latest image (home dir comes from the bucket)")
        hunt(args, image=None, home_source="bucket")
        return
    zone, status, _ = inst
    if status == "RUNNING":
        print(f"{NAME} is already running in {zone} {set_default_zone(zone)}")
        return
    if not args.fresh:
        print(f"{NAME} is stopped in {zone}; restarting it in place (nothing to copy)...")
        for attempt in range(1, 4):
            while status == "STOPPING":
                time.sleep(15)
                status = (find_instance() or (zone, "", ""))[1]
            t0 = int(time.time())
            r = gcloud("compute", "instances", "start", NAME, "--project", PROJECT, "--zone", zone,
                       check=False, timeout=300)
            if r.returncode == 0:
                finish(zone, wait_ready(zone, t0))
                return
            print(f"  attempt {attempt}: no GPU capacity in {zone} right now")
            if attempt < 3:
                time.sleep(20)
        print("moving the VM to another zone instead.")
    else:
        print(f"{NAME} is stopped in {zone}; --fresh given, so it is moved to a new VM.")
    image = make_image(zone, "so the new VM starts from exactly where you left off")
    print(f"deleting the old VM in {zone}...")
    gcloud("compute", "instances", "delete", NAME, "--project", PROJECT, "--zone", zone, "--quiet", timeout=600)
    hunt(args, image=image, home_source="local")


def cmd_down(args):
    inst = find_instance()
    if not inst:
        print(f"no {NAME} VM exists - nothing to do")
        return
    zone, status, _ = inst
    if status == "RUNNING":
        print(f"syncing the home dir to the bucket ({zone})...")
        r = gcloud("compute", "ssh", NAME, "--project", PROJECT, "--zone", zone, "--quiet", *ssh_args(),
                   "--command", "sudo /opt/sync-home.sh", check=False, timeout=3600)
        if r.returncode != 0:
            print("  sync over SSH failed:", (r.stderr or "").strip().splitlines()[-1:] or "")
            print("  (the VM also syncs while stopping, but that has a 90-second limit)")
            if not args.yes and input("  Continue anyway? [y/N] ").lower() != "y":
                sys.exit("aborted - the VM is still running")
        else:
            print("  " + (r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "synced"))
        print(f"stopping {NAME} in {zone}...")
        gcloud("compute", "instances", "stop", NAME, "--project", PROJECT, "--zone", zone, "--quiet", timeout=600)
    elif not args.delete:
        print(f"{NAME} is already stopped in {zone}")
    if args.delete:
        if not args.no_save:
            make_image(zone, "so nothing is lost")
        print(f"deleting {NAME} in {zone}...")
        gcloud("compute", "instances", "delete", NAME, "--project", PROJECT, "--zone", zone, "--quiet", timeout=600)
        print("done - nothing is running or billed except the bucket and the images")
    else:
        print("done - only the disk (about 7 kr/day) is billed; it is deleted automatically after 7 idle days,\n"
              "or now with `python gpu.py down --delete`")


def cmd_status(args):
    inst = find_instance()
    if inst:
        zone, status, last_stop = inst
        r = gcloud("compute", "instances", "describe", NAME, "--project", PROJECT, "--zone", zone,
                   "--format", "value(machineType.basename(),guestAccelerators[0].acceleratorType.basename(),"
                               "scheduling.provisioningModel)")
        print(f"{NAME}: {status} in {zone}  ({r.stdout.strip().replace(chr(9), ', ')})")
        if status == "TERMINATED":
            stopped = parse_ts(last_stop)
            age = (dt.datetime.now(dt.timezone.utc) - stopped).days if stopped else "?"
            print(f"  stopped {age} day(s) ago; disk kept (~7 kr/day), auto-deleted after 7 idle days.")
    else:
        print(f"{NAME}: no VM (nothing billed except images and the bucket)")
    print(f"last home-dir sync to bucket: {last_sync() or 'never'}")
    imgs = list_images()
    print("images (newest is used for new VMs): " + (", ".join(f"{n} ({t[:16]})" for n, t in imgs) or "none"))


def cmd_ssh(args):
    inst = find_instance()
    if not inst or inst[1] != "RUNNING":
        sys.exit(f"{NAME} is not running - `python gpu.py up` first")
    sys.exit(subprocess.call([GCLOUD, "compute", "ssh", NAME, "--project", PROJECT, "--zone", inst[0], *ssh_args()]))


def cmd_save_env(args):
    inst = find_instance()
    if not inst:
        sys.exit(f"{NAME} does not exist - nothing to capture")
    name = make_image(inst[0], "on request")
    print(f"done - future VMs start from {name}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    up = sub.add_parser("up", help="restart the VM, or find a GPU and (re)create it")
    up.add_argument("--fresh", action="store_true", help="do not restart in place; move to a new VM")
    up.add_argument("--l4-only", action="store_true", help="do not fall back to a T4")
    up.add_argument("--spot", action="store_true", help="try Spot L4s first (cheaper, may be interrupted)")
    up.add_argument("--us", action="store_true", help="also try US zones")
    up.add_argument("-y", "--yes", action="store_true", help="do not ask for confirmation")
    down = sub.add_parser("down", help="sync and stop the VM (disk kept)")
    down.add_argument("--delete", action="store_true", help="capture an image, then delete the VM and its disk")
    down.add_argument("--no-save", action="store_true", help="with --delete: skip the image capture")
    down.add_argument("-y", "--yes", action="store_true", help="do not ask for confirmation")
    sub.add_parser("status")
    sub.add_parser("ssh")
    sub.add_parser("save-env", help="capture the disk as a new image now")
    args = p.parse_args()
    {"up": cmd_up, "down": cmd_down, "status": cmd_status, "ssh": cmd_ssh, "save-env": cmd_save_env}[args.cmd](args)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        sys.exit(f"gcloud error: {e}")
    except KeyboardInterrupt:
        sys.exit("\ninterrupted")
