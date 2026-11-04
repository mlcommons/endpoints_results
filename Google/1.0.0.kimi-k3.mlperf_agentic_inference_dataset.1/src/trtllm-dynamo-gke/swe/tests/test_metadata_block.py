"""Tests for the GCE metadata block (CS-NET-001): swe-host-entry.sh, swe_probe.py, swe_pilot.py.

The marked section of the entry script runs in bash against simulated iptables (legacy and
nft backends, IPv4 and IPv6, rule counters), curl and getent: PATH shims that all point to
one small Python simulator, and a simulated /proc/net for the socket precheck. The service user
section (account files copied, groupadd/useradd/install stubbed) and prepare()'s ownership run
the same way. No root, no network beyond 127.0.0.1, no cluster access.
"""
import contextlib
import fcntl
import importlib.util
import inspect
import io
import marshal
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

SWE_DIR = Path(__file__).resolve().parents[1]
ENTRY_PATH = SWE_DIR / "swe-host-entry.sh"
ENTRY = ENTRY_PATH.read_text()
START = "# >>> metadata block (CS-NET-001) >>>\n"
END = "# <<< metadata block <<<\n"
SECTION = ENTRY.split(START, 1)[-1].split(END, 1)[0]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sp = _load("swe_pilot", SWE_DIR / "pilot" / "swe_pilot.py")
probe = _load("swe_probe", SWE_DIR / "swe_probe.py")

OURS = "KIMI-SWE-MDBLOCK"
COMMENT = "kimi-swe-host: GCE metadata block for the SWE service user"
URL4 = "http://<ip>/computeMetadata/v1/instance/id"
URL_GKE = "http://<ip>:988/"
URL6 = "http://[fd20:ce::254]/computeMetadata/v1/instance/id"
BODY = "SECRET-METADATA-BODY"
WRITES = ("-N", "-A", "-I", "-D", "-F", "-X")
IPT_TOOLS = ("iptables-legacy", "iptables-nft", "ip6tables-legacy", "ip6tables-nft")
SAVE_TOOLS = tuple(t + "-save" for t in IPT_TOOLS)
SHIMS = IPT_TOOLS + SAVE_TOOLS + ("curl", "getent", "pkill", "pgrep", "simctl")

# What `iptables -S` prints for the block of uid 61000 (the simulator renders like iptables:
# /32 and /128 prefixes, -m udp/tcp before --dport, the comment in double quotes).
JUMP = f'-A OUTPUT -m owner --uid-owner 61000 -m comment --comment "{COMMENT}" -j {OURS}'
RULES4 = [JUMP,
          f"-A {OURS} -d <ip>/32 -p udp -m udp --dport 53 -j RETURN",
          f"-A {OURS} -d <ip>/32 -p tcp -m tcp --dport 53 -j RETURN",
          f"-A {OURS} -d <ip>/32 -j REJECT --reject-with icmp-port-unreachable",
          f"-A {OURS} -d <ip>/32 -j REJECT --reject-with icmp-port-unreachable"]
RULES6 = [JUMP,
          f"-A {OURS} -d fd20:ce::254/128 -p udp -m udp --dport 53 -j RETURN",
          f"-A {OURS} -d fd20:ce::254/128 -p tcp -m tcp --dport 53 -j RETURN",
          f"-A {OURS} -d fd20:ce::254/128 -j REJECT --reject-with icmp6-port-unreachable"]

# ----------------------------------------------------------------------------- simulator
# One script behind every shim; it dispatches on its own name. State: a marshal file (fast to
# load: every iptables/curl call is a new process). Both backends' OUTPUT chains apply to a
# packet, as in the kernel: a REJECT in either refuses it.
SIM = r'''
import fcntl, marshal, os, sys

TOOL = os.path.basename(sys.argv[0])
ARGS = sys.argv[1:]
ENV = os.environ
ST = ENV["SIM_STATE"]
OURS = "KIMI-SWE-MDBLOCK"
BUILTIN = ("INPUT", "FORWARD", "OUTPUT")
VERDICTS = ("ACCEPT", "DROP", "REJECT", "RETURN")
WHO = ENV.get("SIM_AS", "root")


def norm(tokens, fam):
    out, i = [], 0
    while i < len(tokens):
        t = tokens[i]
        if t in ("-d", "-s") and i + 1 < len(tokens):
            a = tokens[i + 1]
            if "/" not in a:
                a += "/32" if fam == "4" else "/128"
            out += [t, a]
            i += 2
            continue
        if t == "-p" and i + 1 < len(tokens):
            out += [t, tokens[i + 1]]
            i += 2
            if i < len(tokens) and tokens[i] == "--dport":
                out += ["-m", tokens[i - 1]]
            continue
        out.append(t)
        i += 1
    return out


def render(tokens):
    return " ".join('"%s"' % t if (" " in t or not t) else t for t in tokens)


def target(spec):
    return spec[spec.index("-j") + 1] if "-j" in spec else None


def violation(st, msg):
    st["violations"].append("%s %s: %s" % (TOOL, " ".join(ARGS), msg))


def err(msg, rc=1):
    sys.stderr.write("iptables: %s\n" % msg)
    return rc


def lock_leaked():
    path = ENV.get("SIM_LOCK_PATH", "")
    for fd in os.listdir("/proc/self/fd"):
        try:
            if os.readlink("/proc/self/fd/" + fd) == path:
                return True
        except OSError:
            pass
    return False


def save(st, fam, be):
    if fam == "6" and ENV.get("SIM_V6_BROKEN") == "1":
        return 1
    t = st["tables"][be][fam]
    out = ["*filter"]
    out += [":%s %s [0:0]" % (c, "ACCEPT" if c in BUILTIN else "-") for c in t["chains"]]
    out += ["-A %s %s" % (c, render(r[0])) for c, rs in t["chains"].items() for r in rs]
    out.append("COMMIT")
    if t["nat"]:
        out += ["*nat", ":PREROUTING ACCEPT [0:0]"]
        out += ["-A PREROUTING -d 10.0.%d.%d/32 -j ACCEPT" % (i // 250, i % 250) for i in range(t["nat"])]
        out.append("COMMIT")
    print("\n".join(out))
    return 0


def ipt(st, fam, be):
    if fam == "6" and ENV.get("SIM_V6_BROKEN") == "1":
        return err("can't initialize ip6tables table `filter': Table does not exist", 3)
    a = list(ARGS)
    if "-w" in a:
        i = a.index("-w")
        del a[i]
        if i < len(a) and a[i].isdigit():
            del a[i]
    else:
        violation(st, "no -w")
    if "-t" in a:
        i = a.index("-t")
        if a[i + 1] != "filter":
            violation(st, "a table other than filter")
        del a[i:i + 2]
    fail = ENV.get("SIM_FAIL_MATCH")
    if fail and ENV.get("SIM_FAIL_TOOL", TOOL) == TOOL and fail in " ".join(ARGS):
        return err("simulated failure")
    chains = st["tables"][be][fam]["chains"]
    cmd, rest = a[0], a[1:]
    if cmd == "-S":
        names = rest[:1] or list(chains)
        if names[0] not in chains:
            return err("No chain/target/match by that name.")
        out = [("-P %s ACCEPT" % c) if c in BUILTIN else ("-N " + c) for c in names]
        out += ["-A %s %s" % (c, render(r[0])) for c in names for r in chains[c]]
        print("\n".join(out))
        return 0
    if cmd == "-L":
        c = rest[0]
        if c not in chains:
            return err("No chain/target/match by that name.")
        print("Chain %s (1 references)" % c)
        print("    pkts      bytes target     prot opt in     out     source               destination")
        for spec, pkts, nbytes in chains[c]:
            print("%8d %10d %-10s all  --  *      *       0.0.0.0/0            0.0.0.0/0"
                  % (pkts, nbytes, target(spec) or ""))
        return 0
    c, tail = rest[0], rest[1:]
    if cmd in ("-N", "-F", "-X") and c != OURS:
        violation(st, "%s on a chain that is not ours" % cmd)
        return err("refused by the simulator")
    if cmd == "-N":
        if c in chains:
            return err("Chain already exists.")
        chains[c] = []
        return 0
    if c not in chains:
        return err("No chain/target/match by that name.")
    if cmd == "-F":
        chains[c] = []
        return 0
    if cmd == "-X":
        if chains[c]:
            return err("Directory not empty.")
        if any(target(r[0]) == c for rs in chains.values() for r in rs):
            return err("Too many links.")
        del chains[c]
        return 0
    pos = 0
    if cmd == "-I" and tail and tail[0].isdigit():
        pos, tail = int(tail[0]) - 1, tail[1:]
    spec = norm(tail, fam)
    tg = target(spec)
    if cmd == "-D":
        if not (c == OURS or (c == "OUTPUT" and tg == OURS)):
            violation(st, "deletes a rule that is not ours")
        for i, r in enumerate(chains[c]):
            if r[0] == spec:
                del chains[c][i]
                return 0
        return err("Bad rule (does a matching rule exist in that chain?).")
    if cmd == "-A" and c != OURS:
        violation(st, "appends to a chain that is not ours")
    if cmd == "-I" and not (c == "OUTPUT" and tg == OURS):
        violation(st, "inserts a rule that is not ours")
    if cmd not in ("-A", "-I"):
        violation(st, "unexpected command")
        return err("unsupported by the simulator")
    if tg not in VERDICTS and tg not in chains:
        return err("Couldn't load target `%s'" % tg)
    if cmd == "-A":
        chains[c].append([spec, 0, 0])
    else:
        chains[c].insert(pos, [spec, 0, 0])
    return 0


def matches(spec, uid, dst, port):
    i = 0
    while i < len(spec):
        t = spec[i]
        if t == "-j":
            return True
        val = spec[i + 1] if i + 1 < len(spec) else ""
        if t == "-d":
            if val.split("/")[0] != dst:
                return False
        elif t == "-p":
            if val not in ("tcp", "all"):
                return False
        elif t == "--dport":
            if val != str(port):
                return False
        elif t == "--uid-owner":
            if val != str(uid):
                return False
        elif t not in ("-m", "--comment"):
            return False          # negations, conntrack, ...: never match here
        i += 2
    return True


def walk(chains, chain, fam, uid, dst, port, depth):
    if depth > 8:
        return None
    for r in chains.get(chain, []):
        if not matches(r[0], uid, dst, port):
            continue
        tg = target(r[0])
        if tg == "RETURN":
            return None
        if tg in ("REJECT", "DROP"):
            r[1] += 1
            r[2] += 60 if fam == "4" else 80
            return tg
        if tg == "ACCEPT":
            return tg
        res = walk(chains, tg, fam, uid, dst, port, depth + 1)
        if res:
            return res
    return None


def curl(st):
    a = ARGS
    if not a or a[0] != "-q":
        violation(st, "curl: -q is not the first option")
    withval = ("-o", "--noproxy", "--connect-timeout", "-m", "-H", "-w", "--retry", "--retry-delay")
    url = out = None
    wfmt, headers, i = "", [], 0
    while i < len(a):
        if a[i] in withval:
            if a[i] == "-H":
                headers.append(a[i + 1])
            elif a[i] == "-o":
                out = a[i + 1]
            elif a[i] == "-w":
                wfmt = a[i + 1]
            i += 2
        elif a[i].startswith("-"):
            i += 1
        else:
            url = a[i]
            i += 1
    st["urls"].append([WHO, url])
    if out != "/dev/null":
        violation(st, "curl: the response body is not discarded")
    hostport = url.split("://", 1)[1].partition("/")[0]
    if hostport.startswith("["):
        host, _, p = hostport[1:].partition("]")
        port, fam = (int(p[1:]) if p else 80), "6"
    else:
        host, _, p = hostport.partition(":")
        port, fam = (int(p) if p else 80), "4"
    uid = int(ENV.get("SIM_USER_UID", "61000")) if WHO == "user" else 0
    fault = ENV.get("SIM_FAULT", "")
    # dnat: redirected past our rule and answered; drop: redirected past our rule and dropped.
    bypass = fault in ("dnat", "drop") and WHO == "user" and host == "<ip>"
    dst = "<ip>" if bypass else host
    if any(walk(st["tables"][be][fam]["chains"], "OUTPUT", fam, uid, dst, port, 0) in ("REJECT", "DROP")
           for be in ("legacy", "nft")):
        code, rc = "000", 7
    elif fam == "6" and ENV.get("SIM_V6_REACHABLE") != "1":
        code, rc = "000", int(ENV.get("SIM_V6_EXIT", "7"))     # no IPv6 route on this node
    elif fault == "drop" and bypass:
        code, rc = "000", 28              # curl times out
    elif fault == "metadata_down" and host == "<ip>":
        code, rc = "000", 28
    elif "Metadata-Flavor: Google" not in headers:
        code, rc = "403", 0
    else:
        code, rc = "200", 0
    if code != "000" and out != "/dev/null":
        sys.stdout.write("SECRET-METADATA-BODY")
    sys.stdout.write(wfmt.replace("%{http_code}", code))
    return rc


def getent(st):
    if WHO != "user":
        violation(st, "the DNS check must run as the service user")
    if ENV.get("SIM_FAULT") == "dns":
        return 2
    print("<ip>      registry.gitlab.com")
    return 0


def simctl(st):
    if ARGS[0] == "drop-jump":
        for be in ("legacy", "nft"):
            out = st["tables"][be][ARGS[1]]["chains"]["OUTPUT"]
            out[:] = [r for r in out if target(r[0]) != OURS]
        return 0
    if ARGS[0] == "snapshot":
        st["snapshots"].append(marshal.loads(marshal.dumps(st["tables"])))
        return 0
    return 2


def dispatch(st):
    if WHO == "user" and lock_leaked():
        violation(st, "the node lock descriptor reached a service-user process")
    if TOOL == "curl":
        return curl(st)
    if TOOL == "getent":
        return getent(st)
    if TOOL in ("pkill", "pgrep"):
        return 1
    if TOOL == "simctl":
        return simctl(st)
    name = TOOL[:-5] if TOOL.endswith("-save") else TOOL
    fam = "6" if name.startswith("ip6tables-") else "4"
    be = name.split("-", 1)[1]
    return save(st, fam, be) if TOOL.endswith("-save") else ipt(st, fam, be)


def main():
    with open(ST + ".lock", "a") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        with open(ST, "rb") as f:
            st = marshal.load(f)
        st["calls"].append([TOOL, WHO] + ARGS)
        rc = dispatch(st)
        with open(ST + ".tmp", "wb") as f:
            marshal.dump(st, f)
        os.replace(ST + ".tmp", ST)
    sys.stdout.flush()
    return rc


sys.exit(main())
'''

# The entry script's globals that the marked section uses; everything else must be inside it.
HARNESS_HEAD = r"""set -euo pipefail
log() { printf '[swe-host] %s\n' "$*"; }
die() { log "FATAL: $*"; exit 1; }
STATE=$SIM_STATE_DIR
RUN_AS=kimiswe
RUN_UID=${SIM_RUN_UID:-61000}
METADATA_BLOCK=${SIM_MDB:-1}
AS_USER=(env SIM_AS=user)
TERM_SEEN=0
trap 'TERM_SEEN=1' TERM INT
"""
APPLY = "node_lock\nmetadata_block_apply\nsimctl snapshot\n"
FULL = APPLY + "metadata_block_remove\n"
# A sandbox's attempt while the service runs (as the service user, like every sandbox).
ATTEMPT = ("\"${AS_USER[@]}\" curl -q -sS -o /dev/null -H 'Metadata-Flavor: Google' "
           "-w '%{http_code}' " + URL4 + " {NODE_LOCK_FD}<&- || true\n")


def rule(*tokens):
    return [list(tokens), 0, 0]


def seed_chains(fam):
    lo = "127.0.0.0/8" if fam == "4" else "::1/128"
    dns = "<ip>/32" if fam == "4" else "fd00::a/128"
    rej = "icmp-port-unreachable" if fam == "4" else "icmp6-port-unreachable"
    return {
        "INPUT": [rule("-j", "KUBE-FIREWALL")],
        "FORWARD": [rule("-m", "comment", "--comment", "kubernetes forwarding rules", "-j", "KUBE-FORWARD")],
        "OUTPUT": [rule("-m", "comment", "--comment", "kubernetes service portals", "-j", "KUBE-SERVICES"),
                   rule("-j", "KUBE-FIREWALL")],
        "KUBE-FIREWALL": [rule("!", "-s", lo, "-d", lo, "-m", "comment", "--comment",
                               "block incoming localnet connections", "-m", "conntrack", "!",
                               "--ctstate", "RELATED,ESTABLISHED,DNAT", "-j", "DROP")],
        "KUBE-FORWARD": [rule("-m", "conntrack", "--ctstate", "INVALID", "-j", "DROP")],
        "KUBE-SERVICES": [rule("-d", dns, "-p", "tcp", "-m", "comment", "--comment",
                               "kube-system/kube-dns:dns-tcp has no endpoints", "-m", "tcp",
                               "--dport", "53", "-j", "REJECT", "--reject-with", rej)],
    }


def make_state(active="nft", tie=False):
    tables = {}
    for be in ("legacy", "nft"):
        busy = tie or be == active
        tables[be] = {fam: {"chains": seed_chains(fam) if busy else {"INPUT": [], "FORWARD": [], "OUTPUT": []},
                            "nat": 40 if busy else 0} for fam in ("4", "6")}
    return {"tables": tables, "calls": [], "urls": [], "violations": [], "snapshots": []}


def stale_jump(uid):
    return rule("-m", "owner", "--uid-owner", str(uid), "-m", "comment", "--comment", COMMENT, "-j", OURS)


def specs(tables):
    """Tables without counters, for comparisons."""
    return {be: {fam: {c: [r[0] for r in rs] for c, rs in t["chains"].items()}
                 for fam, t in fams.items()} for be, fams in tables.items()}


def rendered(tables, be, fam):
    """Our rules in `iptables -S` form: the jump(s) to our chain, then the chain."""
    ch = tables[be][fam]["chains"]
    render = lambda spec: " ".join(f'"{t}"' if " " in t else t for t in spec)
    out = [f"-A OUTPUT {render(r[0])}" for r in ch["OUTPUT"] if r[0][-2:] == ["-j", OURS]]
    return out + [f"-A {OURS} {render(r[0])}" for r in ch.get(OURS, [])]


# The node's socket tables (/proc/net/*) for the socket precheck, in the kernel's layout: column 8
# is the owner's uid, column 10 the inode (0: a closing socket no process holds). By default: root's
# sockets, GKE agents' (uid 1000 and 1001; one uid 1000 socket is closing) and none of 61000.
PROC_HEAD4 = ("  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  "
              "timeout inode")
PROC_HEAD6 = ("  sl  local_address                         remote_address                        st "
              "tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode")


def sock(table: str, uid: int, inode: int, sl: int = 0, port: int = 0x2382) -> str:
    """One row of /proc/net/<table> (tcp* rows carry TCP's extra columns, the others ref/pointer/drops)."""
    v6 = table.endswith("6")
    local = ("00000000000000000000000001000000" if v6 else "0100007F") + f":{port:04X}"
    remote = ("0" * 32 if v6 else "00000000") + ":0000"
    tail = "1 0000000000000000 100 0 0 10 0" if table.startswith("tcp") else "2 0000000000000000 0"
    return (f"{sl:4d}: {local} {remote} 0A 00000000:00000000 00:00000000 00000000 {uid:5d}        0 "
            f"{inode} {tail}")


DEFAULT_SOCKETS = {
    "tcp": [sock("tcp", 0, 11111), sock("tcp", 1000, 22222, sl=1), sock("tcp", 1000, 0, sl=2)],
    "tcp6": [sock("tcp6", 0, 33333), sock("tcp6", 1000, 44444, sl=1)],
    "udp": [sock("udp", 0, 55555), sock("udp", 1001, 66666, sl=1)],
    "udp6": [sock("udp6", 0, 12121)],
    "raw": [],
    "raw6": [],
}


def write_procnet(d: Path, tables: dict | None = None, tcp_header: str | None = None) -> None:
    d.mkdir(exist_ok=True)
    for name, rows in (DEFAULT_SOCKETS if tables is None else tables).items():
        head = PROC_HEAD6 if name.endswith("6") else PROC_HEAD4
        if name == "tcp" and tcp_header is not None:
            head = tcp_header
        (d / name).write_text("\n".join([head] + rows) + "\n")


class Sim:
    def __init__(self, tmp: str):
        self.dir = Path(tmp)
        self.bin = self.dir / "bin"
        self.bin.mkdir()
        sim = self.bin / "sim.py"
        sim.write_text(f"#!{sys.executable} -BSI\n" + SIM)
        sim.chmod(0o755)
        for name in SHIMS:
            (self.bin / name).symlink_to(sim)
        self.state_dir = self.dir / "state"
        self.state_dir.mkdir()
        self.lock = self.state_dir / "host-entry.lock"
        self.state_file = self.dir / "sim.state"
        self.procnet = self.dir / "procnet"
        write_procnet(self.procnet)

    def run(self, steps: str, state: dict, **env):
        self.state_file.write_bytes(marshal.dumps(state))
        script = self.dir / "harness.sh"
        script.write_text(HARNESS_HEAD + SECTION + "MDB_PROC_NET=$SIM_PROC_NET\n"
                          + "trap exit_cleanup EXIT\n" + steps)
        e = {"PATH": f"{self.bin}:{os.environ['PATH']}", "HOME": str(self.dir), "LC_ALL": "C",
             "SIM_STATE": str(self.state_file), "SIM_STATE_DIR": str(self.state_dir),
             "SIM_LOCK_PATH": str(self.lock), "SIM_PROC_NET": str(self.procnet), **env}
        p = subprocess.run(["bash", str(script)], env=e, capture_output=True, text=True,
                           timeout=120, check=False)
        return p, marshal.loads(self.state_file.read_bytes())


class SimCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.sim = Sim(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def run_sim(self, steps, state=None, **env):
        p, st = self.sim.run(steps, state or make_state(), **env)
        self.out = p.stdout + p.stderr
        return p, st

    def assertClean(self, st, p):
        """Common invariants: no rule hygiene or lock violation, no body, allowed URLs only."""
        self.assertEqual(st["violations"], [], self.out)
        self.assertNotIn(BODY, self.out)
        for who, url in st["urls"]:
            self.assertIn(url, (URL4, URL_GKE, URL6), (who, url))
            for word in ("token", "service-accounts", "attributes"):
                self.assertNotIn(word, url)

    @staticmethod
    def writes(st, tool_prefix=""):
        return [c for c in st["calls"] if c[0].startswith(tool_prefix) and c[0] in IPT_TOOLS
                and any(a in WRITES for a in c[2:])]

    def lines(self, prefix):
        return [l.split("] ", 1)[1] for l in self.out.splitlines() if l.startswith("[swe-host] ")
                and l.split("] ", 1)[1].startswith(prefix)]


# ----------------------------------------------------------------------------- behaviour
class MetadataBlockSimulated(SimCase):
    def test_happy_path_nft_dual_stack(self):
        state = make_state("nft")
        p, st = self.run_sim(APPLY + ATTEMPT + "metadata_block_remove\n", state)
        self.assertEqual(p.returncode, 0, self.out)
        self.assertClean(st, p)
        after_apply = st["snapshots"][0]
        self.assertEqual(rendered(after_apply, "nft", "4"), RULES4)
        self.assertEqual(rendered(after_apply, "nft", "6"), RULES6)
        for fam in ("4", "6"):          # the jump is the first rule of OUTPUT
            self.assertEqual(after_apply["nft"][fam]["chains"]["OUTPUT"][0][0][-1], OURS)
        self.assertEqual(specs(after_apply)["legacy"], specs(state["tables"])["legacy"])
        # The log shows the rules exactly as iptables -S prints them.
        self.assertEqual(self.lines("metadata block: IPv4 rule: "), ["metadata block: IPv4 rule: " + r for r in RULES4])
        self.assertEqual(self.lines("metadata block: IPv6 rule: "), ["metadata block: IPv6 rule: " + r for r in RULES6])
        # Nothing else on the node changed, and nothing of ours is left.
        self.assertEqual(specs(st["tables"]), specs(state["tables"]))
        for want in ("metadata block: iptables backend nft (rules on the node: nft 94, legacy 0;",
                     "metadata block: before the block, kimiswe gets HTTP 200 (curl exit 0)",
                     f"metadata block self-test: root -> {URL4}: HTTP 200: ok",
                     f"metadata block self-test: kimiswe -> {URL4}: refused (HTTP 000, curl exit 7): ok",
                     f"metadata block self-test: kimiswe -> {URL_GKE}: refused (HTTP 000, curl exit 7): ok",
                     f"metadata block self-test: kimiswe -> {URL6}: refused (HTTP 000, curl exit 7): ok",
                     "metadata block self-test: our REJECT rules counted 3 refused packet(s): ok",
                     "registry.gitlab.com resolves: ok",
                     "metadata block: 0 sockets of uid 61000 in the node's network namespace before the "
                     "service starts (6 socket tables, IPv4 and IPv6): ok",
                     "metadata block: 2 socket(s) of uid 1000 in the node's network namespace (GKE runs "
                     "hostNetwork agents as uid 1000; informational: the block does not match them)",
                     "metadata block: active for uid 61000 (IPv4 and IPv6, iptables-nft); self-test passed",
                     "metadata block: 1 packet(s) from uid 61000 to the metadata servers refused while the service ran",
                     "POSTCHECK metadata block still in place at teardown (kimiswe -> <ip> refused): ok",
                     f"metadata block removed: no {OURS} chain and no OUTPUT jump to it (IPv4 and IPv6, iptables-nft)"):
            self.assertIn(want, self.out)
        self.assertNotIn("removed on exit", self.out)
        self.assertNotIn("WARNING", self.out)
        # Root never asked the GKE server; the service user asked only through AS_USER.
        self.assertNotIn(["root", URL_GKE], st["urls"])
        self.assertEqual(sum(1 for c in st["calls"] if c[0] == "getent" and c[1] == "user"), 1)

    def test_legacy_node_uses_legacy_only(self):
        state = make_state("legacy")
        p, st = self.run_sim(FULL, state)
        self.assertEqual(p.returncode, 0, self.out)
        self.assertClean(st, p)
        self.assertIn("iptables backend legacy (rules on the node: nft 0, legacy 94;", self.out)
        self.assertEqual(rendered(st["snapshots"][0], "legacy", "4"), RULES4)
        self.assertEqual(self.writes(st, "iptables-nft") + self.writes(st, "ip6tables-nft"), [])
        self.assertTrue(self.writes(st, "iptables-legacy"))
        self.assertEqual(specs(st["tables"]), specs(state["tables"]))
        self.assertIn("(IPv4 and IPv6, iptables-legacy)", self.out)

    def test_every_iptables_call_waits_for_the_xtables_lock(self):
        p, st = self.run_sim(FULL)
        calls = [c for c in st["calls"] if c[0] in IPT_TOOLS]
        self.assertTrue(calls)
        self.assertTrue(all(c[2:4] == ["-w", "10"] for c in calls), calls)

    def test_tie_fails_closed_without_writes(self):
        p, st = self.run_sim(FULL, make_state(tie=True))
        self.assertEqual(p.returncode, 1, self.out)
        self.assertIn("FATAL: metadata block: cannot tell which iptables backend the node uses "
                      "(rules: nft 94, legacy 94)", self.out)
        self.assertEqual(self.writes(st), [])
        self.assertEqual([u for u in st["urls"]], [])
        self.assertNotIn("removed on exit", self.out)

    def test_disabled_logs_a_warning_and_touches_nothing(self):
        p, st = self.run_sim(FULL, SIM_MDB="0")
        self.assertEqual(p.returncode, 0, self.out)
        self.assertIn("WARNING: SWE_METADATA_BLOCK=0: no metadata block.", self.out)
        self.assertIn("WARNING: metadata block: none in this run (SWE_METADATA_BLOCK=0)", self.out)
        self.assertEqual([c for c in st["calls"] if c[0] != "simctl"], [])

    def test_root_service_user_is_refused(self):
        p, st = self.run_sim(FULL, SIM_RUN_UID="0")
        self.assertEqual(p.returncode, 1, self.out)
        self.assertIn("FATAL: the metadata block needs an unprivileged service user", self.out)
        self.assertEqual([c for c in st["calls"] if c[0] != "simctl"], [])

    def test_stale_block_is_replaced(self):
        state = make_state("nft")
        out4 = state["tables"]["nft"]["4"]["chains"]
        out4["OUTPUT"][1:1] = [stale_jump(1000)]      # e.g. a v1 host pod's (appuser)
        out4["OUTPUT"].append(stale_jump(1234))
        out4[OURS] = [rule("-d", "<ip>/32", "-j", "REJECT", "--reject-with", "icmp-port-unreachable")]
        state["tables"]["nft"]["6"]["chains"][OURS] = []          # a chain without a jump
        p, st = self.run_sim(FULL, state)
        self.assertEqual(p.returncode, 0, self.out)
        self.assertClean(st, p)
        self.assertIn(f"removing a stale IPv4 {OURS} left by an earlier host pod", self.out)
        self.assertIn(f"removing a stale IPv6 {OURS} left by an earlier host pod", self.out)
        self.assertEqual(rendered(st["snapshots"][0], "nft", "4"), RULES4)
        self.assertEqual(rendered(st["snapshots"][0], "nft", "6"), RULES6)
        self.assertEqual(specs(st["tables"]), specs(make_state("nft")["tables"]))

    def test_stale_chain_used_by_someone_else_fails_closed(self):
        state = make_state("nft")
        chains = state["tables"]["nft"]["4"]["chains"]
        chains[OURS] = []
        chains["FORWARD"].append(rule("-j", OURS))                 # not ours to delete
        p, st = self.run_sim(FULL, state)
        self.assertEqual(p.returncode, 1, self.out)
        self.assertIn(f"FATAL: metadata block: cannot remove the stale IPv4 {OURS}", self.out)
        self.assertEqual(st["violations"], [])
        self.assertIn(["-j", OURS], [r[0] for r in st["tables"]["nft"]["4"]["chains"]["FORWARD"]])

    def test_bypass_is_detected_and_the_exit_trap_cleans_up(self):
        # The service user's request is redirected past our rule (like a DNAT): it must fail.
        p, st = self.run_sim(FULL, SIM_FAULT="dnat")
        self.assertEqual(p.returncode, 1, self.out)
        self.assertIn(f"FATAL: metadata block self-test: kimiswe -> {URL4} must be refused "
                      "(HTTP 000, curl exit 7), got HTTP 200, curl exit 0", self.out)
        self.assertIn(f"metadata block removed on exit: no {OURS} chain and no OUTPUT jump to it", self.out)
        self.assertIn(["pkill", "root", "-KILL", "-U", "61000"], st["calls"])
        self.assertEqual(specs(st["tables"]), specs(make_state("nft")["tables"]))
        self.assertClean(st, p)

    def test_a_timeout_is_not_our_rule(self):
        # Only "connection refused" (curl exit 7) shows that our REJECT rule answered.
        p, st = self.run_sim(FULL, SIM_FAULT="drop")
        self.assertEqual(p.returncode, 1, self.out)
        self.assertIn(f"FATAL: metadata block self-test: kimiswe -> {URL4} must be refused "
                      "(HTTP 000, curl exit 7), got HTTP 000, curl exit 28", self.out)
        self.assertIn("metadata block removed on exit", self.out)
        self.assertEqual(specs(st["tables"]), specs(make_state("nft")["tables"]))
        self.assertClean(st, p)

    def test_a_refusal_by_another_rule_is_not_enough(self):
        # Someone else's rule (here in the other backend) refuses one probe before ours does:
        # every probe fails, but our counters show that our rules answered only two of three.
        state = make_state("nft")
        state["tables"]["legacy"]["4"]["chains"]["OUTPUT"].append(
            rule("-m", "owner", "--uid-owner", "61000", "-d", "<ip>/32", "-j", "REJECT"))
        p, st = self.run_sim(FULL, state)
        self.assertEqual(p.returncode, 1, self.out)
        self.assertIn("FATAL: metadata block self-test: our REJECT rules counted 2 refused packet(s) "
                      "for 3 refused request(s)", self.out)
        self.assertIn("metadata block removed on exit", self.out)
        self.assertEqual(specs(st["tables"]), specs(state["tables"]))
        self.assertClean(st, p)

    def test_root_must_still_reach_the_metadata_server(self):
        p, st = self.run_sim(FULL, SIM_FAULT="metadata_down")
        self.assertEqual(p.returncode, 1, self.out)
        self.assertIn("FATAL: metadata block self-test: root must get HTTP 200 from <ip>, "
                      "got HTTP 000 (curl exit 28)", self.out)
        self.assertIn("metadata block removed on exit", self.out)

    def test_no_ip6tables_and_no_ipv6_route_is_fine(self):
        p, st = self.run_sim(FULL, SIM_V6_BROKEN="1")
        self.assertEqual(p.returncode, 0, self.out)
        self.assertClean(st, p)
        self.assertIn("metadata block: no IPv6 block (ip6tables-nft unusable here); the metadata server "
                      "is not reachable over IPv6 from this node (curl exit 7): ok", self.out)
        self.assertIn("active for uid 61000 (IPv4, iptables-nft); self-test passed", self.out)
        self.assertIn(f"metadata block removed: no {OURS} chain and no OUTPUT jump to it (IPv4, iptables-nft)",
                      self.out)
        self.assertEqual(self.writes(st, "ip6tables"), [])

    def test_no_ip6tables_but_ipv6_metadata_reachable_fails_closed(self):
        p, st = self.run_sim(FULL, SIM_V6_BROKEN="1", SIM_V6_REACHABLE="1")
        self.assertEqual(p.returncode, 1, self.out)
        self.assertIn("FATAL: metadata block: no IPv6 block, but root reaches the metadata server at "
                      "[fd20:ce::254] (HTTP 200)", self.out)
        self.assertIn("metadata block removed on exit", self.out)
        self.assertEqual(specs(st["tables"]), specs(make_state("nft")["tables"]))

    def test_no_ipv6_block_accepts_a_timeout(self):
        p, st = self.run_sim(FULL, SIM_V6_BROKEN="1", SIM_V6_EXIT="28")
        self.assertEqual(p.returncode, 0, self.out)
        self.assertIn("is not reachable over IPv6 from this node (curl exit 28): ok", self.out)

    def test_no_ipv6_block_and_an_unclear_curl_error_fails_closed(self):
        # curl exit 1 (say, no IPv6 support in curl) shows nothing about the node's network.
        p, st = self.run_sim(FULL, SIM_V6_BROKEN="1", SIM_V6_EXIT="1")
        self.assertEqual(p.returncode, 1, self.out)
        self.assertIn("FATAL: metadata block: no IPv6 block, and root's request to [fd20:ce::254] "
                      "failed with curl exit 1, not as unreachable (7) or timed out (28)", self.out)
        self.assertIn("metadata block removed on exit", self.out)
        self.assertEqual(specs(st["tables"]), specs(make_state("nft")["tables"]))
        self.assertClean(st, p)

    def test_a_failed_ipv6_install_is_rolled_back(self):
        p, st = self.run_sim(FULL, SIM_FAIL_TOOL="ip6tables-nft", SIM_FAIL_MATCH="-j REJECT")
        self.assertEqual(p.returncode, 0, self.out)
        self.assertClean(st, p)
        self.assertIn("WARNING: metadata block: cannot install the IPv6 rules; removing the partial "
                      "IPv6 rules", self.out)
        self.assertEqual(rendered(st["snapshots"][0], "nft", "6"), [])
        self.assertIn("active for uid 61000 (IPv4, iptables-nft); self-test passed", self.out)
        self.assertEqual(specs(st["tables"]), specs(make_state("nft")["tables"]))

    def test_dns_must_still_work_for_the_service_user(self):
        p, st = self.run_sim(FULL, SIM_FAULT="dns")
        self.assertEqual(p.returncode, 1, self.out)
        self.assertIn("FATAL: metadata block self-test: DNS as kimiswe fails", self.out)
        self.assertIn("metadata block removed on exit", self.out)

    def test_teardown_recheck_catches_a_removed_jump(self):
        steps = APPLY + "simctl drop-jump 4\nmetadata_block_remove || echo REMOVE_RC=$?\n"
        p, st = self.run_sim(steps)
        self.assertEqual(p.returncode, 0, self.out)
        self.assertIn("POSTCHECK WARNING: metadata block NOT in place at teardown: kimiswe -> "
                      "<ip> got HTTP 200, curl exit 0", self.out)
        self.assertIn("REMOVE_RC=1", self.out)
        self.assertIn(f"metadata block removed: no {OURS} chain", self.out)
        self.assertEqual(specs(st["tables"]), specs(make_state("nft")["tables"]))

    def test_exit_trap_runs_after_errexit(self):
        p, st = self.run_sim(APPLY + "false\necho NOT_REACHED\n")
        self.assertEqual(p.returncode, 1, self.out)
        self.assertNotIn("NOT_REACHED", self.out)
        self.assertIn("metadata block removed on exit", self.out)
        self.assertEqual(specs(st["tables"]), specs(make_state("nft")["tables"]))


class NodeLock(SimCase):
    def test_a_second_host_pod_on_the_node_is_refused(self):
        self.sim.lock.touch()
        with open(self.sim.lock) as held:
            fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
            p, st = self.run_sim(FULL)
        self.assertEqual(p.returncode, 1, self.out)
        self.assertIn("FATAL: another kimi-swe host pod (smoke or service) runs on this node", self.out)
        self.assertEqual([c for c in st["calls"] if c[0] != "simctl"], [])

    def test_the_lock_is_held_for_the_whole_run_and_released_at_exit(self):
        steps = ('node_lock\nif flock -n "$STATE/host-entry.lock" true; then echo SECOND_LOCK_GRANTED; '
                 'else echo SECOND_LOCK_REFUSED; fi\n')
        p, _ = self.run_sim(steps)
        self.assertEqual(p.returncode, 0, self.out)
        self.assertIn("SECOND_LOCK_REFUSED", self.out)
        self.assertEqual(self.sim.lock.stat().st_mode & 0o777, 0o600)
        with open(self.sim.lock) as f:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)            # released: no exception

    def test_the_simulator_sees_a_leaked_lock_descriptor(self):
        # Meta-test: a service-user command without {NODE_LOCK_FD}<&- is reported.
        steps = APPLY + "\"${AS_USER[@]}\" curl -q -o /dev/null -w '%{http_code}' " + URL4 + " || true\n"
        p, st = self.run_sim(steps)
        self.assertTrue(any("node lock descriptor" in v for v in st["violations"]), st["violations"])

    def test_the_simulator_sees_a_missing_wait_flag(self):
        p, st = self.run_sim("iptables-nft -S >/dev/null\n")
        self.assertTrue(any(v.endswith(": no -w") for v in st["violations"]), st["violations"])


# ----------------------------------------------------------------------------- wiring
def logical_lines(text: str) -> list[str]:
    out, buf = [], ""
    for line in text.splitlines():
        if line.rstrip().endswith("\\"):
            buf += line.rstrip()[:-1] + " "
        else:
            out.append(buf + line)
            buf = ""
    return out


class EntryScriptWiring(unittest.TestCase):
    def test_valid_bash_and_one_marked_section(self):
        p = subprocess.run(["bash", "-n", str(ENTRY_PATH)], capture_output=True, text=True, check=False)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual((ENTRY.count(START), ENTRY.count(END)), (1, 1))
        self.assertLess(ENTRY.index(END), ENTRY.index("# ------------------------------------------------------------------- service lifecycle"))

    def test_main_flow_order(self):
        top = [l for l in ENTRY.splitlines() if l and not l.startswith((" ", "#", "}"))]
        i = top.index("trap exit_cleanup EXIT")
        self.assertEqual(top[i + 1:i + 6], ["stage", "install_enroot", "node_lock", "prepare",
                                            "metadata_block_apply"])
        self.assertLess(ENTRY.index("\nmetadata_block_apply\n"), ENTRY.index("\n  load_credentials\n"))
        self.assertLess(ENTRY.index("exit_cleanup() {"), ENTRY.index("trap exit_cleanup EXIT"))

    def test_traps(self):
        self.assertEqual(ENTRY.count("trap 'TERM_SEEN=1' TERM INT\n"), 1)
        self.assertEqual(len(re.findall(r"^trap ", ENTRY, re.M)), 2)
        self.assertEqual(ENTRY.count("trap exit_cleanup EXIT\n"), 1)

    def test_teardown_removes_the_block_after_postcheck_in_both_modes(self):
        self.assertEqual(ENTRY.count("  postcheck || fail=1\n  metadata_block_remove || fail=1\n"), 2)
        self.assertEqual(ENTRY.count("metadata_block_remove || fail=1"), 2)

    def test_iptables_is_installed_and_checked(self):
        lines = logical_lines(ENTRY)
        apt = [l for l in lines if "apt-get install" in l]
        self.assertEqual(len(apt), 1)
        self.assertIn(" util-linux iptables >/dev/null", apt[0])
        loop = [l for l in lines if l.strip().startswith("for c in unshare ")]
        self.assertEqual(len(loop), 1)
        words = loop[0].replace(";", " ").split()
        for tool in IPT_TOOLS + SAVE_TOOLS:
            self.assertIn(tool, words)

    def test_toggle_default_and_validation(self):
        self.assertIn("METADATA_BLOCK=${SWE_METADATA_BLOCK:-1}\n", ENTRY)
        self.assertIn('case "$METADATA_BLOCK" in 0|1) ;; *) die "SWE_METADATA_BLOCK must be 0 or 1" ;; esac',
                      ENTRY)

    def test_no_service_user_process_gets_the_lock_descriptor(self):
        users = [l for l in logical_lines(ENTRY)
                 if '"${AS_USER[@]}"' in l and not l.lstrip().startswith("#")]
        self.assertGreaterEqual(len(users), 5)       # service, smoke probe, prewarm, curl, getent
        for line in users:
            self.assertIn("{NODE_LOCK_FD}<&-", line)

    def test_smoke_passes_the_probe_flag_only_with_the_block(self):
        self.assertIn("if [[ $METADATA_BLOCK == 1 ]]; then mdb=(--metadata-block); fi", ENTRY)
        smoke = [l for l in logical_lines(ENTRY) if "swe_probe.py" in l and " smoke " in l]
        self.assertEqual(len(smoke), 1)
        self.assertIn('"${mdb[@]}"', smoke[0])

    def test_the_section_requests_only_the_instance_id_path(self):
        self.assertIn("MDB_PATH=/computeMetadata/v1/instance/id ", SECTION)
        for word in ("token", "service-accounts", "attributes"):
            self.assertNotIn(word, SECTION.replace("node service account's token", ""))


class PilotMetadataBlock(unittest.TestCase):
    node = "gke-node-a"

    @staticmethod
    def env(pod):
        return {e["name"]: e["value"] for e in pod["spec"]["containers"][0]["env"]}

    def test_on_by_default_with_the_nodes_xtables_lock(self):
        for pod in (sp.host_pod(sp.POD["smoke"], "smoke", self.node, "smoke-1", mode="smoke"),
                    sp.host_pod(sp.POD["host"], "host", self.node, "pilot-1", mode="service"),
                    sp.host_pod(sp.POD["host"], "host", self.node, "run-1", mode="service",
                                handoff=True, reports_dir=sp.REPORTS_DIR)):
            self.assertEqual(self.env(pod)["SWE_METADATA_BLOCK"], "1")
            self.assertEqual([v for v in pod["spec"]["volumes"] if v["name"] == "xtables-lock"],
                             [{"name": "xtables-lock",
                               "hostPath": {"path": "/run/xtables.lock", "type": "FileOrCreate"}}])
            self.assertEqual([m for m in pod["spec"]["containers"][0]["volumeMounts"]
                              if m["name"] == "xtables-lock"],
                             [{"name": "xtables-lock", "mountPath": "/run/xtables.lock"}])

    def test_opt_out_is_explicit(self):
        pod = sp.host_pod(sp.POD["host"], "host", self.node, "pilot-1", mode="service",
                          metadata_block=False)
        self.assertEqual(self.env(pod)["SWE_METADATA_BLOCK"], "0")
        self.assertNotIn("xtables-lock", [v["name"] for v in pod["spec"]["volumes"]])

    def test_parser(self):
        ap = sp.parser()
        self.assertTrue(ap.parse_args(["smoke", "--node", "n", "--no-metadata-block"]).no_metadata_block)
        self.assertFalse(ap.parse_args(["smoke", "--node", "n"]).no_metadata_block)
        self.assertTrue(ap.parse_args(["host", "--node", "n", "--run-id", "r",
                                       "--no-metadata-block"]).no_metadata_block)
        self.assertFalse(ap.parse_args(["host", "--node", "n", "--run-id", "r"]).no_metadata_block)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            ap.parse_args(["serve", "--no-metadata-block"])          # production: no opt-out

    def test_production_refuses_the_opt_out_before_any_cluster_call(self):
        class NoKube:
            def __getattr__(self, name):
                raise AssertionError(f"kube.{name} used")

        for extra in (["--handoff"], ["--scan-reports"]):
            args = sp.parser().parse_args(["host", "--node", "n", "--run-id", "r",
                                           "--no-metadata-block", *extra])
            err = io.StringIO()
            with contextlib.redirect_stderr(err), self.assertRaises(SystemExit):
                sp.cmd_host(args, NoKube())
            self.assertIn("production", err.getvalue())

    def test_serve_keeps_the_default(self):
        self.assertNotIn("metadata_block", inspect.getsource(sp.serve_one))

    def test_down_host_echoes_the_block_lines(self):
        for line in (f"[swe-host 12:00:00Z] metadata block removed: no {OURS} chain and no OUTPUT "
                     "jump to it (IPv4, iptables-nft)",
                     "[swe-host 12:00:00Z] metadata block: 0 packet(s) from uid 61000 to the metadata "
                     "servers refused while the service ran"):
            self.assertTrue(sp._TEARDOWN.search(line), line)


class ProbeMetadataCheck(unittest.TestCase):
    def test_parser_flag(self):
        base = ["smoke", "--image", "x", "--drop", "d", "--out", "o"]
        ap_args = []

        def fake(args):
            ap_args.append(args)
            return 0

        with unittest.mock.patch.object(probe, "cmd_smoke", fake):
            probe.main(base)
            probe.main(base + ["--metadata-block"])
        self.assertEqual([a.metadata_block for a in ap_args], [False, True])

    def test_targets_are_validated(self):
        for bad in ((), ("<ip>/80; id",), ("$(id)/80",), ("<internal-host>/80",)):
            with self.assertRaises(ValueError):
                probe.metadata_probe_command(bad)
        cmd = probe.metadata_probe_command()
        self.assertIn("<ip>/80 <ip>/988", cmd)
        p = subprocess.run(["bash", "-n", "-c", cmd], capture_output=True, text=True, check=False)
        self.assertEqual(p.returncode, 0, p.stderr)

    def test_refused_passes_open_and_odd_outputs_fail(self):
        with socket.socket() as srv:
            srv.bind(("127.0.0.1", 0))
            srv.listen(4)
            open_t = f"127.0.0.1/{srv.getsockname()[1]}"
            with socket.socket() as s:
                s.bind(("127.0.0.1", 0))
                closed_t = f"127.0.0.1/{s.getsockname()[1]}"
            out = subprocess.run(["bash", "-c", probe.metadata_probe_command((closed_t, open_t))],
                                 capture_output=True, text=True, timeout=30, check=False).stdout
        self.assertTrue(probe.metadata_probe_ok(out, (closed_t,)), out)
        self.assertFalse(probe.metadata_probe_ok(out, (open_t,)), out)
        self.assertFalse(probe.metadata_probe_ok(out, (closed_t, open_t)), out)
        t = probe.MDB_TARGETS[0]
        for odd in ("", f"MDB {t} rc=1 bash: /dev/tcp/{t}: No such file or directory",
                    f"MDB {t} rc=124 ", f"MDB {t} rc=0 "):
            self.assertFalse(probe.metadata_probe_ok(odd, (t,)), odd)


# ----------------------------------------------------------------------------- socket precheck
class SocketPrecheck(SimCase):
    """Before any rule goes in, no live socket in the node's network namespace is the service uid's."""

    def tables(self, header=None, **changes):
        """Rewrite the simulated /proc/net: DEFAULT_SOCKETS plus extra rows; None drops a table."""
        t = {k: list(v) for k, v in DEFAULT_SOCKETS.items()}
        for name, rows in changes.items():
            if rows is None:
                t.pop(name, None)
            else:
                t[name] = t.get(name, []) + rows
        shutil.rmtree(self.sim.procnet)
        write_procnet(self.sim.procnet, t, header)

    def test_none_of_ours_passes_before_the_backend_is_chosen(self):
        p, st = self.run_sim(FULL)
        self.assertEqual(p.returncode, 0, self.out)
        self.assertClean(st, p)
        ok = ("metadata block: 0 sockets of uid 61000 in the node's network namespace before the "
              "service starts (6 socket tables, IPv4 and IPv6): ok")
        info = ("metadata block: 2 socket(s) of uid 1000 in the node's network namespace (GKE runs "
                "hostNetwork agents as uid 1000; informational: the block does not match them)")
        self.assertIn(ok, self.out)
        self.assertIn(info, self.out)
        self.assertLess(self.out.index(ok), self.out.index("metadata block: iptables backend"))
        self.assertNotIn("closing socket", self.out)

    def test_a_live_socket_of_the_service_uid_stops_before_any_rule(self):
        for table in ("tcp", "tcp6", "udp", "udp6", "raw", "raw6"):
            with self.subTest(table=table):
                self.tables(**{table: [sock(table, 61000, 77777, sl=9)]})
                p, st = self.run_sim(FULL)
                self.assertEqual(p.returncode, 1, self.out)
                self.assertIn("FATAL: metadata block: 1 socket(s) in the node's network namespace belong "
                              "to uid 61000 before the service starts: something else on this node runs as "
                              "the service uid, and the block would cut it off too. No rule installed",
                              self.out)
                self.assertEqual(st["calls"], [])           # no iptables, no curl, no pkill
                self.assertNotIn("removed on exit", self.out)

    def test_optional_tables_count_too(self):
        for table in ("icmp", "icmp6", "udplite", "udplite6"):
            with self.subTest(table=table):
                self.tables(**{table: [sock(table, 61000, 88888)]})
                p, st = self.run_sim(FULL)
                self.assertEqual(p.returncode, 1, self.out)
                self.assertIn("FATAL: metadata block: 1 socket(s) in the node's network namespace "
                              "belong to uid 61000", self.out)
                self.assertEqual(st["calls"], [])

    def test_a_closing_socket_that_no_process_holds_is_not_counted(self):
        self.tables(tcp=[sock("tcp", 61000, 0, sl=9)])
        p, st = self.run_sim(FULL)
        self.assertEqual(p.returncode, 0, self.out)
        self.assertIn("metadata block: 0 sockets of uid 61000", self.out)
        self.assertIn("metadata block: also 1 closing socket(s) of uid 61000 that no process holds any "
                      "more (left by an ended process, for example an earlier host pod's; not counted)",
                      self.out)

    def test_other_uids_are_not_ours(self):
        self.tables(tcp=[sock("tcp", u, 70000 + i, sl=10 + i)
                         for i, u in enumerate((6100, 61001, 60999, 610000, 161000, 65534))])
        p, st = self.run_sim(FULL)
        self.assertEqual(p.returncode, 0, self.out)
        self.assertIn("metadata block: 0 sockets of uid 61000", self.out)

    def test_an_ipv4_only_node(self):
        self.tables(tcp6=None, udp6=None, raw6=None)
        p, st = self.run_sim(FULL)
        self.assertEqual(p.returncode, 0, self.out)
        self.assertIn("before the service starts (3 socket tables, IPv4; no IPv6 socket tables): ok", self.out)

    def test_missing_tables_fail_closed(self):
        d = self.sim.procnet
        for missing, msg in ((("udp",), f"FATAL: metadata block: cannot read {d}/udp"),
                             (("raw",), f"FATAL: metadata block: cannot read {d}/raw"),
                             (("udp6",), f"FATAL: metadata block: some IPv6 socket tables are missing in {d} (udp6)"),
                             (("tcp6", "raw6"), f"some IPv6 socket tables are missing in {d} (tcp6 raw6)")):
            with self.subTest(missing=missing):
                self.tables(**{m: None for m in missing})
                p, st = self.run_sim(FULL)
                self.assertEqual(p.returncode, 1, self.out)
                self.assertIn(msg, self.out)
                self.assertEqual(st["calls"], [])

    def test_an_unexpected_format_fails_closed(self):
        d = self.sim.procnet
        for name, kw in (("short row", {"tcp": ["   9: 0100007F:0016 00000000:0000 0A"]}),
                         ("uid not a number", {"udp": [sock("udp", 61000, 5).replace(" 61000 ", " uid61k ")]}),
                         ("header without uid", {"header": "  sl  local_address rem_address   st inode"})):
            with self.subTest(name):
                self.tables(**kw)
                p, st = self.run_sim(FULL)
                self.assertEqual(p.returncode, 1, self.out)
                self.assertRegex(self.out, r"FATAL: metadata block: [0-9]+ line\(s\) of the socket tables in "
                                 + re.escape(str(d)) + " are not in the expected format")
                self.assertEqual(st["calls"], [])


# ----------------------------------------------------------------------------- service user
SU_START = "# >>> service user (CS-NET-001) >>>\n"
SU_END = "# <<< service user <<<\n"
SU_SECTION = ENTRY.split(SU_START, 1)[-1].split(SU_END, 1)[0]
# The pinned client image's account files (ghcr.io/mlcommons/endpoints 4235a9c, arm64, read
# from its layers; v1.0.2's 89904fb is built from the same scripts/Dockerfile.dev on
# python:3.12.11-slim with user appuser 1000:1000, and service_user verifies the live image's
# /etc/passwd and /etc/group at runtime during smoke): Debian trixie's plus appuser 1000:1000.
# Nothing uses 61000.
IMAGE_PASSWD = """root:x:0:0:root:/root:/bin/bash
daemon:x:1:1:daemon:/usr/sbin:/usr/sbin/nologin
bin:x:2:2:bin:/bin:/usr/sbin/nologin
sys:x:3:3:sys:/dev:/usr/sbin/nologin
sync:x:4:65534:sync:/bin:/bin/sync
games:x:5:60:games:/usr/games:/usr/sbin/nologin
man:x:6:12:man:/var/cache/man:/usr/sbin/nologin
lp:x:7:7:lp:/var/spool/lpd:/usr/sbin/nologin
mail:x:8:8:mail:/var/mail:/usr/sbin/nologin
news:x:9:9:news:/var/spool/news:/usr/sbin/nologin
uucp:x:10:10:uucp:/var/spool/uucp:/usr/sbin/nologin
proxy:x:13:13:proxy:/bin:/usr/sbin/nologin
www-data:x:33:33:www-data:/var/www:/usr/sbin/nologin
backup:x:34:34:backup:/var/backups:/usr/sbin/nologin
list:x:38:38:Mailing List Manager:/var/list:/usr/sbin/nologin
irc:x:39:39:ircd:/run/ircd:/usr/sbin/nologin
_apt:x:42:65534::/nonexistent:/usr/sbin/nologin
nobody:x:65534:65534:nobody:/nonexistent:/usr/sbin/nologin
appuser:x:1000:1000::<workdir>:/bin/bash
"""
IMAGE_GROUP = "".join(f"{g}:x:{n}:\n" for g, n in (
    ("root", 0), ("daemon", 1), ("bin", 2), ("sys", 3), ("adm", 4), ("tty", 5), ("disk", 6), ("lp", 7),
    ("mail", 8), ("news", 9), ("uucp", 10), ("man", 12), ("proxy", 13), ("kmem", 15), ("dialout", 20),
    ("fax", 21), ("voice", 22), ("cdrom", 24), ("floppy", 25), ("tape", 26), ("sudo", 27), ("audio", 29),
    ("dip", 30), ("www-data", 33), ("backup", 34), ("operator", 37), ("list", 38), ("irc", 39),
    ("src", 40), ("shadow", 42), ("utmp", 43), ("video", 44), ("sasl", 45), ("plugdev", 46),
    ("staff", 50), ("games", 60), ("users", 100), ("nogroup", 65534), ("appuser", 1000)))
SU_HARNESS = r"""set -euo pipefail
log() { printf '[swe-host] %s\n' "$*"; }
die() { log "FATAL: $*"; exit 1; }
WORK=$SIM_WORK
SVC_USER=${SIM_SVC_USER-kimiswe}
SVC_UID=${SIM_SVC_UID-61000}
SVC_GID=${SIM_SVC_GID-61000}
"""
# After the section: its account files are a copy, and groupadd/useradd/install are stubs that
# record their arguments (install cannot chown as a normal user: it sets the mode only).
SU_STUBS = r"""SVC_ETC=$SIM_ETC
record() { local a; printf '%s' "$1" >> "$SIM_CALLS"; shift; for a in "$@"; do printf ' [%s]' "$a" >> "$SIM_CALLS"; done; printf '\n' >> "$SIM_CALLS"; }
groupadd() {
  record groupadd "$@"
  [[ $# == 3 && $1 == -g ]] || { echo "groupadd: unexpected arguments" >&2; return 2; }
  if [[ -n ${SIM_GROUPADD_FAIL:-} ]]; then echo "groupadd: $SIM_GROUPADD_FAIL" >&2; return 4; fi
  printf '%s:x:%s:\n' "$3" "$2" >> "$SVC_ETC/group"
}
useradd() {
  record useradd "$@"
  local u= g= d= s= c=
  while (( $# > 1 )); do
    case $1 in
      -u) u=$2; shift 2 ;; -g) g=$2; shift 2 ;; -d) d=$2; shift 2 ;; -s) s=$2; shift 2 ;;
      -c) c=$2; shift 2 ;; -M|-N) shift ;;
      *) echo "useradd: unexpected option $1" >&2; return 2 ;;
    esac
  done
  if [[ -n ${SIM_USERADD_FAIL:-} ]]; then echo "useradd: $SIM_USERADD_FAIL" >&2; return 4; fi
  echo "useradd: warning: $1's uid $u outside of the UID_MIN 1000 and UID_MAX 60000 range." >&2
  printf '%s:x:%s:%s:%s:%s:%s\n' "$1" "$u" "${SIM_USERADD_GID:-$g}" "$c" "$d" "$s" >> "$SVC_ETC/passwd"
}
install() {
  record install "$@"
  local mode= dir=
  while (( $# )); do
    case $1 in -d) shift ;; -m) mode=$2; shift 2 ;; -o|-g) shift 2 ;; *) dir=$1; shift ;; esac
  done
  mkdir "$dir" && chmod "$mode" "$dir"
}
service_user
printf 'RESULT %s:%s:%s\n' "$RUN_UID" "$RUN_GID" "$RUN_HOME"
"""


def entry_function(name: str) -> str:
    """A top-level function of the entry script, from `name() {` to its closing brace."""
    m = re.search(rf"^{re.escape(name)}\(\) \{{.*?^\}}\n", ENTRY, re.M | re.S)
    if not m:
        raise AssertionError(f"no function {name} in the entry script")
    return m.group(0)


class ServiceUser(unittest.TestCase):
    """The marked `service user` section: a new user; any other use of its name or ids fails closed."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.etc = self.dir / "etc"
        self.work = self.dir / "work"
        self.etc.mkdir()
        self.work.mkdir()
        self.home = self.work / "home"
        self.calls_file = self.dir / "calls"

    def tearDown(self):
        self.tmp.cleanup()

    def run_su(self, passwd=IMAGE_PASSWD, group=IMAGE_GROUP, **env):
        (self.etc / "passwd").write_text(passwd)
        (self.etc / "group").write_text(group)
        self.calls_file.write_text("")
        script = self.dir / "su.sh"
        script.write_text(SU_HARNESS + SU_SECTION + SU_STUBS)
        e = {"PATH": os.environ["PATH"], "LC_ALL": "C", "SIM_ETC": str(self.etc),
             "SIM_WORK": str(self.work), "SIM_CALLS": str(self.calls_file), **env}
        p = subprocess.run(["bash", str(script)], env=e, capture_output=True, text=True, timeout=60,
                           check=False)
        self.out = p.stdout + p.stderr
        return p, self.calls_file.read_text().splitlines()

    def test_one_marked_section(self):
        self.assertEqual((ENTRY.count(SU_START), ENTRY.count(SU_END)), (1, 1))
        self.assertLess(ENTRY.index(SU_END), ENTRY.index("\nprepare() {"))

    def test_a_fresh_image_gets_the_user_with_exact_flags(self):
        p, calls = self.run_su()
        self.assertEqual(p.returncode, 0, self.out)
        home = str(self.home)
        self.assertEqual(calls, [
            "groupadd [-g] [61000] [kimiswe]",
            f"useradd [-u] [61000] [-g] [61000] [-M] [-N] [-d] [{home}] [-s] [/usr/sbin/nologin] "
            "[-c] [kimi-swe SWE service] [kimiswe]",
            f"install [-d] [-m] [0700] [-o] [61000] [-g] [61000] [{home}]"])
        self.assertEqual(self.home.stat().st_mode & 0o7777, 0o700)
        self.assertIn(f"RESULT 61000:61000:{home}", self.out)
        self.assertIn(f"service user kimiswe (61000:61000): created in this container; home {home} "
                      "(new, 0700); no supplementary groups", self.out)
        self.assertNotIn("UID_MAX", self.out)        # useradd's expected warning shows only on failure
        passwd = (self.etc / "passwd").read_text().splitlines()
        self.assertEqual(passwd, IMAGE_PASSWD.splitlines()
                         + [f"kimiswe:x:61000:61000:kimi-swe SWE service:{home}:/usr/sbin/nologin"])
        self.assertEqual((self.etc / "group").read_text().splitlines(),
                         IMAGE_GROUP.splitlines() + ["kimiswe:x:61000:"])

    def test_the_band_edges_are_accepted(self):
        for uid in ("60578", "61183"):
            with self.subTest(uid=uid):
                if self.home.exists():
                    self.home.rmdir()
                p, _ = self.run_su(SIM_SVC_UID=uid, SIM_SVC_GID=uid)
                self.assertEqual(p.returncode, 0, self.out)
                self.assertIn(f"RESULT {uid}:{uid}:", self.out)

    def test_already_exactly_in_the_image(self):
        p, calls = self.run_su(IMAGE_PASSWD + "kimiswe:x:61000:61000::/nonexistent:/usr/sbin/nologin\n",
                               IMAGE_GROUP + "kimiswe:x:61000:\n")
        self.assertEqual(p.returncode, 0, self.out)
        self.assertEqual(calls, [f"install [-d] [-m] [0700] [-o] [61000] [-g] [61000] [{self.home}]"])
        self.assertIn("service user kimiswe (61000:61000): already in the image with exactly these ids",
                      self.out)

    def test_only_the_group_exists(self):
        p, calls = self.run_su(group=IMAGE_GROUP + "kimiswe:x:61000:\n")
        self.assertEqual(p.returncode, 0, self.out)
        self.assertEqual([c.split()[0] for c in calls], ["useradd", "install"])

    def test_any_other_use_of_the_name_uid_or_gid_fails_closed(self):
        passwd_msg = (f"FATAL: service user: {self.etc}/passwd already uses the name kimiswe, uid 61000 "
                      "or gid 61000 otherwise (name:uid:gid): ")
        group_msg = (f"FATAL: service user: {self.etc}/group already uses the name kimiswe or gid 61000 "
                     "otherwise (name:gid): ")
        cases = {
            "uid taken": (IMAGE_PASSWD + "other:x:61000:61001::/:/bin/sh\n", IMAGE_GROUP,
                          passwd_msg + "other:61000:61001"),
            "another user's primary gid": (IMAGE_PASSWD + "svc:x:61500:61000::/:/bin/sh\n", IMAGE_GROUP,
                                           passwd_msg + "svc:61500:61000"),
            "name with another uid": (IMAGE_PASSWD + "kimiswe:x:1000:1000::<workdir>:/bin/sh\n", IMAGE_GROUP,
                                      passwd_msg + "kimiswe:1000:1000"),
            "name with another gid": (IMAGE_PASSWD + "kimiswe:x:61000:100::/:/bin/sh\n", IMAGE_GROUP,
                                      passwd_msg + "kimiswe:61000:100"),
            "a second user on the uid": (IMAGE_PASSWD + "kimiswe:x:61000:61000::/:/bin/sh\n"
                                         "alias:x:61000:61000::/:/bin/sh\n", IMAGE_GROUP + "kimiswe:x:61000:\n",
                                         passwd_msg + "kimiswe:61000:61000 alias:61000:61000"),
            "gid taken by a group": (IMAGE_PASSWD, IMAGE_GROUP + "other:x:61000:\n", group_msg + "other:61000"),
            "group name with another gid": (IMAGE_PASSWD, IMAGE_GROUP + "kimiswe:x:1000:\n",
                                            group_msg + "kimiswe:1000"),
            "a member of another group": (IMAGE_PASSWD, IMAGE_GROUP.replace("sudo:x:27:\n", "sudo:x:27:kimiswe\n"),
                                          group_msg + "sudo:27"),
            "one of several members": (IMAGE_PASSWD, IMAGE_GROUP.replace("audio:x:29:\n", "audio:x:29:pulse,kimiswe\n"),
                                       group_msg + "audio:29"),
        }
        for name, (passwd, group, msg) in cases.items():
            with self.subTest(name):
                p, calls = self.run_su(passwd, group)
                self.assertEqual(p.returncode, 1, self.out)
                self.assertIn(msg, self.out)
                self.assertEqual(calls, [])              # nothing created, no home
                self.assertFalse(self.home.exists())

    def test_similar_names_and_its_own_group_are_not_conflicts(self):
        group = (IMAGE_GROUP.replace("users:x:100:\n", "users:x:100:kimiswe2,akimiswe,kimi\n")
                 + "kimiswe:x:61000:kimiswe\n")
        p, calls = self.run_su(group=group)
        self.assertEqual(p.returncode, 0, self.out)
        self.assertEqual([c.split()[0] for c in calls], ["useradd", "install"])

    def test_ids_outside_the_unused_band_are_refused(self):
        for uid, gid in (("0", "61000"), ("999", "61000"), ("1000", "1000"), ("1001", "61000"),
                         ("60000", "61000"), ("60577", "61000"), ("61184", "61000"), ("65532", "61000"),
                         ("65534", "65534"), ("61000", "1000"), ("61000", "65534"), ("061000", "61000"),
                         ("61000x", "61000"), ("", "61000")):
            with self.subTest(uid=uid, gid=gid):
                p, calls = self.run_su(SIM_SVC_UID=uid, SIM_SVC_GID=gid)
                self.assertEqual(p.returncode, 1, self.out)
                self.assertIn(f"FATAL: service user: uid:gid {uid}:{gid} is not in 60578-61183 (unused in "
                              "systemd's uid map; never 1000, 1001, 65532 or 65534)", self.out)
                self.assertEqual(calls, [])

    def test_the_home_must_be_fresh(self):
        for kind in ("directory", "dangling symlink"):
            with self.subTest(kind):
                if kind == "directory":
                    self.home.mkdir()
                else:
                    self.home.rmdir()
                    self.home.symlink_to(self.dir / "elsewhere")
                p, calls = self.run_su()
                self.assertEqual(p.returncode, 1, self.out)
                self.assertIn(f"FATAL: service user: home {self.home} exists already; it must be fresh",
                              self.out)
                self.assertNotIn("install", [c.split()[0] for c in calls])

    def test_failures_of_the_account_tools_are_fatal_and_shown(self):
        p, calls = self.run_su(SIM_GROUPADD_FAIL="GID '61000' already exists")
        self.assertEqual(p.returncode, 1, self.out)
        self.assertIn("FATAL: service user: groupadd failed: groupadd: GID '61000' already exists", self.out)
        self.assertEqual([c.split()[0] for c in calls], ["groupadd"])
        p, calls = self.run_su(SIM_USERADD_FAIL="UID 61000 is not unique")
        self.assertEqual(p.returncode, 1, self.out)
        self.assertIn("FATAL: service user: useradd failed: useradd: UID 61000 is not unique", self.out)
        self.assertFalse(self.home.exists())

    def test_the_result_is_verified(self):
        p, calls = self.run_su(SIM_USERADD_GID="100")
        self.assertEqual(p.returncode, 1, self.out)
        self.assertIn(f"FATAL: service user: {self.etc}/passwd and {self.etc}/group do not show exactly "
                      "kimiswe (61000:61000)", self.out)
        self.assertFalse(self.home.exists())


# ----------------------------------------------------------------------------- ownership
PREP_HARNESS = r"""set -euo pipefail
log() { printf '[swe-host] %s\n' "$*"; }
die() { log "FATAL: $*"; exit 1; }
STATE=$SIM_DIR/state
PRIV=$SIM_DIR/priv
WORK=$SIM_DIR/work
SECRET_DIR=$SIM_DIR/secret
REGISTRY_PATH=registry.gitlab.com/example/swebench-arm64
RUN_ID=run-1
HANDOFF=${SIM_HANDOFF:-0}
IMAGE_CACHE=${SIM_CACHE:-run}
SVC_USER=kimiswe
RUN_AS=${SIM_RUN_AS:-kimiswe}
record() { local a; printf '%s' "$1" >> "$SIM_CALLS"; shift; for a in "$@"; do printf ' %s' "$a" >> "$SIM_CALLS"; done; printf '\n' >> "$SIM_CALLS"; }
service_user() { RUN_UID=${SIM_UID:-61000} RUN_GID=${SIM_GID:-61000} RUN_HOME=$WORK/home; }
id() { if [[ $1 == -u ]]; then echo "${SIM_ID_U:-$RUN_UID}"; else echo "$RUN_GID"; fi; }
hostname() { echo gke-node-a; }
chown() { record chown "$@"; }
install() {
  record install "$@"
  local d=0 mode= args=()
  while (( $# )); do
    case $1 in -d) d=1; shift ;; -m) mode=$2; shift 2 ;; -o|-g) shift 2 ;; *) args+=("$1"); shift ;; esac
  done
  if (( d )); then mkdir -p "${args[@]}"; chmod "$mode" "${args[@]}"
  else cp "${args[0]}" "${args[1]}"; chmod "$mode" "${args[1]}"; fi
}
"""
CREDENTIALS = "machine registry.gitlab.com login bot password EXAMPLE-TOKEN-0123\nmachine gitlab.com login bot password EXAMPLE-TOKEN-0123\n"


class PrepareOwnership(unittest.TestCase):
    """prepare() and load_credentials(): every path the service user writes is its own; root's stay root's."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        for d in ("priv", "work", "secret"):
            (self.dir / d).mkdir()
        (self.dir / "secret" / ".credentials").write_text(CREDENTIALS)
        self.state = self.dir / "state"
        self.calls_file = self.dir / "calls"

    def tearDown(self):
        self.tmp.cleanup()

    def run_prepare(self, service=True, **env):
        self.calls_file.write_text("")
        script = self.dir / "prep.sh"
        script.write_text(PREP_HARNESS + entry_function("prepare") + entry_function("load_credentials")
                          + "prepare\n" + ("load_credentials\n" if service else "")
                          + 'printf "AS_USER %s\\n" "${AS_USER[*]}"\n')
        e = {"PATH": os.environ["PATH"], "LC_ALL": "C", "SIM_DIR": str(self.dir),
             "SIM_CALLS": str(self.calls_file), **env}
        p = subprocess.run(["bash", str(script)], env=e, capture_output=True, text=True, timeout=60,
                           check=False)
        self.out = p.stdout + p.stderr
        self.assertNotIn("EXAMPLE-TOKEN", self.out)
        return p

    def owners(self):
        """path -> (owner uid or None for root's default, group, mode) from the recorded installs."""
        got = {}
        for line in self.calls_file.read_text().splitlines():
            words = line.split()
            if words[0] != "install":
                continue
            opts, paths, i = {}, [], 1
            while i < len(words):
                if words[i] in ("-m", "-o", "-g"):
                    opts[words[i]] = words[i + 1]
                    i += 2
                elif words[i] == "-d":
                    opts["-d"] = True
                    i += 1
                else:
                    paths.append(words[i])
                    i += 1
            targets = paths if opts.get("-d") else paths[-1:]
            for t in targets:
                got[t] = (opts.get("-o"), opts.get("-g"), opts.get("-m"))
        return got

    def test_pilot_run(self):
        p = self.run_prepare()
        self.assertEqual(p.returncode, 0, self.out)
        s, run, priv = str(self.state), f"{self.state}/runs/run-1", f"{self.dir}/priv"
        self.assertEqual(self.owners(), {
            s: (None, None, "0755"), f"{s}/runs": (None, None, "0755"),
            run: ("61000", "61000", "0700"),
            f"{run}/swe-connection.json": ("61000", "61000", "0600"),
            f"{priv}/root": (None, None, "0700"),
            f"{priv}/enroot": ("61000", "61000", "0700"),
            f"{priv}/enroot/.credentials": ("61000", "61000", "0600")})
        self.assertIn("run run-1 on gke-node-a: service user kimiswe (61000:61000), image cache: run", self.out)
        self.assertIn(f"AS_USER env HOME={self.dir}/work/home USER=kimiswe LOGNAME=kimiswe setpriv "
                      "--reuid=61000 --regid=61000 --clear-groups --inh-caps=-all --", self.out)
        self.assertIn("registry: <email>/example/swebench-arm64 (token not shown)", self.out)

    def test_production_handoff(self):
        p = self.run_prepare(SIM_HANDOFF="1")
        self.assertEqual(p.returncode, 0, self.out)
        s = str(self.state)
        got = self.owners()
        self.assertEqual(got[f"{s}/handoff"], (None, None, "0755"))
        self.assertEqual(got[f"{s}/handoff/run-1"], (None, None, "0711"))
        self.assertEqual(got[f"{s}/handoff/run-1/swe-connection.json"], ("61000", "61000", "0600"))
        self.assertNotIn(f"{s}/runs/run-1/swe-connection.json", got)

    def test_persistent_cache_is_the_service_users(self):
        p = self.run_prepare(SIM_CACHE="persistent")
        self.assertEqual(p.returncode, 0, self.out)
        s, run = str(self.state), f"{self.state}/runs/run-1"
        got = self.owners()
        self.assertEqual(got[f"{s}/enroot-cache"], ("61000", "61000", "0700"))
        self.assertEqual(got[f"{run}/swe_service"], ("61000", "61000", "0700"))
        self.assertIn(f"chown -h 61000:61000 {run}/swe_service/enroot-cache",
                      self.calls_file.read_text().splitlines())
        # No other owner anywhere: every chowned path is root's or the service user's.
        self.assertEqual({o[:2] for o in got.values()} - {(None, None)}, {("61000", "61000")})

    def test_a_cache_written_by_another_uid_is_refused_not_chowned(self):
        cache = self.state / "enroot-cache"
        (cache / "image").mkdir(parents=True)                     # owned by the test's own uid
        p = self.run_prepare(SIM_CACHE="persistent")
        self.assertEqual(p.returncode, 1, self.out)
        self.assertIn(f"FATAL: the persistent image cache {cache} has 2 entries that kimiswe (61000:61000) "
                      "does not own, for example from runs as uid 1000. Delete it on the node, or run "
                      "without --persist-cache", self.out)
        self.assertNotIn("chown", self.calls_file.read_text())
        self.assertNotIn(str(cache), self.owners())
        # The same cache, owned by the service user, is reused (a new run: prepare refuses old ones).
        shutil.rmtree(self.state / "runs")
        p = self.run_prepare(SIM_CACHE="persistent", SIM_UID=str(os.getuid()), SIM_GID=str(os.getgid()))
        self.assertEqual(p.returncode, 0, self.out)
        self.assertTrue((cache / "image").is_dir())

    def test_a_cache_that_is_not_a_directory_is_refused(self):
        self.state.mkdir()
        (self.state / "enroot-cache").symlink_to(self.dir / "work")
        p = self.run_prepare(SIM_CACHE="persistent", SIM_UID=str(os.getuid()), SIM_GID=str(os.getgid()))
        self.assertEqual(p.returncode, 1, self.out)
        self.assertIn(f"FATAL: {self.state}/enroot-cache is not a directory", self.out)

    def test_id_must_agree(self):
        p = self.run_prepare(SIM_ID_U="1000")
        self.assertEqual(p.returncode, 1, self.out)
        self.assertIn("FATAL: id kimiswe does not report uid:gid 61000:61000", self.out)

    def test_the_root_fallback_is_unchanged(self):
        p = self.run_prepare(SIM_RUN_AS="root")
        self.assertEqual(p.returncode, 0, self.out)
        self.assertIn("WARNING: SWE_RUN_AS=root. Sandboxes then run as real root (remapped). Fallback only.",
                      self.out)
        self.assertEqual(self.owners()[f"{self.state}/runs/run-1"], ("0", "0", "0700"))
        self.assertIn("AS_USER env HOME=/root USER=root LOGNAME=root setpriv --reuid=0 --regid=0 "
                      "--clear-groups", self.out)


OWNER_SNIPPET = ENTRY[ENTRY.index("    owners=$(find"):]
OWNER_SNIPPET = OWNER_SNIPPET[:OWNER_SNIPPET.index("\n    fi\n") + len("\n    fi\n")]


class PostcheckOwners(unittest.TestCase):
    """The teardown counts the run directory's entries that are not the service user's (warning only)."""

    def run_snippet(self, run, uid, gid):
        script = ("set -euo pipefail\nlog() { printf '[swe-host] %s\\n' \"$*\"; }\n"
                  f"RUN={run}\nRUN_UID={uid}\nRUN_GID={gid}\nRUN_AS=kimiswe\nowners=\n" + OWNER_SNIPPET)
        p = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=60, check=False)
        return p.returncode, p.stdout + p.stderr

    def test_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "client").mkdir()
            (Path(tmp) / "client" / "out.json").write_text("{}")
            rc, out = self.run_snippet(tmp, os.getuid(), os.getgid())
            self.assertEqual(rc, 0, out)
            self.assertIn(f"POSTCHECK all 3 entries in the run directory belong to kimiswe "
                          f"({os.getuid()}:{os.getgid()}): ok", out)
            rc, out = self.run_snippet(tmp, 61000, 61000)
            self.assertEqual(rc, 0, out)
            self.assertIn("POSTCHECK WARNING: 3 of 3 entries in the run directory do not belong to kimiswe "
                          "(61000:61000)", out)
            rc, out = self.run_snippet(f"{tmp}/missing", 61000, 61000)
            self.assertEqual(rc, 0, out)
            self.assertIn("POSTCHECK WARNING: cannot list who owns the entries of the run directory", out)

    def test_it_runs_before_root_writes_its_reports(self):
        body = entry_function("postcheck")
        self.assertLess(body.index("owners=$(find"), body.index("probe leakscan"))
        self.assertLess(body.index("owners=$(find"), body.index("probe postcheck-report"))
        self.assertGreater(body.index("owners=$(find"), body.index("pkill -KILL -U"))


# ----------------------------------------------------------------------------- v2 wiring
TOP_VALIDATION = ENTRY[ENTRY.index("MODE=${SWE_POD_MODE:-}\n"):ENTRY.index("[[ $(id -u) == 0 ]]")]


class ServiceUserWiring(unittest.TestCase):
    node = "gke-node-a"

    @staticmethod
    def env(pod):
        return {e["name"]: e["value"] for e in pod["spec"]["containers"][0]["env"]}

    def validate(self, **env):
        script = "log() { printf '%s\\n' \"$*\"; }\ndie() { log \"FATAL: $*\"; exit 1; }\n" + TOP_VALIDATION
        e = {"PATH": os.environ["PATH"], "SWE_POD_MODE": "service", "SWE_RUN_ID": "r1",
             "SWE_SERVICE_USER": "kimiswe", "SWE_SERVICE_UID": "61000", "SWE_SERVICE_GID": "61000", **env}
        e = {k: v for k, v in e.items() if v is not None}
        p = subprocess.run(["bash", "-c", script], env=e, capture_output=True, text=True, timeout=30,
                           check=False)
        return p.returncode, p.stdout + p.stderr

    def test_one_definition_in_the_driver(self):
        self.assertEqual((sp.SERVICE_USER, sp.SERVICE_UID, sp.SERVICE_GID), ("kimiswe", 61000, 61000))
        for uid in (sp.SERVICE_UID, sp.SERVICE_GID):
            self.assertNotIn(uid, (0, 1000, 1001, 65532, 65534))
            p = subprocess.run(["bash", "-c", entry_function("svc_id_ok") + f"svc_id_ok {uid}"],
                               capture_output=True, text=True, timeout=30, check=False)
            self.assertEqual(p.returncode, 0, p.stderr)
        self.assertNotIn("APP_UID", (SWE_DIR / "pilot" / "swe_pilot.py").read_text())

    def test_every_host_pod_passes_it_and_the_entry_script_has_no_default(self):
        for pod in (sp.host_pod(sp.POD["smoke"], "smoke", self.node, "smoke-1", mode="smoke"),
                    sp.host_pod(sp.POD["host"], "host", self.node, "pilot-1", mode="service"),
                    sp.host_pod(sp.POD["host"], "host", self.node, "run-1", mode="service",
                                handoff=True, reports_dir=sp.REPORTS_DIR)):
            env = self.env(pod)
            self.assertEqual((env["SWE_SERVICE_USER"], env["SWE_SERVICE_UID"], env["SWE_SERVICE_GID"]),
                             ("kimiswe", "61000", "61000"))
            self.assertNotIn("SWE_RUN_AS", env)
        for line in ("SVC_USER=${SWE_SERVICE_USER:-}\n", "SVC_UID=${SWE_SERVICE_UID:-}\n",
                     "SVC_GID=${SWE_SERVICE_GID:-}\n", "RUN_AS=${SWE_RUN_AS:-$SVC_USER}"):
            self.assertIn(line, ENTRY)
        self.assertNotIn("61000", ENTRY.replace("kimiswe, 61000:61000", ""))

    def test_the_entry_script_validates_them(self):
        self.assertEqual(self.validate(), (0, ""))
        self.assertEqual(self.validate(SWE_RUN_AS="root")[0], 0)
        msg = ("FATAL: SWE_SERVICE_USER, SWE_SERVICE_UID and SWE_SERVICE_GID must be a user name and two "
               "numbers (swe_pilot.py sets them)")
        for bad in ({"SWE_SERVICE_UID": None}, {"SWE_SERVICE_USER": None}, {"SWE_SERVICE_GID": "0"},
                    {"SWE_SERVICE_UID": "061000"}, {"SWE_SERVICE_USER": "Kimi Swe"}):
            with self.subTest(bad=bad):
                rc, out = self.validate(**bad)
                self.assertEqual(rc, 1, out)
                self.assertIn(msg, out)
        rc, out = self.validate(SWE_RUN_AS="appuser")
        self.assertEqual(rc, 1, out)
        self.assertIn("FATAL: SWE_RUN_AS must be kimiswe (the default) or root", out)

    def test_client_and_collect_run_as_the_service_user(self):
        a = SimpleNamespace(app_uid=None, app_gid=None)
        with unittest.mock.patch.object(sp, "load_record", lambda run_id: {}):
            self.assertEqual(sp.app_ids(a, "r1"), (61000, 61000))
        with unittest.mock.patch.object(sp, "load_record", lambda run_id: {"uid": 61001, "gid": 61002}):
            self.assertEqual(sp.app_ids(a, "r1"), (61001, 61002))       # the host log's ids win
            self.assertEqual(sp.app_ids(SimpleNamespace(app_uid=60600, app_gid=60601), "r1"), (60600, 60601))
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                sp.app_ids(SimpleNamespace(app_uid=0, app_gid=None), "r1")
        client = sp.client_pod(self.node, "r1", sp.SERVICE_UID, sp.SERVICE_GID)
        sc = client["spec"]["containers"][0]["securityContext"]
        self.assertEqual((sc["runAsUser"], sc["runAsGroup"], sc["runAsNonRoot"]), (61000, 61000, True))
        env = self.env(client)
        self.assertEqual((env["USER"], env["LOGNAME"]), ("kimiswe", "kimiswe"))
        self.assertIn('mkdir -p -m 0700 "$KIMI_SWE_RUN/client/home"\n', sp.CLIENT_SCRIPT)
        self.assertIn('export HOME="$KIMI_SWE_RUN/client/home"\n', sp.CLIENT_SCRIPT)
        collect = sp.collect_pod(self.node, "r1", sp.SERVICE_UID, sp.SERVICE_GID)
        self.assertEqual(collect["spec"]["containers"][0]["securityContext"]["runAsUser"], 61000)
        ap = sp.parser()
        for cmd, extra in (("client", []), ("collect", ["--dest", "d"])):
            ns = ap.parse_args([cmd, "--node", "n", "--run-id", "r1", "--app-uid", "61000", "--app-gid", "61000",
                                *extra])
            self.assertEqual((ns.app_uid, ns.app_gid), (61000, 61000))

    def test_the_drivers_log_regex_reads_both_service_user_lines(self):
        regex = r"service user \S+ \((\d+):(\d+)\)"
        self.assertIn(regex, (SWE_DIR / "pilot" / "swe_pilot.py").read_text())
        for line in ("[swe-host 12:00:00Z] service user kimiswe (61000:61000): created in this container; "
                     "home /opt/kimi-swe/home (new, 0700); no supplementary groups",
                     "[swe-host 12:00:01Z] run r1 on gke-node-a: service user kimiswe (61000:61000), "
                     "image cache: run"):
            self.assertEqual(re.search(regex, line).groups(), ("61000", "61000"))

    def test_setpriv_clears_supplementary_groups(self):
        self.assertIn("--clear-groups --inh-caps=-all --)", ENTRY)
        self.assertNotIn("--init-groups", ENTRY)

    def test_the_socket_precheck_comes_first(self):
        body = entry_function("metadata_block_apply")
        self.assertLess(body.index('[[ $RUN_UID != 0 ]] || die "the metadata block needs'),
                        body.index("\n  mdb_socket_precheck\n"))
        self.assertLess(body.index("\n  mdb_socket_precheck\n"), body.index("\n  mdb_pick_backend\n"))
        self.assertIn("MDB_PROC_NET=/proc/net ", SECTION)

    def test_every_constant_the_driver_uses_is_defined(self):
        # A removed constant (APP_UID) still named somewhere would only fail at run time.
        import ast
        import builtins
        tree = ast.parse((SWE_DIR / "pilot" / "swe_pilot.py").read_text())
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
                 and re.fullmatch(r"[A-Z][A-Z0-9_]+", n.id)}
        self.assertEqual({n for n in names if not hasattr(sp, n) and not hasattr(builtins, n)}, set())


import unittest.mock  # noqa: E402  (used by ProbeMetadataCheck)

if __name__ == "__main__":
    unittest.main()
