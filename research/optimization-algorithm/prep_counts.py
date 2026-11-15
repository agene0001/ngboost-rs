#!/usr/bin/env python3
"""Prepare the real count datasets of Cycle 26 (O17) as numeric CSVs.

    python3 -I prep_counts.py RAW_FOOTBALL_DIR RAW_RDATASETS_DIR OUT_DIR

Inputs (downloaded 2026-11-14, not kept in the repository):
  * football-data.co.uk result files `<league>_<season>.csv` for leagues
    E0 E1 D1 I1 SP1 F1, seasons 1920 … 2425;
  * Rdatasets CSVs `bioChemists.csv` (pscl), `DoctorVisits.csv`,
    `NMES1988.csv` (AER).

Outputs, each with a header and the target in column `y`:
  * `fbo_home.csv`, `fbo_away.csv` — one row per match, target = home /
    away goals, features = pre-match market odds (average, not closing)
    plus team form; first two columns `season` (0–5, a split key, not a
    feature) and `league` (0–5);
  * `fbf_home.csv`, `fbf_away.csv` — the same matches without any odds:
    Elo ratings and team form only;
  * `biochem.csv`, `doctor.csv`, `nmes.csv` — categorical columns coded
    as integers (binary) or ordered levels.

Every football feature uses matches played strictly before the row's
match. Rows are in date order. Stdlib only.
"""
import csv
import datetime
import os
import sys
from collections import defaultdict, deque

LEAGUES = ["E0", "E1", "D1", "I1", "SP1", "F1"]
SEASONS = ["1920", "2021", "2122", "2223", "2324", "2425"]


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def parse_date(s):
    for fmt in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            return datetime.datetime.strptime(s.strip(), fmt).date()
        except ValueError:
            pass
    return None


def football(raw_dir):
    matches = []
    skipped = 0
    for li, lg in enumerate(LEAGUES):
        for si, se in enumerate(SEASONS):
            path = os.path.join(raw_dir, f"{lg}_{se}.csv")
            with open(path, encoding="latin-1", newline="") as fh:
                for r in csv.DictReader(fh):
                    d = parse_date(r.get("Date") or "")
                    hg, ag = fnum(r.get("FTHG")), fnum(r.get("FTAG"))
                    if d is None or hg is None or ag is None or not r.get("HomeTeam"):
                        skipped += 1
                        continue

                    def odds(*names):
                        for n in names:
                            v = fnum(r.get(n))
                            if v is not None and v > 1.0:
                                return v
                        return None

                    matches.append(
                        dict(
                            date=d, league=li, season=si,
                            home=r["HomeTeam"].strip(), away=r["AwayTeam"].strip(),
                            hg=hg, ag=ag,
                            hst=fnum(r.get("HST")), ast=fnum(r.get("AST")),
                            oh=odds("AvgH", "B365H"), od=odds("AvgD", "B365D"),
                            oa=odds("AvgA", "B365A"),
                            oo=odds("Avg>2.5", "B365>2.5"), ou=odds("Avg<2.5", "B365<2.5"),
                            ah=fnum(r.get("AHh")),
                        )
                    )
    matches.sort(key=lambda m: (m["date"], m["league"], m["home"]))

    # team state, keyed by (league, team); carried across seasons
    hist = defaultdict(lambda: deque(maxlen=20))  # (gf, ga, sotf, sota)
    elo = defaultdict(lambda: 1500.0)
    lg_goals = defaultdict(lambda: [27.0, 20.0])  # running league mean goals per team-match (prior 1.35 × 20)

    def form(key, k, lg):
        h = list(hist[key])[-k:]
        mean = lg_goals[lg][0] / lg_goals[lg][1]
        if not h:
            return [mean, mean, 4.0, 4.0]
        n = len(h)
        return [sum(x[j] for x in h) / n for j in range(4)]

    rows_o_h, rows_o_a, rows_f_h, rows_f_a = [], [], [], []
    no_odds = 0
    for m in matches:
        lg = m["league"]
        hk, ak = (lg, m["home"]), (lg, m["away"])
        f6h, f6a = form(hk, 6, lg), form(ak, 6, lg)
        f20h, f20a = form(hk, 20, lg), form(ak, 20, lg)
        nh, na = float(len(hist[hk])), float(len(hist[ak]))
        eh, ea = elo[hk], elo[ak]
        base = [float(m["season"]), float(lg)]
        formfeat = f6h + f6a + [nh, na]
        f_only = base + [eh, ea, eh - ea] + f6h + f6a + f20h[:2] + f20a[:2] + [nh, na]
        rows_f_h.append(f_only + [m["hg"]])
        rows_f_a.append(f_only + [m["ag"]])
        if None in (m["oh"], m["od"], m["oa"], m["oo"], m["ou"]):
            no_odds += 1
        else:
            inv = [1 / m["oh"], 1 / m["od"], 1 / m["oa"]]
            s = sum(inv)
            po = (1 / m["oo"]) / (1 / m["oo"] + 1 / m["ou"])
            ah = m["ah"] if m["ah"] is not None else 0.0
            o = base + [inv[0] / s, inv[1] / s, inv[2] / s, po, ah] + formfeat
            rows_o_h.append(o + [m["hg"]])
            rows_o_a.append(o + [m["ag"]])
        # update state after the match
        hst = m["hst"] if m["hst"] is not None else 4.0
        ast = m["ast"] if m["ast"] is not None else 4.0
        hist[hk].append((m["hg"], m["ag"], hst, ast))
        hist[ak].append((m["ag"], m["hg"], ast, hst))
        lg_goals[lg][0] += m["hg"] + m["ag"]
        lg_goals[lg][1] += 2.0
        exp_h = 1.0 / (1.0 + 10 ** (-(eh + 60.0 - ea) / 400.0))
        res = 1.0 if m["hg"] > m["ag"] else 0.5 if m["hg"] == m["ag"] else 0.0
        elo[hk] = eh + 20.0 * (res - exp_h)
        elo[ak] = ea - 20.0 * (res - exp_h)

    form_names = ["h_gf6", "h_ga6", "h_sotf6", "h_sota6", "a_gf6", "a_ga6", "a_sotf6", "a_sota6"]
    hdr_o = ["season", "league", "p_home", "p_draw", "p_away", "p_over25", "ah_line"] + form_names + ["h_n", "a_n", "y"]
    hdr_f = (["season", "league", "elo_h", "elo_a", "elo_diff"] + form_names
             + ["h_gf20", "h_ga20", "a_gf20", "a_ga20", "h_n", "a_n", "y"])
    print(f"football: {len(matches)} matches, {skipped} unusable rows skipped, {no_odds} without odds")
    return {
        "fbo_home": (hdr_o, rows_o_h), "fbo_away": (hdr_o, rows_o_a),
        "fbf_home": (hdr_f, rows_f_h), "fbf_away": (hdr_f, rows_f_a),
    }


def rdataset(path, target, coding, drop=()):
    """`coding`: column -> {level: number}; other columns must be numeric."""
    rows, hdr = [], None
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            cols = [c for c in r if c not in ("rownames", target) and c not in drop]
            if hdr is None:
                hdr = cols + ["y"]
            row = []
            for c in cols:
                v = r[c]
                row.append(float(coding[c][v]) if c in coding else float(v))
            row.append(float(r[target]))
            rows.append(row)
    return hdr, rows


def main():
    raw_fb, raw_rd, out = sys.argv[1:4]
    os.makedirs(out, exist_ok=True)
    sets = football(raw_fb)
    yn = {"yes": 1, "no": 0}
    sets["biochem"] = rdataset(
        os.path.join(raw_rd, "bioChemists.csv"), "art",
        {"fem": {"Men": 0, "Women": 1}, "mar": {"Single": 0, "Married": 1}},
    )
    sets["doctor"] = rdataset(
        os.path.join(raw_rd, "DoctorVisits.csv"), "visits",
        {"gender": {"male": 0, "female": 1}, "private": yn, "freepoor": yn,
         "freerepat": yn, "nchronic": yn, "lchronic": yn},
    )
    # NMES1988: `visits` (physician office visits) is the target; the other
    # visit counts of the same year are outcomes too and are dropped
    sets["nmes"] = rdataset(
        os.path.join(raw_rd, "NMES1988.csv"), "visits",
        {"health": {"poor": 0, "average": 1, "excellent": 2},
         "adl": {"normal": 0, "limited": 1},
         "region": {"other": 0, "northeast": 1, "midwest": 2, "west": 3},
         "afam": yn, "gender": {"male": 0, "female": 1}, "married": yn,
         "employed": yn, "insurance": yn, "medicaid": yn},
        drop=("nvisits", "ovisits", "novisits", "emergency", "hospital"),
    )
    for name, (hdr, rows) in sets.items():
        with open(os.path.join(out, name + ".csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(hdr)
            for r in rows:
                w.writerow([repr(v) if v != int(v) else str(int(v)) for v in r])
        ys = [r[-1] for r in rows]
        mean = sum(ys) / len(ys)
        var = sum((v - mean) ** 2 for v in ys) / (len(ys) - 1)
        print(f"{name:<9} rows {len(rows):>6} features {len(hdr) - 1:>3} mean {mean:.3f} "
              f"var/mean {var / mean:.2f} zeros {sum(v == 0 for v in ys) / len(ys):.2f} max {max(ys):.0f}")


if __name__ == "__main__":
    main()
