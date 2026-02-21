"""
╔══════════════════════════════════════════════════════════════════════════╗
║           URL RISK ANALYZER — MODEL VERIFICATION v2                     ║
║                                                                          ║
║  Fix from v1: Uses core_engine's REAL feature extraction pipeline        ║
║  instead of a re-implemented approximation. This means the verify        ║
║  script sees the EXACT same numbers the model was trained on.            ║
╚══════════════════════════════════════════════════════════════════════════╝

Run: python verify_model.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

GREEN  = "\033[92m"; RED = "\033[91m"; YELLOW = "\033[93m"
BOLD   = "\033[1m";  RESET = "\033[0m"
def g(t): return f"{GREEN}{t}{RESET}"
def r(t): return f"{RED}{t}{RESET}"
def y(t): return f"{YELLOW}{t}{RESET}"
def b(t): return f"{BOLD}{t}{RESET}"

print("\n" + "="*70)
print(b("  URL RISK ANALYZER — MODEL VERIFICATION v2"))
print("="*70)

# ─── STEP 1: Model files ──────────────────────────────────────────────────────
print(f"\n{b('STEP 1 — Checking model files...')}")
MODEL_DIR = ROOT / "models"
files = {
    "risk_model.pkl":      "Risk Level Classifier  (Low / Medium / High)",
    "risk_type_model.pkl": "Risk Type Classifier   (Phishing/Malware/Scam...)",
    "anomaly_model.pkl":   "Anomaly Detector       (IsolationForest)",
}
all_present = True
for fname, desc in files.items():
    p = MODEL_DIR / fname
    if p.exists():
        print(f"  {g('✓')} {fname:<25} {desc}  [{p.stat().st_size//1024} KB]")
    else:
        print(f"  {r('✗')} {fname:<25} {r('MISSING')} — run train_model.py first")
        all_present = False
print(f"\n  {g('All model files present!') if all_present else r('Some models missing!')}")

# ─── STEP 2: Load + inspect ───────────────────────────────────────────────────
print(f"\n{b('STEP 2 — Loading models...')}")
import joblib, numpy as np

risk_model = risk_type_model = anomaly_model = None
try:
    if (MODEL_DIR/"risk_model.pkl").exists():
        risk_model = joblib.load(MODEL_DIR/"risk_model.pkl")
        print(f"  {g('✓')} Risk Model:  trees={risk_model.n_estimators}  "
              f"depth={risk_model.max_depth}  classes={list(risk_model.classes_)}")
    if (MODEL_DIR/"risk_type_model.pkl").exists():
        risk_type_model = joblib.load(MODEL_DIR/"risk_type_model.pkl")
        print(f"  {g('✓')} Type Model:  trees={risk_type_model.n_estimators}  "
              f"classes={list(risk_type_model.classes_)}")
    if (MODEL_DIR/"anomaly_model.pkl").exists():
        anomaly_model = joblib.load(MODEL_DIR/"anomaly_model.pkl")
        print(f"  {g('✓')} Anomaly:     estimators={anomaly_model.n_estimators}")
except Exception as e:
    print(f"  {r('Load error:')} {e}")

# ─── STEP 3: Run real predictions using core_engine pipeline ──────────────────
print(f"\n{b('STEP 3 — Running test predictions using core_engine real pipeline...')}")
print(f"  {y('No live HTTP calls — redirect_score=0 for all tests')}\n")

try:
    from core_engine import (
        calculate_domain_score, calculate_url_score,
        calculate_keyword_score_and_type, calculate_security_score,
        is_trusted_domain, is_gambling_platform,
    )
except ImportError as e:
    print(f"  {r('Cannot import core_engine:')} {e}")
    sys.exit(1)

# (url, expected_risk_label, expected_type, display_label)
TEST_CASES = [
    # Safe
    ("https://google.com",                                    0, "Safe",             "Google"),
    ("https://github.com",                                    0, "Safe",             "GitHub"),
    ("https://wikipedia.org",                                 0, "Safe",             "Wikipedia"),
    ("https://amazon.in",                                     0, "Safe",             "Amazon India"),
    ("https://flipkart.com",                                  0, "Safe",             "Flipkart"),
    ("https://netflix.com",                                   0, "Safe",             "Netflix"),
    ("https://paypal.com",                                    0, "Safe",             "PayPal (legit)"),
    # Gambling/Skill
    ("https://dream11.com",                                   1, "Gambling/Betting", "Dream11"),
    ("https://rummycircle.com",                               1, "Gambling/Betting", "RummyCircle"),
    ("https://mpl.live",                                      1, "Gambling/Betting", "MPL"),
    ("https://bet365.com",                                    1, "Gambling/Betting", "Bet365"),
    ("https://pokerstars.com",                                1, "Gambling/Betting", "PokerStars"),
    # Phishing
    ("http://paypal-login-verify.tk/account/secure/confirm",         2, "Phishing", "PayPal phish"),
    ("http://sbi-netbanking-verify.tk/login/account/secure",         2, "Phishing", "SBI bank phish"),
    ("http://185.220.101.90/login/account/verify/credential",        2, "Phishing", "IP phish"),
    ("http://microsoft-account-verify.tk/login/security/credential", 2, "Phishing", "MS phish"),
    # Malware
    ("http://fake-update.top/download/install/flash-player-setup.exe",       2, "Malware", "Fake Flash update"),
    ("http://flash-player-update-now.tk/download/install/flash-codec.exe",   2, "Malware", "Flash malware"),
    ("http://malware-host.xyz/download/install/update.exe",                  2, "Malware", "Malware host"),
    ("http://virus-detected-alert.ml/download/install/scanner-setup.exe",    2, "Malware", "Fake AV"),
    # Scam
    ("http://you-have-won-iphone.ml/claim-now/free/prize/reward",            2, "Scam", "iPhone prize scam"),
    ("http://kbc-lottery-winner.tk/claim-prize/bonus/free/reward",           2, "Scam", "KBC lottery scam"),
    ("http://congratulations-winner.tk/claim/prize/free/bonus",              2, "Scam", "Congrats winner"),
    ("http://guaranteed-profit-bonus.xyz/free/claim/prize/winner",           2, "Scam", "Guaranteed profit scam"),
    # Piracy
    ("http://mod-apk-download.ml/netflix-mod-apk-cracked-unlocked-premium",  2, "Piracy", "Netflix mod APK"),
    ("http://skidrow-games.ml/gta6-crack-keygen-repack-free-download",       2, "Piracy", "GTA6 piracy"),
    ("http://warez-download.ml/software-crack-keygen-serial-nulled-warez",   2, "Piracy", "Warez download"),
    # Financial Fraud
    ("http://bitcoin-invest-profit.tk/invest/bitcoin/trading/profit/wallet", 2, "Financial Fraud", "Bitcoin fraud"),
    ("http://forex-trading-profit.ml/invest/forex/trading/profit/bitcoin",   2, "Financial Fraud", "Forex fraud"),
]

risk_map   = {0: "Low", 1: "Medium", 2: "High"}
correct_risk = 0
correct_type = 0
total = len(TEST_CASES)

header = f"  {'Label':<28} {'Expected':<10} {'Got':<10} {'Type Expected':<18} {'Type Got':<18} {'Risk':<6} {'Type'}"
print(header)
print("  " + "─" * 102)

type_failures = []

for url, exp_label, exp_type, label in TEST_CASES:
    from urllib.parse import urlparse
    domain = urlparse(url).netloc.split(':')[0]

    # ── Use EXACT same pipeline as core_engine ────────────────────────────────
    ds = calculate_domain_score(domain)
    us = calculate_url_score(url)
    ks, inferred_type = calculate_keyword_score_and_type(url, domain)
    ss = calculate_security_score(url)
    rs = 0   # no live redirect check
    feat = np.array([[ds, us, ks, ss, rs]])

    # Risk level prediction
    if is_trusted_domain(domain):
        pred_label = 0
    elif risk_model:
        pred_label = int(risk_model.predict(feat)[0])
    else:
        # Rule-based fallback
        total_score = ds + us + ks + ss + rs
        if is_gambling_platform(domain): pred_label = 1
        elif total_score > 60:           pred_label = 2
        elif total_score > 35:           pred_label = 1
        else:                            pred_label = 0

    # Type prediction
    if is_trusted_domain(domain):
        pred_type = "Safe"
    elif risk_type_model and pred_label > 0:
        try:    pred_type = risk_type_model.predict(feat)[0]
        except: pred_type = inferred_type
    else:
        pred_type = inferred_type if pred_label > 0 else "Safe"

    risk_ok = pred_label == exp_label
    type_ok = pred_type  == exp_type
    if risk_ok: correct_risk += 1
    if type_ok: correct_type += 1

    ri = g("PASS") if risk_ok else r("FAIL")
    ti = g("PASS") if type_ok else r("FAIL")

    if not type_ok:
        type_failures.append((label, exp_type, pred_type, ds, us, ks, ss, rs, inferred_type))

    print(f"  {label:<28} {risk_map[exp_label]:<10} {risk_map.get(pred_label,'?'):<10} "
          f"{exp_type:<18} {str(pred_type):<18} {ri:<14} {ti}")

risk_acc = correct_risk / total * 100
type_acc = correct_type / total * 100

print("\n  " + "─" * 70)
print(f"  Risk Level Accuracy: {correct_risk}/{total}  ", end="")
if risk_acc == 100:   print(g(f"→ {risk_acc:.0f}%  ✓  PERFECT"))
elif risk_acc >= 80:  print(g(f"→ {risk_acc:.0f}%  ✓  GOOD"))
elif risk_acc >= 60:  print(y(f"→ {risk_acc:.0f}%  ⚠  ACCEPTABLE"))
else:                 print(r(f"→ {risk_acc:.0f}%  ✗  POOR — retrain"))

print(f"  Type Accuracy:       {correct_type}/{total}  ", end="")
if type_acc >= 85:    print(g(f"→ {type_acc:.0f}%  ✓  GOOD"))
elif type_acc >= 65:  print(y(f"→ {type_acc:.0f}%  ⚠  ACCEPTABLE"))
else:                 print(r(f"→ {type_acc:.0f}%  ✗  POOR — check feature scores below"))

# Show diagnostic for type failures
if type_failures:
    print(f"\n  {b('TYPE FAILURE DIAGNOSTICS:')} (shows real features the model saw)")
    print(f"  {'Label':<28} {'Expected':<18} {'Got':<18} {'Features [ds,us,ks,ss,rs]':<30} {'Inferred kw type'}")
    print("  " + "─" * 102)
    for label, exp, got, ds, us, ks, ss, rs, inf in type_failures:
        print(f"  {label:<28} {exp:<18} {str(got):<18} [{ds:2},{us:2},{ks:2},{ss:2},{rs:2}]"
              f"                 {inf}")
    print(f"\n  {y('Tip: If inferred kw type ≠ expected type, add more category-specific')}")
    print(f"  {y('     keywords to those URLs in train_model.py and retrain.')}")

# ─── STEP 4: Feature score table for key URLs ─────────────────────────────────
print(f"\n{b('STEP 4 — Feature score table for insight...')}")
print(f"  {'Type':<18} {'Avg domain':>10} {'Avg url':>8} {'Avg keyword':>12} "
      f"{'Avg security':>13} {'Avg redirect':>13}")
print("  " + "─" * 76)

from collections import defaultdict
type_scores = defaultdict(list)
for url, exp_label, exp_type, label in TEST_CASES:
    from urllib.parse import urlparse
    domain = urlparse(url).netloc.split(':')[0]
    ds = calculate_domain_score(domain)
    us = calculate_url_score(url)
    ks, _ = calculate_keyword_score_and_type(url, domain)
    ss = calculate_security_score(url)
    type_scores[exp_type].append([ds, us, ks, ss, 0])

for t, scores in sorted(type_scores.items()):
    arr = np.array(scores)
    avg = arr.mean(axis=0)
    print(f"  {t:<18} {avg[0]:>10.1f} {avg[1]:>8.1f} {avg[2]:>12.1f} "
          f"{avg[3]:>13.1f} {avg[4]:>13.1f}")

# ─── STEP 5: DB stats ─────────────────────────────────────────────────────────
print(f"\n{b('STEP 5 — Database statistics...')}")
try:
    from database import initialize_database, get_record_count, get_class_distribution
    initialize_database()
    count = get_record_count()
    risk_dist, type_dist = get_class_distribution()
    rl = {0: "Low (Safe)", 1: "Medium (Gambling)", 2: "High (Malicious)"}
    print(f"  Total URLs: {b(str(count))}")
    for k in sorted(risk_dist):
        v = risk_dist[k]; pct = v/count*100 if count else 0
        print(f"    {rl.get(k,k):<22} {v:>4} ({pct:4.1f}%)  {'█'*(v//3)}")
    print()
    for k, v in sorted(type_dist.items(), key=lambda x: x[1], reverse=True):
        pct = v/count*100 if count else 0
        print(f"    {str(k):<22} {v:>4} ({pct:4.1f}%)  {'█'*(v//3)}")
except Exception as e:
    print(f"  {r('DB error:')} {e}")

# ─── Summary ─────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(b("  SUMMARY"))
print("="*70)
if all_present and risk_acc >= 80 and type_acc >= 75:
    print(f"  {g('✓')} Models are working well — ready for production!")
elif all_present:
    print(f"  {y('⚠')} Models trained but type accuracy needs improvement.")
    print(f"  {y('  → Check the TYPE FAILURE DIAGNOSTICS above.')}")
    print(f"  {y('  → Ensure failing URL types have strong keyword signals.')}")
else:
    print(f"  {r('✗')} Run train_model.py first.")
print(f"  → Start:  uvicorn backend.main:app --reload --port 8000")
print(f"  → UI:     streamlit run frontend/app.py")
print("="*70 + "\n")