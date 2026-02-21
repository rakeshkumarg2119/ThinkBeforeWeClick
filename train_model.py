"""
╔══════════════════════════════════════════════════════════════════════════╗
║       URL RISK ANALYZER — TRAINING SCRIPT v3 (FINAL)                    ║
║                                                                          ║
║  Root cause of 56% type accuracy found via runtime analysis:             ║
║                                                                          ║
║  v2 problem: Training used SYNTHETIC feature values that did not         ║
║  match what core_engine ACTUALLY computes from those URLs at runtime.    ║
║  Example: "fake-update.top/windows/patch.exe"                            ║
║    v2 trained with:  keyword_score=21 (Malware fingerprint)              ║
║    core_engine gets: keyword_score=5  (only 1 malware keyword hit)       ║
║  The model learned the wrong number, so inference fails.                 ║
║                                                                          ║
║  v3 fix: Call core_engine's OWN scoring functions directly on each URL   ║
║  to get the EXACT same feature values the model will see at inference.   ║
║  Training and inference now use identical feature computation.           ║
║                                                                          ║
║  Additionally: URLs with ambiguous keyword profiles are reinforced        ║
║  by adding keyword-rich variants that give the model clearer signals.    ║
╚══════════════════════════════════════════════════════════════════════════╝

HOW TO RUN:
    Step 1 — Delete old data:
        Windows:  del db\\url_risk.db  &&  del models\\*.pkl
        Mac/Linux: rm db/url_risk.db  &&  rm models/*.pkl

    Step 2 — python train_model.py

    Step 3 — python verify_model.py
             Expected: Risk Level 100%,  Type >80%
"""

import sys, sqlite3, threading
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from database import initialize_database, get_record_count, get_class_distribution
from core_engine import (
    train_models,
    calculate_domain_score,
    calculate_url_score,
    calculate_keyword_score_and_type,
    calculate_security_score,
    is_trusted_domain,
    is_gambling_platform,
)

DB_DIR  = ROOT / "db"
DB_PATH = DB_DIR / "url_risk.db"
db_lock = threading.Lock()

initialize_database()


# ─────────────────────────────────────────────────────────────────────────────
# CORE FIX: Use core_engine's OWN functions to compute every feature score.
# This guarantees training features == inference features — no mismatch.
# ─────────────────────────────────────────────────────────────────────────────
def compute_real_features(url: str, domain: str) -> dict:
    """Call core_engine's exact same pipeline used during live analysis."""
    ds = calculate_domain_score(domain)
    us = calculate_url_score(url)
    ks, inferred_type = calculate_keyword_score_and_type(url, domain)
    ss = calculate_security_score(url)
    rs = 0   # redirect score: 0 for synthetic data (no live HTTP call)
    total = min(ds + us + ks + ss + rs, 100)
    return {
        "domain_score":   ds,
        "url_score":      us,
        "keyword_score":  ks,
        "security_score": ss,
        "redirect_score": rs,
        "total_score":    total,
        "inferred_type":  inferred_type,
    }


def store_row(url, domain, risk_label, risk_type):
    """Compute real features then store in SQLite. Skip if URL already exists."""
    f = compute_real_features(url, domain)

    # Confidence + severity based on risk label
    conf_map = {0: 94.0, 1: 71.0, 2: 85.0}
    sev_map  = {0: 4,    1: 42,   2: 76}
    confidence = conf_map[risk_label]
    severity   = sev_map[risk_label]

    if risk_type == "Safe":
        why = "Verified trusted domain"
    elif risk_type == "Gambling/Betting":
        why = "Real money gaming platform — financial risk present"
    else:
        parts = []
        if f["domain_score"] > 10: parts.append("suspicious domain/TLD")
        if f["keyword_score"] > 10: parts.append(f"{risk_type.lower()} keywords detected")
        if f["security_score"] > 0: parts.append("no HTTPS encryption")
        if f["url_score"] > 10: parts.append("suspicious URL structure")
        why = ", ".join(parts).capitalize() if parts else "risk indicators present"

    with db_lock:
        conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        try:
            conn.execute("""
                INSERT OR IGNORE INTO url_analysis (
                    url, domain, domain_score, url_score, keyword_score,
                    security_score, redirect_score, total_score,
                    predicted_risk_level, predicted_risk_type,
                    confidence_percent, anomaly_detected,
                    risk_severity_index, why_risk
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                url, domain,
                f["domain_score"], f["url_score"], f["keyword_score"],
                f["security_score"], f["redirect_score"], f["total_score"],
                risk_label, risk_type,
                confidence, 0, severity, why,
            ))
            conn.commit()
            return conn.total_changes > 0
        finally:
            conn.close()


def inject(url_list, risk_label, risk_type, label):
    ok = sum(1 for url, domain in url_list
             if store_row(url, domain, risk_label, risk_type))
    print(f"  ✓ {label}: {ok}/{len(url_list)} URLs loaded")
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# KEY INSIGHT from runtime analysis:
#
# Some URLs have LOW keyword counts because their keywords are SHORT and
# shared across categories. We solve this by using URLs that have RICH,
# UNAMBIGUOUS keyword signals — multiple strong keywords from one category.
#
# Bad  (ambiguous): "fake-update.top/patch.exe"        → only 1 malware kw
# Good (clear):     "fake-update.top/download/flash-player-setup.exe"
#                                                       → download+flash+setup = 3 hits
# ─────────────────────────────────────────────────────────────────────────────

SAFE_URLS = [
    # Big Tech
    ("https://google.com",                    "google.com"),
    ("https://www.google.co.in",              "google.co.in"),
    ("https://mail.google.com",               "mail.google.com"),
    ("https://drive.google.com",              "drive.google.com"),
    ("https://docs.google.com",               "docs.google.com"),
    ("https://youtube.com",                   "youtube.com"),
    ("https://github.com",                    "github.com"),
    ("https://github.com/trending",           "github.com"),
    ("https://stackoverflow.com",             "stackoverflow.com"),
    ("https://microsoft.com",                 "microsoft.com"),
    ("https://office.com",                    "office.com"),
    ("https://outlook.com",                   "outlook.com"),
    ("https://apple.com",                     "apple.com"),
    ("https://icloud.com",                    "icloud.com"),
    ("https://amazon.com",                    "amazon.com"),
    ("https://amazon.in",                     "amazon.in"),
    ("https://aws.amazon.com",                "aws.amazon.com"),
    ("https://facebook.com",                  "facebook.com"),
    ("https://instagram.com",                 "instagram.com"),
    ("https://twitter.com",                   "twitter.com"),
    ("https://linkedin.com",                  "linkedin.com"),
    ("https://reddit.com",                    "reddit.com"),
    ("https://discord.com",                   "discord.com"),
    ("https://whatsapp.com",                  "whatsapp.com"),
    ("https://telegram.org",                  "telegram.org"),
    ("https://zoom.us",                       "zoom.us"),
    ("https://slack.com",                     "slack.com"),
    # Dev tools
    ("https://npmjs.com",                     "npmjs.com"),
    ("https://pypi.org",                      "pypi.org"),
    ("https://docker.com",                    "docker.com"),
    ("https://cloudflare.com",                "cloudflare.com"),
    ("https://gitlab.com",                    "gitlab.com"),
    ("https://bitbucket.org",                 "bitbucket.org"),
    ("https://heroku.com",                    "heroku.com"),
    ("https://vercel.com",                    "vercel.com"),
    ("https://netlify.com",                   "netlify.com"),
    ("https://digitalocean.com",              "digitalocean.com"),
    # Education
    ("https://wikipedia.org",                 "wikipedia.org"),
    ("https://en.wikipedia.org/wiki/Machine_learning", "en.wikipedia.org"),
    ("https://coursera.org",                  "coursera.org"),
    ("https://udemy.com",                     "udemy.com"),
    ("https://khanacademy.org",               "khanacademy.org"),
    ("https://edx.org",                       "edx.org"),
    ("https://mit.edu",                       "mit.edu"),
    ("https://stanford.edu",                  "stanford.edu"),
    # News
    ("https://bbc.com/news",                  "bbc.com"),
    ("https://cnn.com",                       "cnn.com"),
    ("https://reuters.com",                   "reuters.com"),
    ("https://theguardian.com",               "theguardian.com"),
    ("https://nytimes.com",                   "nytimes.com"),
    ("https://thehindu.com",                  "thehindu.com"),
    ("https://ndtv.com",                      "ndtv.com"),
    ("https://timesofindia.com",              "timesofindia.com"),
    # Finance (legit)
    ("https://paypal.com",                    "paypal.com"),
    ("https://stripe.com",                    "stripe.com"),
    ("https://razorpay.com",                  "razorpay.com"),
    ("https://paytm.com",                     "paytm.com"),
    ("https://phonepe.com",                   "phonepe.com"),
    ("https://visa.com",                      "visa.com"),
    ("https://mastercard.com",                "mastercard.com"),
    # Entertainment
    ("https://netflix.com",                   "netflix.com"),
    ("https://spotify.com",                   "spotify.com"),
    ("https://hotstar.com",                   "hotstar.com"),
    ("https://primevideo.com",                "primevideo.com"),
    ("https://hulu.com",                      "hulu.com"),
    ("https://twitch.tv",                     "twitch.tv"),
    # India Govt
    ("https://india.gov.in",                  "india.gov.in"),
    ("https://mygov.in",                      "mygov.in"),
    ("https://irctc.co.in",                   "irctc.co.in"),
    ("https://incometax.gov.in",              "incometax.gov.in"),
    ("https://uidai.gov.in",                  "uidai.gov.in"),
    ("https://epfindia.gov.in",               "epfindia.gov.in"),
    ("https://digilocker.gov.in",             "digilocker.gov.in"),
    # eCommerce
    ("https://flipkart.com",                  "flipkart.com"),
    ("https://myntra.com",                    "myntra.com"),
    ("https://meesho.com",                    "meesho.com"),
    ("https://nykaa.com",                     "nykaa.com"),
    ("https://ebay.com",                      "ebay.com"),
    ("https://etsy.com",                      "etsy.com"),
    ("https://shopify.com",                   "shopify.com"),
    # SaaS / Productivity
    ("https://medium.com",                    "medium.com"),
    ("https://notion.so",                     "notion.so"),
    ("https://figma.com",                     "figma.com"),
    ("https://canva.com",                     "canva.com"),
    ("https://wordpress.com",                 "wordpress.com"),
    ("https://hubspot.com",                   "hubspot.com"),
    ("https://salesforce.com",                "salesforce.com"),
    ("https://trello.com",                    "trello.com"),
    ("https://atlassian.com",                 "atlassian.com"),
    ("https://dropbox.com",                   "dropbox.com"),
    ("https://adobe.com",                     "adobe.com"),
    ("https://jetbrains.com",                 "jetbrains.com"),
    ("https://openai.com",                    "openai.com"),
    ("https://anthropic.com",                 "anthropic.com"),
    ("https://huggingface.co",                "huggingface.co"),
    ("https://wix.com",                       "wix.com"),
    ("https://squarespace.com",               "squarespace.com"),
    ("https://asana.com",                     "asana.com"),
    ("https://box.com",                       "box.com"),
    ("https://autodesk.com",                  "autodesk.com"),
    ("https://zendesk.com",                   "zendesk.com"),
    ("https://intercom.com",                  "intercom.com"),
    ("https://mailchimp.com",                 "mailchimp.com"),
]

GAMBLING_SKILL_URLS = [
    # Fantasy Sports
    ("https://dream11.com",                   "dream11.com"),
    ("https://www.dream11.com/games",         "dream11.com"),
    ("https://dream11.com/fantasy-cricket",   "dream11.com"),
    ("https://my11circle.com",                "my11circle.com"),
    ("https://ballebaazi.com",                "ballebaazi.com"),
    ("https://howzat.com",                    "howzat.com"),
    ("https://gamezy.com",                    "gamezy.com"),
    ("https://myteam11.com",                  "myteam11.com"),
    ("https://playerzpot.com",                "playerzpot.com"),
    ("https://halaplay.com",                  "halaplay.com"),
    ("https://fantasypower11.com",            "fantasypower11.com"),
    # Rummy
    ("https://rummycircle.com",               "rummycircle.com"),
    ("https://ace2three.com",                 "ace2three.com"),
    ("https://classicrummy.com",              "classicrummy.com"),
    ("https://junglee.com",                   "junglee.com"),
    ("https://rummyculture.com",              "rummyculture.com"),
    ("https://rummytime.com",                 "rummytime.com"),
    ("https://khelplayrummy.com",             "khelplayrummy.com"),
    ("https://addarummy.com",                 "addarummy.com"),
    ("https://rummynabob.com",                "rummynabob.com"),
    ("https://rummypassion.com",              "rummypassion.com"),
    # Multi-game
    ("https://mpl.live",                      "mpl.live"),
    ("https://www.mpl.live/games",            "mpl.live"),
    ("https://winzo.com",                     "winzo.com"),
    ("https://zupee.com",                     "zupee.com"),
    ("https://getmega.com",                   "getmega.com"),
    ("https://firstgames.in",                 "firstgames.in"),
    ("https://paytmfirstgames.com",           "paytmfirstgames.com"),
    # Poker
    ("https://pokerbaazi.com",                "pokerbaazi.com"),
    ("https://adda52.com",                    "adda52.com"),
    ("https://khelo365.com",                  "khelo365.com"),
    # International licensed
    ("https://bet365.com",                    "bet365.com"),
    ("https://betway.com",                    "betway.com"),
    ("https://1xbet.com",                     "1xbet.com"),
    ("https://10cric.com",                    "10cric.com"),
    ("https://dafabet.com",                   "dafabet.com"),
    ("https://parimatch.in",                  "parimatch.in"),
    ("https://betfair.com",                   "betfair.com"),
    ("https://pokerstars.com",                "pokerstars.com"),
    ("https://draftkings.com",                "draftkings.com"),
    ("https://fanduel.com",                   "fanduel.com"),
    ("https://caesars.com",                   "caesars.com"),
]

# ── PHISHING: keyword-rich URLs with LOGIN/VERIFY/SECURE/ACCOUNT/CREDENTIAL ──
PHISHING_URLS = [
    ("http://paypal-login-verify.tk/account/secure/confirm",         "paypal-login-verify.tk"),
    ("http://paypal-security-alert.ml/signin/account/update",        "paypal-security-alert.ml"),
    ("http://secure-paypal-account.xyz/login/verify/credential",     "secure-paypal-account.xyz"),
    ("http://pp-login-confirm.club/account/validate/secure",         "pp-login-confirm.club"),
    ("http://sbi-netbanking-verify.tk/login/account/secure",         "sbi-netbanking-verify.tk"),
    ("http://hdfcbank-secure-login.ml/account/update/credential",    "hdfcbank-secure-login.ml"),
    ("http://icici-account-verify.xyz/netbanking/login/secure",      "icici-account-verify.xyz"),
    ("http://axisbank-secure.tk/auth/login/account/verify",          "axisbank-secure.tk"),
    ("http://kotak-bank-login.top/signin/verify/credential",         "kotak-bank-login.top"),
    ("http://pnb-netbanking.ml/account/secure/login/verify",         "pnb-netbanking.ml"),
    ("http://google-account-verify.xyz/signin/security/credential",  "google-account-verify.xyz"),
    ("http://accounts-googIe.tk/ServiceLogin/verify/secure",         "accounts-googIe.tk"),
    ("http://gmail-login-secure.ml/accounts/signin/verify",          "gmail-login-secure.ml"),
    ("http://microsoft-account-verify.tk/login/security/credential", "microsoft-account-verify.tk"),
    ("http://office365-login-confirm.xyz/owa/auth/account/verify",   "office365-login-confirm.xyz"),
    ("http://outlook-verify-email.ml/owa/login/credential/secure",   "outlook-verify-email.ml"),
    ("http://amazon-account-suspended.tk/signin/verify/secure",      "amazon-account-suspended.tk"),
    ("http://amazon-prime-verify.xyz/account/signin/credential",     "amazon-prime-verify.xyz"),
    ("http://amaz0n-security-alert.ml/ap/signin/account/verify",     "amaz0n-security-alert.ml"),
    ("http://irctc-account-verify.tk/signin/credential/secure",      "irctc-account-verify.tk"),
    ("http://uidai-aadhar-update.ml/resident/login/verify",          "uidai-aadhar-update.ml"),
    ("http://paytm-kyc-verify.xyz/login/account/update/secure",      "paytm-kyc-verify.xyz"),
    ("http://epf-india-login.tk/member/account/credential/verify",   "epf-india-login.tk"),
    ("http://185.220.101.90/login/account/verify/credential",        "185.220.101.90"),
    ("http://45.133.174.12/paypal/signin/account/verify",            "45.133.174.12"),
    ("http://194.165.16.28/bank/secure/login/verify",                "194.165.16.28"),
    ("http://faceb00k-login.tk/login/account/verify",                "faceb00k-login.tk"),
    ("http://lnstagram-verify.ml/accounts/login/secure",             "lnstagram-verify.ml"),
    ("http://linkedIn-jobs-verify.xyz/login/account/security",       "linkedIn-jobs-verify.xyz"),
    ("http://whatsapp-account-verify.tk/verify/login/credential",    "whatsapp-account-verify.tk"),
    ("http://credential-update-required.ml/secure/account/login",    "credential-update-required.ml"),
    ("http://password-reset-alert.tk/account/verify/credential",     "password-reset-alert.tk"),
    ("http://authenticate-your-account.xyz/signin/credential",       "authenticate-your-account.xyz"),
    ("http://account-suspended-verify.top/login/secure/credential",  "account-suspended-verify.top"),
    ("http://security-warning-login.ml/validate/account/verify",     "security-warning-login.ml"),
    ("http://blocked-account-verify.tk/secure/login/credential",     "blocked-account-verify.tk"),
    ("http://login-update-required.xyz/account/signin/verify",       "login-update-required.xyz"),
    ("http://verify-secure-account.ml/login/credential/update",      "verify-secure-account.ml"),
    ("http://signin-account-warning.tk/login/secure/verify",         "signin-account-warning.tk"),
    ("http://update-credential-now.top/account/login/verify",        "update-credential-now.top"),
]

# ── MALWARE: keyword-rich with DOWNLOAD+EXE+INSTALL+SETUP+CODEC+FLASH ────────
MALWARE_URLS = [
    ("http://185.220.101.90/files/download/setup.exe",                        "185.220.101.90"),
    ("http://194.165.16.28/install/download/setup.exe",                       "194.165.16.28"),
    ("http://malware-host.xyz/download/install/update.exe",                   "malware-host.xyz"),
    ("http://fake-update.top/download/install/flash-player-setup.exe",        "fake-update.top"),
    ("http://driver-update-now.ml/download/install/driver-setup.exe",         "driver-update-now.ml"),
    ("http://123.188.83.131/download/install/bin.sh",                         "123.188.83.131"),
    ("http://182.126.247.177/download/install/setup.sh",                      "182.126.247.177"),
    ("http://botnet-c2.xyz/download/install/payload-setup.exe",               "botnet-c2.xyz"),
    ("http://45.95.147.230/download/install/bot-setup.sh",                    "45.95.147.230"),
    ("http://flash-player-update-now.tk/download/install/flash-codec.exe",    "flash-player-update-now.tk"),
    ("http://java-update-required.ml/download/install/java-setup.exe",        "java-update-required.ml"),
    ("http://plugin-required-update.xyz/download/install/codec-plugin.exe",   "plugin-required-update.xyz"),
    ("http://browser-update-alert.top/download/install/chrome-setup.exe",     "browser-update-alert.top"),
    ("http://free-software-download.tk/download/install/activator-setup.exe", "free-software-download.tk"),
    ("http://patch-download.xyz/download/install/windows-activator.exe",      "patch-download.xyz"),
    ("http://invoice-download.tk/download/install/invoice-macro.doc",         "invoice-download.tk"),
    ("http://dhl-shipment.ml/download/install/document-setup.doc",            "dhl-shipment.ml"),
    ("http://your-pc-is-infected.top/download/install/antivirus-setup.exe",   "your-pc-is-infected.top"),
    ("http://virus-detected-alert.ml/download/install/scanner-setup.exe",     "virus-detected-alert.ml"),
    ("http://mining-software.xyz/download/install/xmrig-setup.exe",           "mining-software.xyz"),
    ("http://crypto-miner-tool.tk/download/install/miner-setup.exe",          "crypto-miner-tool.tk"),
    ("http://wordpress-backup.ml/download/install/payload-plugin.exe",        "wordpress-backup.ml"),
    ("http://security-scan-now.tk/download/install/antivirus-codec.exe",      "security-scan-now.tk"),
    ("http://system-cleaner.ml/download/install/optimizer-setup.exe",         "system-cleaner.ml"),
    ("http://codec-pack-required.xyz/download/install/codec-setup-flash.exe", "codec-pack-required.xyz"),
    ("http://activex-plugin-update.tk/download/install/activex-setup.exe",    "activex-plugin-update.tk"),
    ("http://java-plugin-download.ml/download/install/java-plugin.exe",       "java-plugin-download.ml"),
    ("http://windows-update-now.xyz/download/install/update-setup.exe",       "windows-update-now.xyz"),
]

# ── SCAM: keyword-rich with WINNER+PRIZE+CLAIM+FREE+BONUS+REWARD+LOTTERY ──────
SCAM_URLS = [
    ("http://congratulations-winner.tk/claim/prize/free/bonus",               "congratulations-winner.tk"),
    ("http://you-have-won-iphone.ml/claim-now/free/prize/reward",             "you-have-won-iphone.ml"),
    ("http://lucky-draw-winner-2024.xyz/claim/prize/winner/free",             "lucky-draw-winner-2024.xyz"),
    ("http://amazon-prize-claim.top/winner/free/bonus/reward",                "amazon-prize-claim.top"),
    ("http://kbc-lottery-winner.tk/claim-prize/bonus/free/reward",            "kbc-lottery-winner.tk"),
    ("http://jio-free-recharge-winner.ml/free/claim/prize/reward",            "jio-free-recharge-winner.ml"),
    ("http://guaranteed-profit-bonus.xyz/free/claim/prize/winner",            "guaranteed-profit-bonus.xyz"),
    ("http://risk-free-bonus-offer.top/claim/free/prize/reward",              "risk-free-bonus-offer.top"),
    ("http://earn-free-gift-bonus.tk/claim/prize/winner/reward",              "earn-free-gift-bonus.tk"),
    ("http://work-from-home-free-reward.ml/claim/bonus/prize/winner",         "work-from-home-free-reward.ml"),
    ("http://google-free-prize-winner.xyz/claim/bonus/reward/lottery",        "google-free-prize-winner.xyz"),
    ("http://tcs-free-gift-claim.tk/winner/prize/free/bonus",                 "tcs-free-gift-claim.tk"),
    ("http://microsoft-free-reward.ml/claim/prize/bonus/winner",              "microsoft-free-reward.ml"),
    ("http://windows-free-gift.xyz/claim/prize/winner/bonus/reward",          "windows-free-gift.xyz"),
    ("http://pm-kisan-free-bonus.tk/claim/prize/lottery/reward",              "pm-kisan-free-bonus.tk"),
    ("http://free-gas-cylinder-prize.ml/claim/bonus/winner/reward",           "free-gas-cylinder-prize.ml"),
    ("http://ration-card-winner.xyz/claim/free/prize/bonus/lottery",          "ration-card-winner.xyz"),
    ("http://iphone-free-prize.tk/winner/claim/bonus/reward",                 "iphone-free-prize.tk"),
    ("http://limited-offer-free-reward.ml/claim/prize/bonus/winner",          "limited-offer-free-reward.ml"),
    ("http://congratulations-selected.top/gift/free/prize/claim/bonus",       "congratulations-selected.top"),
    ("http://spin-win-free-prize.ml/reward/bonus/claim/winner",               "spin-win-free-prize.ml"),
    ("http://survey-winner-free-claim.xyz/prize/bonus/reward/lottery",        "survey-winner-free-claim.xyz"),
    ("http://free-gift-voucher-claim.tk/winner/bonus/prize/reward",           "free-gift-voucher-claim.tk"),
    ("http://guaranteed-winner-prize.ml/claim/free/bonus/lottery/reward",     "guaranteed-winner-prize.ml"),
    ("http://no-cost-free-prize.xyz/claim/winner/bonus/gift/reward",          "no-cost-free-prize.xyz"),
]

# ── PIRACY: keyword-rich with CRACK+KEYGEN+NULLED+WAREZ+TORRENT+REPACK+MOD-APK
PIRACY_URLS = [
    ("https://fitgirl-repacks.site/adobe-photoshop-crack-keygen-download",    "fitgirl-repacks.site"),
    ("https://cracksnow.com/windows11-crack-keygen-serial-free-download",     "cracksnow.com"),
    ("http://fullcrack4u.com/office-2021-crack-serial-keygen-nulled",         "fullcrack4u.com"),
    ("http://download-crack-serial.xyz/autocad-crack-keygen-warez-download",  "download-crack-serial.xyz"),
    ("http://nulled-scripts.tk/wordpress-premium-nulled-warez-crack",         "nulled-scripts.tk"),
    ("http://warez-download.ml/software-crack-keygen-serial-nulled-warez",    "warez-download.ml"),
    ("http://crackzplanet.com/idm-crack-keygen-serial-patch-download",        "crackzplanet.com"),
    ("http://serialkeyfree.xyz/windows10-crack-serial-keygen-free-download",  "serialkeyfree.xyz"),
    ("http://movies4u.tk/bollywood-movies-free-download-torrent-hd",          "movies4u.tk"),
    ("http://9xmovies.ml/hindi-movies-torrent-free-download-hd",              "9xmovies.ml"),
    ("http://tamilrockers.xyz/movies-torrent-free-download-crack-hd",         "tamilrockers.xyz"),
    ("http://filmywap.top/movies-torrent-crack-free-download-hd",             "filmywap.top"),
    ("http://123movies-free.xyz/watch-movies-torrent-crack-free",             "123movies-free.xyz"),
    ("http://pirate-bay-proxy.tk/torrents-crack-warez-free-download",         "pirate-bay-proxy.tk"),
    ("http://mod-apk-download.ml/netflix-mod-apk-cracked-unlocked-premium",   "mod-apk-download.ml"),
    ("http://hacked-apk.xyz/whatsapp-mod-apk-cracked-nulled-free",            "hacked-apk.xyz"),
    ("http://apk-unlocked.tk/spotify-premium-mod-apk-cracked-nulled",         "apk-unlocked.tk"),
    ("http://premium-free-apk.ml/amazon-prime-mod-apk-crack-nulled",          "premium-free-apk.ml"),
    ("http://skidrow-games.ml/gta6-crack-keygen-repack-free-download",        "skidrow-games.ml"),
    ("http://codex-games.xyz/call-of-duty-crack-keygen-repack-download",      "codex-games.xyz"),
    ("http://dodi-repacks.site/elden-ring-crack-keygen-repack-download",      "dodi-repacks.site"),
    ("http://ocean-of-games.ml/game-crack-keygen-repack-nulled-warez",        "ocean-of-games.ml"),
    ("http://repack-games.xyz/cyberpunk-crack-repack-keygen-download",        "repack-games.xyz"),
    ("http://warez-bb.ml/software-nulled-crack-keygen-serial-warez",          "warez-bb.ml"),
    ("http://crackedpro.xyz/adobe-premiere-crack-keygen-serial-nulled",       "crackedpro.xyz"),
    ("http://torrent-pirate.tk/movies-torrent-warez-crack-free-download",     "torrent-pirate.tk"),
    ("http://game-repacks.ml/gta-repack-crack-keygen-nulled-download",        "game-repacks.ml"),
    ("http://warez-scene.xyz/software-warez-crack-serial-keygen-nulled",      "warez-scene.xyz"),
]

# ── FINANCIAL FRAUD: INVEST+CRYPTO+BITCOIN+PROFIT+FOREX+TRADING+WALLET ───────
FINANCIAL_FRAUD_URLS = [
    ("http://bitcoin-invest-profit.tk/invest/bitcoin/trading/profit/wallet",      "bitcoin-invest-profit.tk"),
    ("http://crypto-trading-profit.ml/invest/crypto/trading/profit/bitcoin",      "crypto-trading-profit.ml"),
    ("http://ethereum-invest-profit.xyz/invest/trading/profit/bitcoin/wallet",    "ethereum-invest-profit.xyz"),
    ("http://binance-profit-invest.top/invest/bitcoin/trading/wallet/profit",     "binance-profit-invest.top"),
    ("http://instant-loan-invest.tk/invest/money/profit/trading/bitcoin",         "instant-loan-invest.tk"),
    ("http://loan-invest-profit.ml/invest/trading/profit/bitcoin/money",          "loan-invest-profit.ml"),
    ("http://pre-approved-invest.xyz/invest/trading/profit/crypto/bitcoin",       "pre-approved-invest.xyz"),
    ("http://zero-interest-invest.tk/invest/bitcoin/trading/profit/wallet",       "zero-interest-invest.tk"),
    ("http://forex-trading-profit.ml/invest/forex/trading/profit/bitcoin",        "forex-trading-profit.ml"),
    ("http://stock-trading-profit.xyz/invest/trading/profit/stock/bitcoin",       "stock-trading-profit.xyz"),
    ("http://sebi-invest-profit.tk/invest/trading/profit/forex/bitcoin",          "sebi-invest-profit.tk"),
    ("http://upi-invest-profit.ml/invest/bitcoin/trading/profit/wallet",          "upi-invest-profit.ml"),
    ("http://google-pay-invest.xyz/invest/profit/trading/bitcoin/wallet",         "google-pay-invest.xyz"),
    ("http://phonepe-invest-profit.top/invest/trading/profit/bitcoin/crypto",     "phonepe-invest-profit.top"),
    ("http://mlm-invest-profit.tk/invest/trading/profit/bitcoin/money",           "mlm-invest-profit.tk"),
    ("http://chain-invest-profit.ml/invest/trading/profit/forex/wallet",          "chain-invest-profit.ml"),
    ("http://nse-invest-profit.xyz/invest/trading/profit/stock/bitcoin",          "nse-invest-profit.xyz"),
    ("http://mutual-fund-invest.tk/invest/profit/trading/money/bitcoin",          "mutual-fund-invest.tk"),
    ("http://crypto-wallet-invest.ml/invest/bitcoin/trading/profit/wallet",       "crypto-wallet-invest.ml"),
    ("http://p2p-invest-profit.xyz/invest/trading/profit/money/bitcoin",          "p2p-invest-profit.xyz"),
]


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "="*70)
    print("  URL RISK ANALYZER — TRAINING SCRIPT v3 (FINAL)")
    print("  Core fix: features computed by core_engine itself at training time")
    print("="*70)

    before = get_record_count()
    print(f"\n  Database currently has: {before} URLs")
    if before > 0:
        print("  ⚠️  Existing URLs are skipped (INSERT OR IGNORE).")
        print("  ⚠️  For a clean retrain, delete db/url_risk.db first.\n")

    print("\n  📥 Loading training data...\n")
    inject(SAFE_URLS,            0, "Safe",             "✅ Safe / Trusted Sites")
    inject(GAMBLING_SKILL_URLS,  1, "Gambling/Betting", "🎮 Skill Gaming / Gambling (Medium)")
    inject(PHISHING_URLS,        2, "Phishing",          "🎣 Phishing (High)")
    inject(MALWARE_URLS,         2, "Malware",           "💀 Malware (High)")
    inject(SCAM_URLS,            2, "Scam",              "💸 Scam (High)")
    inject(PIRACY_URLS,          2, "Piracy",            "🏴‍☠️  Piracy (High)")
    inject(FINANCIAL_FRAUD_URLS, 2, "Financial Fraud",   "📉 Financial Fraud (High)")

    after = get_record_count()
    risk_dist, type_dist = get_class_distribution()
    risk_map = {0: "Low (Safe)", 1: "Medium (Gambling)", 2: "High (Malicious)"}

    print(f"\n  📊 New URLs added this run: {after - before}")
    print(f"  📊 Total in database: {after}")

    print("\n  📈 RISK LEVEL DISTRIBUTION:")
    for k, v in sorted(risk_dist.items()):
        print(f"    {risk_map.get(k,k):<28} {v:>4}  {'█'*(v//3)}")

    print("\n  🏷️  RISK TYPE DISTRIBUTION:")
    for k, v in sorted(type_dist.items(), key=lambda x: x[1], reverse=True):
        print(f"    {str(k):<22} {v:>4}  {'█'*(v//3)}")

    low = risk_dist.get(0,0); med = risk_dist.get(1,0); high = risk_dist.get(2,0)
    print(f"\n  ⚖️  Balance: Low={low}  Medium={med}  High={high}  ", end="")
    print("✅ Good!" if low > 0 and med > 0 and high > 0 else "⚠️ Some classes empty!")

    print("\n" + "="*70)
    print("  🔧 TRIGGERING MODEL TRAINING...")
    print("="*70 + "\n")

    if after >= 30:
        success = train_models()
        if success:
            print("\n  ✅ TRAINING COMPLETE — run: python verify_model.py\n")
    else:
        print(f"  ⚠️  Need 30+ URLs (have {after})")

    print("="*70)
    print("  1. python verify_model.py")
    print("  2. uvicorn backend.main:app --reload --port 8000")
    print("  3. streamlit run frontend/app.py")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()