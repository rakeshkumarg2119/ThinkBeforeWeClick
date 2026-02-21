"""
Core URL Risk Analysis Engine v3.1
FIX: Added type_hint_score as 6th feature to the ML pipeline.

WHY: The 5-feature model (domain, url, keyword, security, redirect) cannot
separate Phishing / Malware / Scam / Piracy / Financial Fraud because their
url_score is 0 for all of them (URLs are under 80 chars, no @, no // in path).
This makes feature vectors nearly identical, so RandomForest defaults to
predicting the most common class (Phishing).

FIX: type_hint_score encodes which keyword category had the most matches:
  0 = Safe/Unknown   1 = Gambling   2 = Phishing   3 = Malware
  4 = Scam           5 = Piracy     6 = Financial Fraud

This gives the type classifier a clean integer separator with no overlap.
"""
import os
import re
import joblib
import numpy as np
import requests
import warnings
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from urllib.parse import urlparse
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

from database import (
    initialize_database, get_cached_result, store_analysis,
    get_training_data, get_record_count, get_class_distribution
)

MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

RISK_MODEL_PATH      = MODEL_DIR / "risk_model.pkl"
RISK_TYPE_MODEL_PATH = MODEL_DIR / "risk_type_model.pkl"
ANOMALY_MODEL_PATH   = MODEL_DIR / "anomaly_model.pkl"

MIN_SAMPLES_FOR_TRAINING = 30
RETRAIN_INTERVAL         = 50

# ── Type hint mapping (used for 6th feature) ─────────────────────────────────
TYPE_HINT_MAP = {
    'Unknown':          0,
    'Safe':             0,
    'Gambling/Betting': 1,
    'Phishing':         2,
    'Malware':          3,
    'Scam':             4,
    'Piracy':           5,
    'Financial Fraud':  6,
}

TRUSTED_DOMAINS = {
    'google.com', 'youtube.com', 'gmail.com', 'google.co.in', 'google.co.uk',
    'facebook.com', 'instagram.com', 'whatsapp.com', 'meta.com',
    'microsoft.com', 'office.com', 'outlook.com', 'live.com', 'xbox.com',
    'apple.com', 'icloud.com', 'itunes.com',
    'amazon.com', 'amazon.in', 'amazon.co.uk', 'aws.amazon.com',
    'twitter.com', 'x.com', 'linkedin.com', 'reddit.com', 'discord.com',
    'telegram.org', 'signal.org', 'snapchat.com', 'tiktok.com',
    'github.com', 'gitlab.com', 'stackoverflow.com', 'stackexchange.com',
    'npmjs.com', 'pypi.org', 'docker.com', 'kubernetes.io',
    'wikipedia.org', 'wikimedia.org', 'scholar.google.com',
    'coursera.org', 'udemy.com', 'khanacademy.org', 'edx.org',
    'bbc.com', 'cnn.com', 'nytimes.com', 'reuters.com', 'theguardian.com',
    'paypal.com', 'stripe.com', 'visa.com', 'mastercard.com',
    'netflix.com', 'spotify.com', 'hulu.com', 'primevideo.com',
    'twitch.tv', 'soundcloud.com',
    'gov.in', 'nic.in', 'gov.uk', 'usa.gov', 'irs.gov',
    'cloudflare.com', 'wordpress.com', 'medium.com', 'zoom.us',
    'slack.com', 'notion.so', 'figma.com', 'canva.com',
    'shopify.com', 'flipkart.com', 'myntra.com', 'meesho.com',
    'hotstar.com', 'irctc.co.in', 'razorpay.com', 'paytm.com',
    'phonepe.com', 'openai.com', 'anthropic.com', 'huggingface.co',
}

GAMBLING_PLATFORMS = {
    'rummycircle.com', 'ace2three.com', 'junglee.com', 'classicrummy.com',
    'dream11.com', 'my11circle.com', 'mpl.live', 'paytmfirstgames.com',
    'winzo.com', 'ballebaazi.com', 'howzat.com', 'gamezy.com',
    '1xbet.com', 'betway.com', 'bet365.com', '10cric.com', 'dafabet.com',
    'fairbet.com', 'pure.win', 'parimatch.in', 'betfair.com',
    'poker.com', 'pokerstars.com', 'zynga.com',
    'draftkings.com', 'fanduel.com', 'caesars.com',
    'bovada.lv', 'betonline.ag', 'ignition.casino',
    'pokerbaazi.com', 'adda52.com', 'khelo365.com',
    'rummyculture.com', 'rummytime.com', 'khelplayrummy.com',
    'addarummy.com', 'rummynabob.com', 'rummypassion.com',
    'zupee.com', 'getmega.com', 'firstgames.in',
    'myteam11.com', 'playerzpot.com', 'halaplay.com', 'fantasypower11.com',
    'borgata.com',
}

TLD_REPUTATION = {
    '.tk': 25, '.ml': 25, '.ga': 25, '.cf': 25, '.gq': 25,
    '.top': 20, '.xyz': 18, '.club': 18, '.win': 20, '.bid': 20,
    '.loan': 22, '.work': 18, '.click': 18, '.download': 20,
    '.stream': 18, '.science': 18, '.racing': 18, '.review': 18,
    '.trade': 18, '.date': 18, '.party': 18, '.faith': 18,
    '.site': 12, '.online': 12, '.store': 10, '.tech': 10,
    '.space': 12, '.fun': 12, '.host': 12, '.website': 10,
    '.press': 10, '.news': 10, '.live': 10, '.world': 10,
    '.com': 0, '.org': 0, '.net': 0, '.edu': 0, '.gov': 0,
    '.co.uk': 0, '.co.in': 0, '.de': 0, '.fr': 0, '.jp': 0,
    '.au': 0, '.ca': 0, '.us': 0, '.info': 2, '.biz': 3,
    '.in': 0,
}


def get_tld_score(domain):
    for tld, score in sorted(TLD_REPUTATION.items(), key=lambda x: len(x[0]), reverse=True):
        if domain.endswith(tld):
            return score
    return 5


def is_trusted_domain(domain):
    if domain in TRUSTED_DOMAINS:
        return True
    for trusted in TRUSTED_DOMAINS:
        if domain.endswith('.' + trusted):
            return True
    return False


def is_gambling_platform(domain):
    if domain in GAMBLING_PLATFORMS:
        return True
    for gambling in GAMBLING_PLATFORMS:
        if domain.endswith('.' + gambling) or gambling in domain:
            return True
    return False


def calculate_domain_score(domain):
    if is_trusted_domain(domain):
        return 0
    if is_gambling_platform(domain):
        return 8
    score = 0
    try:
        parts = domain.split('.')
        if all(part.isdigit() for part in parts if part):
            return 25
        tld_score = get_tld_score(domain)
        score += tld_score
        domain_name = domain.split('.')[0]
        if len(domain_name) > 25:   score += 8
        elif len(domain_name) > 15: score += 5
        elif len(domain_name) < 3:  score += 8
        hyphen_count = domain_name.count('-')
        if hyphen_count > 3:   score += 10
        elif hyphen_count > 2: score += 5
        digit_count = sum(c.isdigit() for c in domain_name)
        if digit_count > 5:   score += 8
        elif digit_count > 3: score += 4
        if re.search(r'\d{4}', domain_name): score += 5
    except:
        score = 15
    return min(score, 25)


def calculate_url_score(url):
    score = 0
    try:
        parsed = urlparse(url)
        netloc = parsed.netloc.split(':')[0]
        try:
            import ipaddress
            ipaddress.ip_address(netloc)
            score += 20
        except:
            pass
        if len(url) > 120:  score += 10
        elif len(url) > 80: score += 5
        if '@' in url: score += 15
        subdomain_count = parsed.netloc.count('.')
        if subdomain_count > 3:   score += 8
        elif subdomain_count > 2: score += 4
        special_chars = len(re.findall(r'[!#$%^&*(),?":{}|<>]', url))
        if special_chars > 5: score += 8
        if '//' in parsed.path: score += 10
        if len(parsed.query) > 100: score += 8
    except:
        score = 10
    return min(score, 25)


def calculate_keyword_score_and_type(url, domain):
    """
    Returns (score, risk_type, type_hint_int)
    type_hint_int is the new 6th feature — encodes which category won.
    """
    url_lower    = url.lower()
    domain_lower = domain.lower()
    is_known_gambling = is_gambling_platform(domain)

    phishing_keywords = [
        'login', 'signin', 'verify', 'account', 'update', 'suspend',
        'confirm', 'secure', 'validate', 'authenticate', 'credential',
        'password', 'security', 'alert', 'warning', 'blocked'
    ]
    financial_keywords = [
        'bank', 'paypal', 'wallet', 'payment', 'credit', 'debit',
        'transaction', 'transfer', 'wire', 'swift', 'iban',
        'crypto', 'bitcoin', 'ethereum', 'blockchain', 'invest', 'trading',
        'forex', 'stock', 'profit', 'money'
    ]
    scam_keywords = [
        'reward', 'prize', 'winner', 'congratulations', 'claim',
        'free', 'bonus', 'gift', 'lottery', 'sweepstakes',
        'offer', 'limited', 'expires', 'urgent', 'act-now',
        'guaranteed', 'risk-free', 'no-cost'
    ]
    gambling_keywords = [
        'bet', 'betting', 'wager', 'gamble', 'casino', 'poker',
        'slots', 'jackpot', 'roulette', 'blackjack', 'odds',
        'rummy', 'fantasy', 'dream11', 'my11', 'contest', 'league',
        'tournament', 'winning', 'cash-prize', 'real-money', 'earn-money',
        'play-win', 'prize-pool', 'join-contest', 'prediction',
        'mpl', 'winzo', 'paytm-games', 'ludo', 'carrom', 'chess-money',
        'skill-game', 'earn-playing', 'game-money', 'withdraw',
        '1xbet', 'betway', 'bet365', '10cric', 'fairbet', 'pure-win',
        'dafabet', 'parimatch', 'melbet'
    ]
    malware_keywords = [
        'download', 'exe', 'install', 'plugin', 'codec',
        'update-now', 'flash', 'java', 'activex', 'setup'
    ]
    piracy_keywords = [
        'crack', 'cracked', 'keygen', 'serial', 'patch', 'nulled',
        'repack', 'repacks', 'fitgirl', 'dodi', 'codex', 'skidrow',
        'torrent', 'pirate', 'warez', 'free-download', 'full-version',
        'activated', 'unlocked', 'premium-free', 'mod-apk', 'hacked'
    ]

    phishing_count  = sum(1 for kw in phishing_keywords  if kw in url_lower)
    financial_count = sum(1 for kw in financial_keywords if kw in url_lower)
    scam_count      = sum(1 for kw in scam_keywords      if kw in url_lower)
    gambling_count  = sum(1 for kw in gambling_keywords  if kw in url_lower or kw in domain_lower)
    malware_count   = sum(1 for kw in malware_keywords   if kw in url_lower)
    piracy_count    = sum(1 for kw in piracy_keywords    if kw in url_lower)

    if is_known_gambling:
        gambling_count += 3

    category_scores = {
        'Phishing':         phishing_count,
        'Financial Fraud':  financial_count,
        'Scam':             scam_count,
        'Gambling/Betting': gambling_count,
        'Malware':          malware_count,
        'Piracy':           piracy_count,
    }

    max_count = max(category_scores.values())

    if max_count == 0:
        risk_type  = 'Unknown'
        score      = 0
        type_hint  = 0
    else:
        risk_type = max(category_scores, key=category_scores.get)
        type_hint = TYPE_HINT_MAP.get(risk_type, 0)

        if risk_type == 'Gambling/Betting':
            score = min(gambling_count * 4, 18)
        elif risk_type == 'Piracy':
            score = piracy_count * 6
        else:
            score = max_count * 5

    return min(score, 25), risk_type, type_hint


def calculate_security_score(url):
    if not url.startswith('https://'):
        return 15
    return 0


def calculate_redirect_score(url):
    try:
        response = requests.get(
            url, timeout=2, allow_redirects=True,
            headers={'User-Agent': 'Mozilla/5.0'},
            verify=False
        )
        redirect_count = len(response.history)
        if redirect_count > 5:   return 10
        elif redirect_count > 3: return 7
        elif redirect_count > 1: return 4
        try:
            original_domain = urlparse(url).netloc
            final_domain    = urlparse(response.url).netloc
            if original_domain != final_domain:
                return 6
        except:
            pass
        return 0
    except:
        return 5


def extract_features(url):
    """
    Extract all features.
    NOW RETURNS 6 features: domain, url, keyword, security, redirect, type_hint
    type_hint is the new 6th feature that separates threat categories.
    """
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.split(':')[0] if ':' in parsed.netloc else parsed.netloc
        if not domain:
            return None

        domain_score   = calculate_domain_score(domain)
        url_score      = calculate_url_score(url)
        keyword_score, inferred_risk_type, type_hint = calculate_keyword_score_and_type(url, domain)
        security_score = calculate_security_score(url)
        redirect_score = calculate_redirect_score(url)

        total_score = domain_score + url_score + keyword_score + security_score + redirect_score

        is_trusted  = is_trusted_domain(domain)
        is_gambling = is_gambling_platform(domain) or inferred_risk_type == 'Gambling/Betting'

        return {
            'domain':            domain,
            'domain_score':      domain_score,
            'url_score':         url_score,
            'keyword_score':     keyword_score,
            'security_score':    security_score,
            'redirect_score':    redirect_score,
            'type_hint':         type_hint,          # ← NEW 6th feature
            'total_score':       min(total_score, 100),
            'inferred_risk_type': inferred_risk_type,
            'is_trusted':        is_trusted,
            'is_gambling':       is_gambling,
        }
    except Exception as e:
        print(f"⚠ Feature extraction error: {e}")
        return None


def load_models():
    models = [None, None, None]
    try:
        if RISK_MODEL_PATH.exists():
            models[0] = joblib.load(RISK_MODEL_PATH)
    except: pass
    try:
        if RISK_TYPE_MODEL_PATH.exists():
            models[1] = joblib.load(RISK_TYPE_MODEL_PATH)
    except: pass
    try:
        if ANOMALY_MODEL_PATH.exists():
            models[2] = joblib.load(ANOMALY_MODEL_PATH)
    except: pass
    return models


def get_feature_array(features):
    """
    Build the feature array passed to every ML model.
    6 features: [domain_score, url_score, keyword_score, security_score, redirect_score, type_hint]
    """
    return np.array([[
        features['domain_score'],
        features['url_score'],
        features['keyword_score'],
        features['security_score'],
        features['redirect_score'],
        features['type_hint'],        # ← 6th feature
    ]])


def train_models():
    print("\n" + "="*60)
    print("🔧 TRAINING MODELS (6-feature pipeline)")
    print("="*60)

    X, y_risk, y_type = get_training_data()

    if X is None or len(X) < MIN_SAMPLES_FOR_TRAINING:
        print(f"❌ Need {MIN_SAMPLES_FOR_TRAINING} samples (have: {len(X) if X else 0})")
        print("="*60 + "\n")
        return False

    print(f"✓ Samples: {len(X)}")
    X      = np.array(X)
    y_risk = np.array(y_risk)

    # ── Risk classifier ───────────────────────────────────────────────────────
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_risk, test_size=0.2, random_state=42
        )
        risk_model = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42, class_weight='balanced'
        )
        risk_model.fit(X_train, y_train)
        accuracy = risk_model.score(X_test, y_test)
        print(f"✓ Risk Model: {accuracy:.0%} accurate  (features={risk_model.n_features_in_})")
        joblib.dump(risk_model, RISK_MODEL_PATH)
    except Exception as e:
        print(f"❌ Risk model failed: {e}")

    # ── Type classifier ───────────────────────────────────────────────────────
    try:
        valid_indices = [i for i, t in enumerate(y_type)
                         if t and t not in ['Unknown', 'Safe']]
        if len(valid_indices) >= 10:
            X_type            = X[valid_indices]
            y_type_filtered   = np.array([y_type[i] for i in valid_indices])
            if len(np.unique(y_type_filtered)) >= 2:
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X_type, y_type_filtered, test_size=0.2, random_state=42
                )
                type_model = RandomForestClassifier(
                    n_estimators=200, max_depth=12, random_state=42
                )
                type_model.fit(X_tr, y_tr)
                accuracy_t = type_model.score(X_te, y_te)
                print(f"✓ Type Model: {accuracy_t:.0%} accurate  (features={type_model.n_features_in_})")
                joblib.dump(type_model, RISK_TYPE_MODEL_PATH)
    except Exception as e:
        print(f"❌ Type model failed: {e}")

    # ── Anomaly detector ──────────────────────────────────────────────────────
    try:
        anomaly_model = IsolationForest(
            n_estimators=100, contamination=0.1, random_state=42
        )
        anomaly_model.fit(X)
        print("✓ Anomaly Model: Trained")
        joblib.dump(anomaly_model, ANOMALY_MODEL_PATH)
    except Exception as e:
        print(f"❌ Anomaly model failed: {e}")

    print("="*60 + "\n")
    return True


def check_and_retrain():
    count = get_record_count()
    if count >= MIN_SAMPLES_FOR_TRAINING and not RISK_MODEL_PATH.exists():
        print(f"\n⚡ AUTO-TRAIN: {count} samples")
        return train_models()
    if count > 0 and count % RETRAIN_INTERVAL == 0:
        print(f"\n⚡ RETRAIN: {count} samples")
        return train_models()
    return False


def generate_risk_explanation(features, risk_type):
    if features.get('is_trusted'):
        return "Verified trusted domain"
    if risk_type == 'Gambling/Betting' or features.get('is_gambling'):
        warnings_list = []
        if features['total_score'] > 40:
            warnings_list.append("⚠️ Financial risk involved")
        else:
            warnings_list.append("Financial risk present")
        warnings_list.append("outcomes depend on probability")
        if features['keyword_score'] > 10:
            warnings_list.append("real money transactions involved")
        return ", ".join(warnings_list).capitalize()
    reasons = []
    if features['domain_score'] > 10:  reasons.append("suspicious domain")
    if features['keyword_score'] > 10: reasons.append(f"{risk_type.lower()} indicators")
    if features['security_score'] > 10: reasons.append("no HTTPS")
    if features['redirect_score'] > 5:  reasons.append("redirects")
    if features['url_score'] > 10:      reasons.append("suspicious URL structure")
    return ", ".join(reasons).capitalize() if reasons else "Low-level risk indicators"


def get_gambling_warning(features):
    if not (features.get('is_gambling') or
            features.get('inferred_risk_type') == 'Gambling/Betting'):
        return None
    score = features.get('total_score', features.get('total_score', 0))
    if score > 50:
        return """
⚠️  FINANCIAL RISK WARNING:
• Money loss is highly probable
• Outcomes are uncertain and depend on chance/skill
• Only use money you can afford to lose
• Gambling can be addictive - seek help if needed"""
    elif score > 30:
        return """
⚠️  FINANCIAL CAUTION:
• Real money transactions involved
• Risk of financial loss exists
• Probability of winning varies - no guarantees
• Set limits and gamble responsibly"""
    else:
        return """
ℹ️  ADVISORY:
• Platform involves real money gaming
• Financial risk is present
• Understand the odds before participating
• Play responsibly within your means"""


def analyze_url(url):
    """Main analysis function — now uses 6-feature pipeline."""
    print(f"\n{'='*60}")
    print(f"🔍 {url}")
    print(f"{'='*60}")

    # Check cache
    cached = get_cached_result(url)
    if cached:
        print("✓ CACHED")
        print("="*60 + "\n")
        return cached

    print("✗ Analyzing...")

    features = extract_features(url)
    if features is None:
        return {"error": "Invalid URL", "url": url}

    print(f"  Domain:     {features['domain']}")
    print(f"  Score:      {features['total_score']}/100")
    print(f"  Type hint:  {features['type_hint']} ({features['inferred_risk_type']})")

    risk_model, risk_type_model, anomaly_model = load_models()
    feature_array = get_feature_array(features)

    # ── Risk level prediction ─────────────────────────────────────────────────
    if features.get('is_trusted'):
        risk_label  = 0
        risk_level  = 'Low'
        risk_type   = 'Safe'
        confidence  = 95.0
    else:
        if risk_model:
            # Check feature count compatibility
            if risk_model.n_features_in_ == 6:
                risk_label   = int(risk_model.predict(feature_array)[0])
                probabilities = risk_model.predict_proba(feature_array)[0]
                confidence   = round(max(probabilities) * 100, 2)
            else:
                # Old 5-feature model — use fallback until retrained
                fa5 = np.array([[features['domain_score'], features['url_score'],
                                 features['keyword_score'], features['security_score'],
                                 features['redirect_score']]])
                risk_label   = int(risk_model.predict(fa5)[0])
                probabilities = risk_model.predict_proba(fa5)[0]
                confidence   = round(max(probabilities) * 100, 2)
        else:
            # Rule-based fallback
            if features.get('is_gambling'):
                risk_label = 1
                confidence = 70.0
            elif features['total_score'] > 60:
                risk_label = 2; confidence = 70.0
            elif features['total_score'] > 35:
                risk_label = 1; confidence = 65.0
            else:
                risk_label = 0; confidence = 60.0

        risk_map   = {0: 'Low', 1: 'Medium', 2: 'High', 3: 'Critical'}
        risk_level = risk_map.get(risk_label, 'Low')

        # ── Type prediction ───────────────────────────────────────────────────
        if risk_type_model:
            try:
                if risk_type_model.n_features_in_ == 6:
                    risk_type = risk_type_model.predict(feature_array)[0]
                else:
                    fa5 = np.array([[features['domain_score'], features['url_score'],
                                     features['keyword_score'], features['security_score'],
                                     features['redirect_score']]])
                    risk_type = risk_type_model.predict(fa5)[0]
            except:
                risk_type = features['inferred_risk_type']
        else:
            risk_type = features['inferred_risk_type']

    # ── Anomaly detection ─────────────────────────────────────────────────────
    is_anomaly = False
    if (anomaly_model is not None and
            not features.get('is_trusted') and
            not features.get('is_gambling')):
        try:
            if anomaly_model.n_features_in_ == 6:
                anomaly_pred = anomaly_model.predict(feature_array)[0]
            else:
                fa5 = np.array([[features['domain_score'], features['url_score'],
                                 features['keyword_score'], features['security_score'],
                                 features['redirect_score']]])
                anomaly_pred = anomaly_model.predict(fa5)[0]
            is_anomaly = (anomaly_pred == -1)
        except:
            pass

    # ── Severity ──────────────────────────────────────────────────────────────
    if features.get('is_gambling'):
        severity = int(35 + (features['total_score'] * 0.4) + (confidence * 0.2))
    else:
        severity = int((features['total_score'] * 0.7) + (confidence * 0.3))
    severity = min(severity, 100)

    why_risk         = generate_risk_explanation(features, risk_type)
    gambling_warning = get_gambling_warning(features)

    store_analysis(
        url, features['domain'], features, risk_label, risk_type,
        confidence, is_anomaly, severity, why_risk
    )
    print("✓ Stored")
    check_and_retrain()

    result = {
        'url':                url,
        'domain':             features['domain'],
        'domain_score':       features['domain_score'],
        'url_score':          features['url_score'],
        'keyword_score':      features['keyword_score'],
        'security_score':     features['security_score'],
        'redirect_score':     features['redirect_score'],
        'total_score':        features['total_score'],
        'risk_level':         risk_level,
        'risk_level_numeric': risk_label,
        'confidence_percent': confidence,
        'anomaly_detected':   is_anomaly,
        'risk_severity_index': severity,
        'why_risk':           why_risk,
        'risk_type':          risk_type,
        'gambling_warning':   gambling_warning,
        'cached':             False,
    }

    print("="*60 + "\n")
    return result


def display_result(result):
    print("\n" + "="*60)
    print("📊 RESULT")
    print("="*60)
    if 'error' in result:
        print(f"❌ {result['error']}")
        return
    print(f"\n🌐 {result['url']}")
    print(f"🏠 {result['domain']}")
    print(f"\n📈 SCORES:")
    print(f"  Domain:    {result['domain_score']}/25")
    print(f"  URL:       {result['url_score']}/25")
    print(f"  Keywords:  {result['keyword_score']}/25")
    print(f"  Security:  {result['security_score']}/15")
    print(f"  Redirects: {result['redirect_score']}/10")
    print(f"  ─────────────────")
    print(f"  TOTAL:     {result['total_score']}/100")
    print(f"\n🎯 ASSESSMENT:")
    print(f"  Risk:        {result['risk_level']}")
    print(f"  Type:        {result['risk_type']}")
    print(f"  Confidence:  {result['confidence_percent']:.0f}%")
    print(f"  Severity:    {result['risk_severity_index']}/100")
    print(f"\n💡 {result['why_risk']}")
    if result.get('gambling_warning'):
        print(result['gambling_warning'])
    print("="*60 + "\n")


def show_stats():
    print("\n" + "="*60)
    print("📊 STATS")
    print("="*60)
    count = get_record_count()
    risk_dist, type_dist = get_class_distribution()
    print(f"\n📂 Database: {count} URLs")
    print(f"  Low:    {risk_dist.get(0, 0)}")
    print(f"  Medium: {risk_dist.get(1, 0)}")
    print(f"  High:   {risk_dist.get(2, 0)}")
    if type_dist:
        print(f"\n🏷️  Types:")
        for rtype, cnt in sorted(type_dist.items(), key=lambda x: x[1], reverse=True)[:7]:
            print(f"  {rtype:20s}: {cnt}")
    risk_model, risk_type_model, anomaly_model = load_models()
    print(f"\n🤖 Models:")
    print(f"  Risk:    {'✓' if risk_model else '✗'}" +
          (f"  ({risk_model.n_features_in_} features)" if risk_model else ""))
    print(f"  Type:    {'✓' if risk_type_model else '✗'}" +
          (f"  ({risk_type_model.n_features_in_} features)" if risk_type_model else ""))
    print(f"  Anomaly: {'✓' if anomaly_model else '✗'}")
    print("="*60 + "\n")


if __name__ == "__main__":
    initialize_database()
    print("\n" + "="*60)
    print("🛡️  URL RISK ANALYZER v3.1")
    print("="*60)
    print("\nCommands: <url>  stats  train  exit")
    print("="*60)

    while True:
        try:
            user_input = input("\n>>> ").strip()
            if not user_input:    continue
            if user_input.lower() == 'exit':  print("Bye!"); break
            if user_input.lower() == 'stats': show_stats(); continue
            if user_input.lower() == 'train': train_models(); continue
            result = analyze_url(user_input)
            display_result(result)
        except KeyboardInterrupt:
            print("\n\nStopped"); break
        except Exception as e:
            print(f"❌ Error: {e}")