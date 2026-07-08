# 2. LITERATURE REVIEW

## 2.1 Existing Approaches

The challenge of phishing detection has been a focal point of cybersecurity research for over two decades. Historically, the methodologies developed to combat phishing have evolved through three distinct generational approaches:

### 2.1.1 List-Based Approaches (Blacklisting & Whitelisting)
The earliest and most widely adopted method for phishing mitigation is the use of blacklists. Systems such as Google Safe Browsing, PhishTank, and Microsoft SmartScreen rely on massive databases of known, verified malicious URLs. 
- **Mechanism:** When a user attempts to navigate to a URL, the system queries the database. If the URL is present, access is blocked. 
- **Advantages:** Blacklists are computationally lightweight and provide near 100% precision for known threats (zero false positives for verified entries).
- **Limitations:** They are inherently reactive. A phishing site must be detected, analyzed, verified, and distributed to the blacklist before protection is active. During this window, users are completely vulnerable.

### 2.1.2 Heuristic and Rule-Based Approaches
To address the reactive nature of blacklists, heuristic systems were developed. These systems evaluate URLs and webpage content against a predefined set of human-engineered rules.
- **Mechanism:** A rule might dictate: *“If the URL contains an IP address instead of a domain name, and the page contains a password input field, flag as suspicious.”*
- **Advantages:** Heuristics can proactively catch some newly generated phishing pages without waiting for database updates.
- **Limitations:** Attackers easily bypass static rules. Furthermore, as rules become more complex to catch sophisticated attacks, the rate of false positives on legitimate, complex web applications rises dramatically.

### 2.1.3 Single-Model Machine Learning Approaches
With the advent of data science, researchers began applying individual machine learning algorithms—such as Support Vector Machines (SVM), Naive Bayes, or standalone Decision Trees—to phishing detection.
- **Mechanism:** A model is trained on a dataset of URLs. The features used are typically limited to simple lexical properties (length of URL, number of hyphens, presence of '@' symbol).
- **Advantages:** These models generalize better than static heuristics and can predict unseen URLs with moderate accuracy.
- **Limitations:** Most academic models rely on shallow, 10-to-20 dimension lexical feature sets. They lack the contextual depth required to differentiate between a legitimate login portal hosted on a complex CDN and a sophisticated phishing clone. Furthermore, single models struggle with the extreme variance and adversarial nature of phishing data.

## 2.2 Limitations of Current Systems

Despite significant advancements, contemporary phishing detection solutions deployed in production environments suffer from several critical, systemic limitations that PhishGuard aims to resolve:

1. **The "Zero-Day" Ephemeral Domain Problem:**
   The primary failure of modern systems is their inability to combat ephemeral phishing. Attackers now utilize automated scripts to purchase cheap domains, launch a cloned banking site, send out millions of SMS/Email lures, harvest credentials, and burn the domain down within 4 to 8 hours. Traditional scanners that rely on domain age or WHOIS reputation often fail here because the domain has no negative history.

2. **Inadequate Feature Depth (Lexical Blindness):**
   Many current systems still rely too heavily on the URL string itself. Threat actors easily bypass lexical analysis by utilizing URL shorteners (e.g., bit.ly), legitimate cloud hosting services (e.g., AWS S3 buckets, Google Forms), or compromised benign WordPress websites. When a phishing page is hosted on a legitimate infrastructure, the URL features appear perfectly safe.

3. **Vulnerability to Visual Spoofing and Obfuscation:**
   Current structural scanners search for specific HTML tags or known malicious JavaScript strings. Attackers counter this by heavily obfuscating their code, using base64 encoding, or dynamically pulling the malicious payload from an external server only after the page loads. Furthermore, some phishing sites render the login form as an interactive image map or overlay to bypass text-based scrapers completely.

4. **The "Black Box" Trust Deficit:**
   While advanced Deep Learning systems (like Deep Neural Networks) have been proposed in academia with high accuracy claims, they are rarely deployed in enterprise SOC environments. This is because they operate as "black boxes." If a Deep Learning model flags a crucial internal corporate portal as "Malicious," security analysts cannot determine *why* the model made that decision. Without explainability, it is impossible to audit the model for bias or to quickly resolve false positives, leading to the system being disabled entirely.

5. **Computational Latency in Real-Time Applications:**
   Academic models that do extract deep features (such as full page rendering or deep NLP analysis) are often too slow for production. Analyzing a single URL might take several seconds, which is unacceptable for a proxy server or browser extension that must evaluate dozens of URLs per second as a user browses the web.

PhishGuard was specifically architected to address these five distinct limitations by fusing deep, multi-dimensional feature extraction with ultra-fast ensemble models and mathematically rigorous explainability.
