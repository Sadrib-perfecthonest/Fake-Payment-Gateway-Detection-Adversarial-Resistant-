import pandas as pd
import random
from urllib.parse import urlparse

def homoglyph_domain(url):
    p = urlparse(url)
    host = p.netloc
    host2 = host.replace('l','I').replace('o','0').replace('a','@')
    return url.replace(host, host2)

def add_https(url):
    if url.startswith('http://'):
        return url.replace('http://', 'https://')
    elif url.startswith('https://'):
        return url
    else:
        return 'https://' + url

def mimic_timing(row):
    return random.uniform(1.0, 12.0)

def age_spoof(row):
    return random.choice([30, 60, 365])

def generate_variants(df, n_variants=3):
    adv_rows = []
    for _, row in df.iterrows():
        if row['label'] == 1:
            for i in range(n_variants):
                r = row.copy()
                attack = random.choice(['homoglyph','https','timing','age','redirects','amount_mimic'])
                if attack == 'homoglyph':
                    r['url'] = homoglyph_domain(r['url'])
                    r['notes'] = 'homoglyph'
                elif attack == 'https':
                    r['url'] = add_https(r['url'])
                    r['https_flag'] = 1
                    r['notes'] = 'https_added'
                elif attack == 'timing':
                    r['time_to_confirm'] = mimic_timing(row)
                    r['page_dwell'] = max(0.5, row.get('page_dwell',1.0) * random.uniform(0.7,1.5))
                    r['notes'] = 'timing_mimic'
                elif attack == 'age':
                    r['domain_age_days'] = age_spoof(row)
                    r['notes'] = 'age_spoof'
                elif attack == 'redirects':
                    r['num_redirects'] = row.get('num_redirects',0) + random.randint(1,5)
                    r['notes'] = 'more_redirects'
                elif attack == 'amount_mimic':
                    r['amount'] = random.choice([5.0,9.99,12.0,19.99,29.99])
                    r['notes'] = 'amount_mimic'
                adv_rows.append(r)
    return pd.DataFrame(adv_rows)

if __name__ == "__main__":
    base = pd.read_csv('../Processed/sessions_clean.csv')
   
    phishing_rows = base[base['label']==1]
    adv = generate_variants(phishing_rows, n_variants=4)
    adv.to_csv('../Processed/sessions_with_adv.csv', index=False)
    print('Saved adversarial samples:', len(adv))
