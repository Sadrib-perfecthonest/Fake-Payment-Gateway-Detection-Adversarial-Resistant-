import pandas as pd
import random
from urllib.parse import urlparse
import os


def homoglyph_domain(url):
    p = urlparse(url)
    host = p.netloc
    
    host2 = host.replace('l','I').replace('o','0').replace('a','@')
    return url.replace(host, host2)

def add_https(url):
    if url.startswith('http://'):
        return url.replace('http://', 'https://')
    return url

def mimic_timing(row):
    
    return random.uniform(5.0, 15.0)

def age_spoof(row):
    
    return random.choice([15, 30, 60, 90])


def generate_variants(df, n_variants=5):
    adv_rows = []
    
    
    phish_targets = df[df['label'] == 1]
    
    
    required_cols = list(df.columns) + ['notes', 'num_redirects', 'https_flag', 
                                       'domain_age_days', 'time_to_confirm', 'page_dwell']
    
    for _, row in phish_targets.iterrows():
        for i in range(n_variants):
            r = row.copy()
            
            for col in ['notes', 'num_redirects', 'https_flag', 'domain_age_days', 'time_to_confirm', 'page_dwell']:
                r[col] = None 
                
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
                r['page_dwell'] = max(0.5, random.uniform(0.7, 1.5)) 
                r['notes'] = 'timing_mimic'
            elif attack == 'age':
                r['domain_age_days'] = age_spoof(row)
                r['notes'] = 'age_spoof'
            elif attack == 'redirects':
                r['num_redirects'] = random.randint(3, 7) 
                r['notes'] = 'more_redirects'
            elif attack == 'amount_mimic':
                
                r['TransactionAmount'] = random.choice([0.49, 1.99, 5.0, 9.99])
                r['notes'] = 'amount_mimic'
            
            adv_rows.append(r)
            
   
    adv_df = pd.DataFrame(adv_rows)
   
    adv_df = adv_df.reindex(columns=required_cols)
    return adv_df.fillna('') 

if __name__ == "__main__":
    
    os.makedirs('../processed', exist_ok=True)
    os.makedirs('../raw', exist_ok=True)
    
    input_path = "../raw/adverserial_legit.csv"
    output_path = "../processed/sessions_with_adv.csv"
    
    try:
    
        manual_df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: {input_path} not found. Please create it with the updated structure first.")
        exit()
        
    print(f"Generating adversarial variants from {len(manual_df[manual_df['label'] == 1])} targets...")
    
    
    adv_test_df = generate_variants(manual_df)
    
    
    adv_test_df.to_csv(output_path, index=False)
    
    print(f"Saved {len(adv_test_df)} adversarial test samples to {output_path}")