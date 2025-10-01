import re
from typing import Any, Dict, List

PUBLISHED_SPLIT = re.compile(r'(?=^Published:\s*)', flags=re.MULTILINE)

def split_records(text: str) -> List[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    return [p.strip() for p in PUBLISHED_SPLIT.split(text) if p.strip()]

def extract_summary_block(record: str) -> str:
    """
    Try to grab the text after a 'Summary:' (or 'Summary::') label until the next 'Published:' or end.
    Falls back to heuristic if no explicit Summary found.
    """
    # Normalize Summary label variants
    # Capture everything after Summary: until end or next 'Published:' marker
    m = re.search(
        r'(?:^|\n)Summary:{1,2}\s*(.*?)(?=\nPublished:|\Z)', 
        record, 
        flags=re.DOTALL | re.IGNORECASE
    )
    if m:
        summary = m.group(1).strip()
        # Clean common bracketed lists like "['Objectives:', ...]"
        summary = re.sub(r"\[.*?\]\s*", "", summary, flags=re.DOTALL)
        return summary if summary else ""

    # Heuristic fallback:
    # Take lines after Title: or first non-empty paragraph as a proxy
    lines = record.splitlines()
    try:
        title_idx = next(i for i, ln in enumerate(lines) if ln.strip().lower().startswith("title:"))
        # Take text after title until a blank line or metadata label
        body = "\n".join(lines[title_idx+1:]).strip()
    except StopIteration:
        body = record.strip()

    # Stop at common next metadata fields
    body = re.split(r'\n(?:Authors?:|Copyright|DOI:|PMID:|Affiliation|Published:)\b', body, maxsplit=1)[0].strip()
    # Trim overly long content; keep first paragraph
    para = body.split("\n\n")[0].strip()
    return para

def normalized_summaries(val: Any) -> List[str]:
    return [s for rec in split_records(val) if (s := extract_summary_block(rec))]

def normalize_google_to_snippets(val: Any) -> List[str]:
    if not isinstance(val, list):
        return []
    out = []
    for g in val:
        if isinstance(g, dict) and g.get("snippet"):
            out.append(str(g["snippet"]).strip())
    return out

def flatten_summaries(data: Dict[str, Any]) -> Dict[int, str]:
    items: List[str] = []
    items += normalized_summaries(data.get("pubmed"))
    items += normalize_google_to_snippets(data.get("google"))
    items += normalized_summaries(data.get("arxiv"))
    return items

# ---- Example with your input ----
input_data = {
    'pubmed': "Published: 2025-08-09\nTitle: Endovascular Treatment for Fungal Internal Carotid Artery Aneurysm.\nCopyright Information: Copyright © 2025, Irizato et al.\nSummary::\n['Aspergillus', 'Aspergillus fumigatus']\nFungal cerebral aneurysms, particularly those resulting from direct invasion by fungal sinusitis, are rare and often fatal when involving the cavernous segment of the internal carotid artery (ICA). We present a case of a ruptured fungal ICA aneurysm caused by  sinusitis, successfully treated with parent artery occlusion (PAO). In this case, an 80-year-old woman presented with right ptosis, facial pain, and cranial nerve III, IV, and VI palsies. Magnetic resonance imaging (MRI) and angiography revealed a spindle-shaped aneurysm in the cavernous segment of the right ICA with sphenoid sinus invasion. A balloon occlusion test demonstrated tolerance to ICA occlusion. Polymerase chain reaction analysis of sinus pus confirmed\xa0. Despite antifungal therapy and sinus irrigation, the aneurysm enlarged. While flow diversion was being planned, the aneurysm ruptured, causing massive epistaxis and shock. Emergent PAO using a double catheter technique was performed, preserving collateral flow via the anterior and posterior communicating arteries. Postoperatively, the patient had no new neurological deficits, with only residual oculomotor palsy. This \ncase highlights the importance of early balloon occlusion testing in the management of fungal ICA aneurysms because of their high risk of rupture. \nTight coil packing using a double catheter technique can minimize ischemic complications while preserving vital collateral circulation.\n\nPublished: 2025-07-09\nTitle: Superselective Unilateral Embolization of the Sphenopalatine Artery for Severe Posterior Epistaxis: A Prospective Study on the Safety and Efficacy.\nCopyright Information: \nSummary::\n['Objectives:', 'Methods:', 'Results:', 'Conclusions:']\nEpistaxis is a common condition affecting up to 60% of the population, with approximately 6% ",
    'google': [
        {'title': 'Posterior Epistaxis Nasal Pack - StatPearls - NCBI Bookshelf', 'link': 'https://www.ncbi.nlm.nih.gov/books/NBK576436/', 'snippet': 'Complications · Pain on insertion · Pain on removal [11] · Failure to achieve hemostasis · Rebleeding on removal · Otitis media due to Eustachian tube obstruction [\xa0...'},
        {'title': 'Posterior epistaxis: clinical features and acute complications', 'link': 'https://pubmed.ncbi.nlm.nih.gov/7741333/', 'snippet': 'Although posterior epistaxis is an uncommon otolaryngologic emergency, many patients experience clinically significant complications.'}
    ],
    'arxiv': "Published: 2015-12-22\nTitle: Data-dependent Posterior Propriety of Bayesian Beta-Binomial-Logit Model\nAuthors: Hyungsuk Tak, Carl N. Morris\nSummary: A Beta-Binomial-Logit model is a Beta-Binomial \nmodel with covariate\ninformation incorporated via a logistic regression. Posterior propriety of a\nBayesian Beta-Binomial-Logit model can be data-dependent for improper\nhyper-prior distributions. Various researchers in the literature have\nunknowingly used improper posterior distributions or have given incorrect\nstatements about posterior propriety because checking posterior propriety can\nbe challenging due to the complicated functional form of a Beta-Binomial-Logit\nmodel. We derive data-dependent necessary and sufficient conditions for\nposterior propriety within a class of hyper-prior distributions that encompass\nthose used in previous studies.\n\nPublished: 2023-01-15\nTitle: A Simple Proof of Posterior Robustness\nAuthors: Yasuyuki Hamura\nSummary: Conditions for Bayesian posterior robustness have been examined in recent\nliterature. However, many of the proofs seem to be long and complicated. In\nthis paper, we first summarize some basic lemmas that have been applied\nimplicitly or explicitly. Then, using them, we give a simple proof of posterior\nrobustness. Our sufficient condition is new and practically relevant."
}

#flat = flatten_summaries(input_data)
#documents = [flat[k] for k in sorted(flat)]
#print(flat)