# Data Sources

Tracks where corpus documents come from. Every source added here should also
be reflected in chunk metadata during ingestion (source name, tier, retrieval
date, license/access terms).

## General guideline sources (free access)

| Source | Access | Notes |
|---|---|---|
| WHO Guidelines | Free, public | Published guideline PDFs |
| NICE (UK) | Free (most guidance) | Some content requires registration |
| CDC | Free, public | Guidance documents, MMWR reports |

## Disease-specific / research sources (free access)

| Source | Access | Notes |
|---|---|---|
| PubMed / PMC | Free, public API (E-utilities) | Open-access subset only — check license per article |
| StatPearls (via PMC) | Free, open access | Clinical reference articles |
| PLOS Medicine | Free, open access journal | |
| BMJ Open | Free, open access | |

## To be filled in as corpus is built

- [ ] Finalize initial disease list for the disease-specific tier
- [ ] Document per-source scraping/download method
- [ ] Track document versions (guidelines get revised — need a re-ingestion process)
