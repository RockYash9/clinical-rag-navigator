# Data Sources

The authoritative source registry is [`configs/sources.yaml`](../configs/sources.yaml) —
`scripts/ingest.py` reads it directly, so this file is documentation, not config.
Every source added to the YAML should get a row here too.

## Pilot batch (diabetes + hypertension)

| ID | Title | Tier | Condition | Organization | Year | License |
|---|---|---|---|---|---|---|
| `who_pen_2020` | WHO Package of Essential Noncommunicable (PEN) Disease Interventions for Primary Health Care | general_guidelines | — | WHO | 2020 | CC BY-NC-SA 3.0 IGO |
| `who_hypertension_2021` | WHO Guideline for the Pharmacological Treatment of Hypertension in Adults | disease_specific | hypertension | WHO | 2021 | CC BY-NC-SA 3.0 IGO |
| `who_emro_diabetes` | Diabetes Mellitus Guidelines | disease_specific | diabetes | WHO EMRO | — | WHO EMRO |
| `idf_type2_diabetes_2012` | Global Guideline for Type 2 Diabetes | disease_specific | diabetes | IDF | 2012 | IDF Clinical Guidelines Task Force |

## Research paper batch (diabetes focus)

| ID | Title | Format | Organization | Year | License |
|---|---|---|---|---|---|
| `baynes_2015_classification_diagnosis` | Classification, Pathophysiology, Diagnosis and Management of Diabetes Mellitus | PDF | J Diabetes Metab (Baynes HW) | 2015 | CC BY 4.0 |
| `ohiagu_2021_diabetic_complications` | Pathophysiology of Diabetes Mellitus and Its Complications: Metabolic Events and Control | PDF | Biomedical Research and Therapy | 2021 | CC BY 4.0 |
| `schroeder_endotext_t2d_pharm_mgmt` | Management of Type 2 Diabetes: Selecting Amongst Available Pharmacological Agents | HTML | Endotext / MDText.com | 2026 (last updated) | CC BY-NC-ND |

`schroeder_endotext_t2d_pharm_mgmt` is the first HTML-sourced document in the
corpus — `source_type: html` in sources.yaml routes it through
`download_html()` / `extract_text_from_html()` instead of the PDF path.
HTML extraction is a generic heuristic (strip nav/header/footer, prefer a
`<main>`/`<article>` container, fall back to `<body>`) rather than
site-specific scraping, so it's worth spot-checking the extracted chunks
for leftover site-navigation boilerplate after a fresh ingest.

## Known issue: pmc.ncbi.nlm.nih.gov blocks automated requests (reCAPTCHA)

Unlike `ncbi.nlm.nih.gov/books` (Bookshelf), the newer PMC article domain
(`pmc.ncbi.nlm.nih.gov`) returns a reCAPTCHA challenge page to automated
requests — same class of problem as the original iris.who.int issue.
PMC articles were avoided in this batch for that reason; if a specific PMC
article is needed later, it'll need `manual_download: true`.

## Other free sources worth adding later

| Source | Access | Notes |
|---|---|---|
| PubMed / PMC | Free, public API (E-utilities) | Open-access subset only — check license per article |
| NICE (UK) | Free (most guidance) | Some content requires registration |
| CDC | Free, public | Guidance documents, MMWR reports |
| StatPearls (via PMC) | Free, open access | Clinical reference articles |

## Known issue: iris.who.int blocks automated downloads

`who_pen_2020` is hosted on `iris.who.int`, which returns a small HTML page
instead of the PDF for non-browser requests (confirmed both via `requests`
and via direct fetch tooling — not a code bug on our end). No full-text
mirror was found elsewhere. Workaround, already reflected in
`configs/sources.yaml` (`manual_download: true`):

1. Download the PDF manually in a browser from
   https://www.who.int/publications/i/item/9789240009226
2. Save it as `data/raw/general_guidelines/who_pen_2020.pdf`
3. Re-run `python scripts/ingest.py` — `download_pdf()` skips sources whose
   file already exists, so it'll be picked up and chunked normally.

`who_hypertension_2021` had the same problem on `apps.who.int`; it's since
been repointed to an NCBI Bookshelf mirror of the same official WHO text,
which doesn't block automated requests.

## Process notes

- Run `python scripts/ingest.py` to download the pilot batch into `data/raw/<tier>/`
  and produce `data/processed/chunks.jsonl`.
- `download_pdf()` now validates the response actually starts with the `%PDF`
  magic bytes before saving — a source that returns HTML (bot-blocked, moved,
  paywalled) now fails with a clear `SourceUnavailableError` instead of
  silently saving garbage that breaks later during text extraction.
- Downloaded PDFs and processed chunks are gitignored — only the source list
  and code are version-controlled. Anyone cloning the repo regenerates the
  corpus locally by running ingestion.
- Guidelines get revised periodically — when a source is updated upstream,
  bump a `version` field in `sources.yaml` (not yet added) and re-ingest
  rather than silently overwriting.
