import json
import pandas as pd
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

DATA_DIR    = Path("data")
STORE_PATH  = Path("vectorstore")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

STORE_PATH.mkdir(parents=True, exist_ok=True)

REQUIRED = [
    "dataset.csv",
    "symptom_Description.csv",
    "symptom_precaution.csv",
    "Symptom-severity.csv",
]


def load():
    for f in REQUIRED:
        if not (DATA_DIR / f).exists():
            raise FileNotFoundError(
                f"Missing {f} in ./data/ — run python get_data.py first"
            )
    disease_df     = pd.read_csv(DATA_DIR / "dataset.csv")
    description_df = pd.read_csv(DATA_DIR / "symptom_Description.csv")
    precaution_df  = pd.read_csv(DATA_DIR / "symptom_precaution.csv")
    severity_df    = pd.read_csv(DATA_DIR / "Symptom-severity.csv")

    for df in [disease_df, description_df, precaution_df, severity_df]:
        df.columns = df.columns.str.strip()

    return disease_df, description_df, precaution_df, severity_df


def group_symptoms(disease_df):
    sym_cols = [c for c in disease_df.columns if c.lower().startswith("symptom")]
    grouped = {}
    for _, row in disease_df.iterrows():
        disease = row["Disease"].strip()
        symptoms = [
            str(row[c]).strip().replace("_", " ")
            for c in sym_cols
            if pd.notna(row[c]) and str(row[c]).strip() not in ("", "nan")
        ]
        grouped.setdefault(disease, set()).update(symptoms)
    return {d: sorted(s) for d, s in grouped.items()}


def build_docs(disease_df, description_df, precaution_df, severity_df):
    symptoms_map = group_symptoms(disease_df)

    desc_map = dict(zip(
        description_df["Disease"].str.strip(),
        description_df["Description"].str.strip()
    ))

    prec_cols = [c for c in precaution_df.columns if c.lower().startswith("precaution")]
    prec_map = {}
    for _, row in precaution_df.iterrows():
        d = row["Disease"].strip()
        prec_map[d] = [
            str(row[c]).strip()
            for c in prec_cols
            if pd.notna(row[c]) and str(row[c]).strip() not in ("", "nan")
        ]

    sev_map = dict(zip(
        severity_df["Symptom"].str.strip().str.replace("_", " "),
        severity_df["weight"]
    ))

    docs = []
    for disease, symptoms in sorted(symptoms_map.items()):
        desc  = desc_map.get(disease, "No description available.")
        precs = prec_map.get(disease, [])
        sevs  = {s: sev_map.get(s, "?") for s in symptoms}

        
        docs.append(Document(
            page_content=(
                f"Disease: {disease}\n\n"
                f"Description: {desc}\n\n"
                f"Symptoms: {', '.join(symptoms)}\n\n"
                f"Precautions: {'; '.join(precs) or 'Consult a doctor.'}\n\n"
                f"Severity: {', '.join(f'{s}({w})' for s, w in sevs.items())}"
            ),
            metadata={"source": "Kaggle Health Dataset", "disease": disease, "type": "profile"},
        ))

        
        if symptoms:
            docs.append(Document(
                page_content=(
                    f"Symptoms that indicate {disease}:\n"
                    + "\n".join(f"  • {s} (severity {sevs.get(s, '?')})" for s in symptoms)
                    + f"\n\nThese symptoms may point to {disease}."
                ),
                metadata={"source": "Symptom Index", "disease": disease, "type": "symptoms"},
            ))

        
        if precs:
            docs.append(Document(
                page_content=(
                    f"Precautions for {disease}:\n"
                    + "\n".join(f"  {i+1}. {p}" for i, p in enumerate(precs))
                ),
                metadata={"source": "Precaution Records", "disease": disease, "type": "precautions"},
            ))

    return docs


def main():
    print("loading CSVs...")
    disease_df, description_df, precaution_df, severity_df = load()

    print("building documents...")
    docs = build_docs(disease_df, description_df, precaution_df, severity_df)
    print(f"  {len(docs)} chunks across {len(set(d.metadata['disease'] for d in docs))} diseases")

    print(f"embedding with {EMBED_MODEL}...")
    embedder = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    store = FAISS.from_documents(docs, embedder)
    store.save_local(str(STORE_PATH))

    # save metadata for the bot to display
    meta = {
        "total_docs": len(docs),
        "disease_count": len(set(d.metadata["disease"] for d in docs)),
        "embed_model": EMBED_MODEL,
    }
    (STORE_PATH / "metadata.json").write_text(json.dumps(meta, indent=2))

    print(f"\ndone — index saved to ./{STORE_PATH}/")
    print(f"  diseases : {meta['disease_count']}")
    print(f"  chunks   : {meta['total_docs']}")


if __name__ == "__main__":
    main()
